import os
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import shutil


def get_breakhis_loader(root_dir, magnification='40X', batch_size=32, shuffle=True, num_workers=2):
    """
    Loads BreakHis dataset filtered by magnification level.
    
    Arg:
        root_dir (str): Path to 'breast' folder.
        magnification (str): One of 40X, 100X, 200X, 400X
        batch_size Batch size for DataLoader
        shuffle (bool): Shuffle data.
        num_workers (int): Number of subprocesses.
    
    Returns:
        DataLoader
    """

    magnification = magnification.upper()
    assert magnification in ['40X', '100X', '200X', '400X'], "Magnification must be 40X, 100X, 200X, 400X"

    root = Path(root_dir)
    image_path = []

    for label in ['benign', 'malignant']:
        base = root / label / 'SOB'
        for subtype in os.listdir(base):
            subtype_path = base / subtype
            for case in os.listdir(subtype_path):
                mag_path = subtype_path / case / magnification
                if mag_path.exists():
                    image_path.append(mag_path)

    tem_root = Path("data/tmp_breakhis")
    if tem_root.exists():
        shutil.rmtree(tem_root)
    tem_root.mkdir(parents=True)

    for path in image_path:
        label = 'benign' if 'benign' in str(path) else 'malignant'
        label_dir = tem_root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        for img_file in os.listdir(path):
            src = path / img_file
            dst = label_dir / f"{path.name}_{img_file}"
            try:
                if src.exists() and src.suffix.lower() == '.png':
                    if not dst.exists():
                        shutil.copyfile(src, dst)
            except Exception as e:
                print(f"[SKIPPED] Could not copy: {src} → {dst}. Reason: {e}")


    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(tem_root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader








