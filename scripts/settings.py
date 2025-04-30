from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
LLVM_DIR = THIS_DIR.parent
TOOLCHAIN_DIR = THIS_DIR.joinpath('cmake')
