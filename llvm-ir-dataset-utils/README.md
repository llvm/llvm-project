<h1 align='center'>LLVM-IR Dataset Utilities</h1>

This repository contains utilities to construct large LLVM IR datasets from
multiple sources.

## Getting Started

To get started with the dataset construction utilities, we'd suggest to use the packaged [pipenv](https://pipenv.pypa.io) to isolate the Python from your system isolation or other environments. To install, you then have to

```bash
pipenv install
```

or if you seek to utilize the packaged lockfile

```bash
pipenv sync
```

After that you are ready to activate the environment, and install the dataset construction utilities into it

```bash
pipenv shell && pip install .
```

In case you want to develop the package, this becomes

```bash
pipenv shell && pip install -e .
```

## Creating First Data

To create your first small batch of IR data you then have to run from the root directory of the package

```bash
python3 ./llvm_ir_dataset_utils/tools/corpus_from_description.py \
  --source_dir=/path/to/store/dataset/to/source \
  --corpus_dir=/path/to/store/dataset/to/corpus \
  --build_dir=/path/to/store/dataset/to/build \
  --corpus_description=./corpus_descriptions_test/manual_tree.json
```

> Beware! You'll need to have a version of `llvm-objcopy` on your `$PATH`. If you are missing `llvm-objcopy`, an easy way to obtain it is by downloading an llvm-release from either your preferred package channel such as `apt`, `dnf` or `pacman`, or build llvm from [source](https://github.com/llvm/llvm-project) where only the LLVM project itself needs to be enabled during the build, i.e. `-DLLVM_ENABLE_PROJECTS="llvm"`.

You'll then receive a set of `.bc` files in `/path/to/store/dataset/to/corpus/tree`, which you can convert with `llvm-dis` into LLVM-IR, i.e. from inside of the folder

```bash
llvm-dis *.bc
```

> Last steps into the dataloader to be described here.

## Corpus Description

> Basics of the corpus description to be outlined here to easily enable someone to point the package at a new source.

## IR Sources

The package contains a number of builders to target the LLVM-based languages, and extract IR:

- Individual projects (C/C++)
- Rust crates
- Spack packages
- Autoconf
- Cmake
- Julia packages
- Swift packages
