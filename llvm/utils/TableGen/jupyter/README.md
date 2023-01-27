# Jupyter Tools for TableGen

This folder contains notebooks relating to TableGen and a Jupyter kernel for
TableGen.

## Notebooks

[LLVM_TableGen.ipynb](LLVM_TableGen.ipynb) - A demo of the kernel's capabilities.

[sql_query_backend.ipynb](sql_query_backend.ipynb) - How to write a backend using
JSON output and Python.

Notebooks can be viewed in browser on Github or downloaded and run locally. If
that is not possible, there are Markdown versions next to the notebook files.

## TableGen Kernel

To use the kernel, first install it into jupyter:

    python3 -m tablegen_kernel.install

Then put this folder on your PYTHONPATH so jupyter can find it:

```shell
    export PYTHONPATH=$PYTHONPATH:<path to this dir>
```

Then run one of:

```shell
    jupyter notebook
    # Then in the notebook interface, select 'LLVM TableGen' from the 'New' menu.

    # To run the example notebook in this folder.
    jupyter notebook LLVM_Tablegen.ipynb

    # To use the kernel from the command line.
    jupyter console --kernel tablegen
```

`llvm-tblgen` is expected to be either in the `PATH` or you can set
the environment variable `LLVM_TBLGEN_EXECUTABLE` to point to it directly.

To run the kernel's doctests do:

```shell
    python3 tablegen_kernel/kernel.py
```
