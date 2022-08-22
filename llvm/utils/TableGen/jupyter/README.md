A Jupyter kernel for TableGen (llvm-tblgen)

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

If you just want to see the results of the notebook, you can read
`LLVM_Tablegen.md` instead.

`llvm-tblgen` is expected to be either in the `PATH` or you can set
the environment variable `LLVM_TBLGEN_EXECUTABLE` to point to it directly.

To run the kernel's doctests do:

```shell
    python3 tablegen_kernel/kernel.py
```
