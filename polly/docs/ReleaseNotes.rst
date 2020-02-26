=============================
Release Notes 10.0 (upcoming)
=============================

In Polly 10 the following important changes have been incorporated.

.. warning::

  These releaes notes are for the next release of Polly and describe
  the new features that have recently been committed to our development
  branch.

Statically Linking of Polly
===========================

The mechanism that Polly uses to link itself statically into the opt, bugpoint and clang executables has been generalized such that it can be used by other pass plugins. An example plugin "Bye" has been added to illustate the mechanism. A consequence of this change is that Polly, like the "Bye" plugin, by default is not linked statically into aforementioned executables anymore.

If Polly is not available, the executable will report an unkown argument `-polly`, such as

.. code-block:: console

    $ clang -mllvm -polly -x c -
    clang (LLVM option parsing): Unknown command line argument '-polly'.  Try: 'clang (LLVM option parsing) --help'
    clang (LLVM option parsing): Did you mean '--color'?

.. code-block:: console

    $ opt -polly
    opt: for the -o option: may not occur within a group!
    opt: Unknown command line argument '-polly'.  Try: 'opt --help'
    opt: Did you mean '-o'?

Polly can be made available using the following methods.

- Configure LLVM/Clang with the CMake options LLVM_POLLY_LINK_INTO_TOOLS=ON and LLVM_ENABLE_PROJECTS=polly.

  .. code-block:: console

    $ cmake -DLLVM_POLLY_LINK_INTO_TOOLS=ON -DLLVM_ENABLE_PROJECTS=clang;polly ...

  In future versions, LLVM_POLLY_LINK_INTO_TOOLS=ON will be default again if Polly has been enabled.

- Use the `-load` option to load the Polly module.

  .. code-block:: console

    $ clang -Xclang -load -Xclang path/to/LLVMPolly.so ...

  .. code-block:: console

    $ opt -load path/to/LLVMPolly.so ...

  The LLVMPolly.so module can be found in the `lib/` directory of the build or install-prefix directory.
