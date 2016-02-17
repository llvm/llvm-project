======================
Using Polly with Clang
======================

This documentation discusses how Polly can be used in Clang to automatically
optimize C/C++ code during compilation.


.. warning::

  Warning: clang/LLVM/Polly need to be in sync (compiled from the same SVN
  revision).

Make Polly available from Clang
===============================

By default Polly is configured as a shared library plugin that is loaded in
tools like clang, opt, and bugpoint when they start their execution. By loading
Polly into clang (or opt) the Polly options become automatically available. You
can load Polly either by adding the relevant commands to the CPPFLAGS or by
creating an alias.

.. code-block:: console

  $ export CPPFLAGS="-Xclang -load -Xclang ${POLLY_BUILD_DIR}/lib/LLVMPolly.so"

or

.. code-block:: console

  $ alias pollycc clang -Xclang -load -Xclang ${POLLY_BUILD_DIR}/lib/LLVMPolly.so

To avoid having to load Polly in the tools, Polly can optionally be configured
with cmake to be statically linked in the tools:

.. code-block:: console

  $ cmake -D LINK_POLLY_INTO_TOOLS:Bool=ON

Optimizing with Polly
=====================

Optimizing with Polly is as easy as adding -O3 -mllvm -polly to your compiler
flags (Polly is only available at -O3).

.. code-block:: console

  pollycc -O3 -mllvm -polly file.c

Automatic OpenMP code generation
================================

To automatically detect parallel loops and generate OpenMP code for them you
also need to add -mllvm -polly-parallel -lgomp to your CFLAGS.

.. code-block:: console

  pollycc -O3 -mllvm -polly -mllvm -polly-parallel -lgomp file.c

Automatic Vector code generation
================================

Automatic vector code generation can be enabled by adding -mllvm
-polly-vectorizer=stripmine to your CFLAGS.

.. code-block:: console

  pollycc -O3 -mllvm -polly -mllvm -polly-vectorizer=stripmine file.c

Extract a preoptimized LLVM-IR file
===================================

Often it is useful to derive from a C-file the LLVM-IR code that is actually
optimized by Polly. Normally the LLVM-IR is automatically generated from the C
code by first lowering C to LLVM-IR (clang) and by subsequently applying a set
of preparing transformations on the LLVM-IR. To get the LLVM-IR after the
preparing transformations have been applied run Polly with '-O0'.

.. code-block:: console

  pollycc -O0 -mllvm -polly -S -emit-llvm file.c

Further options
===============
Polly supports further options that are mainly useful for the development or the
analysis of Polly. The relevant options can be added to clang by appending
-mllvm -option-name to the CFLAGS or the clang command line.

Limit Polly to a single function
--------------------------------

To limit the execution of Polly to a single function, use the option
-polly-only-func=functionname.

Disable LLVM-IR generation
--------------------------

Polly normally regenerates LLVM-IR from the Polyhedral representation. To only
see the effects of the preparing transformation, but to disable Polly code
generation add the option polly-no-codegen.

Graphical view of the SCoPs
---------------------------
Polly can use graphviz to show the SCoPs it detects in a program. The relevant
options are -polly-show, -polly-show-only, -polly-dot and -polly-dot-only. The
'show' options automatically run dotty or another graphviz viewer to show the
scops graphically. The 'dot' options store for each function a dot file that
highlights the detected SCoPs. If 'only' is appended at the end of the option,
the basic blocks are shown without the statements the contain.

Change/Disable the Optimizer
----------------------------

Polly uses by default the isl scheduling optimizer. The isl optimizer optimizes
for data-locality and parallelism using the Pluto algorithm. For research it is
also possible to run PoCC as external optimizer. PoCC provides access to the
original Pluto implementation. To use PoCC add -polly-optimizer=pocc to the
command line (only available if Polly was compiled with scoplib support)
[removed after LLVM 3.4.2]. To disable the optimizer entirely use the option
-polly-optimizer=none.

Disable tiling in the optimizer
-------------------------------

By default both optimizers perform tiling, if possible. In case this is not
wanted the option -polly-tiling=false can be used to disable it. (This option
disables tiling for both optimizers).

Ignore possible aliasing
------------------------

By default we only detect scops, if we can prove that the different array
bases can not alias. This is correct do if we optimize automatically. However,
without special user annotations like 'restrict' we can often not prove that
no aliasing is possible. In case the user knows no aliasing can happen in the
code the -polly-ignore-aliasing can be used to disable the check for possible
aliasing.

Import / Export
---------------

The flags -polly-import and -polly-export allow the export and reimport of the
polyhedral representation. By exporting, modifying and reimporting the
polyhedral representation externally calculated transformations can be
applied. This enables external optimizers or the manual optimization of
specific SCoPs. 
