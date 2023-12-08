======================
Using Polly with Clang
======================

This documentation discusses how Polly can be used in Clang to automatically
optimize C/C++ code during compilation.


.. warning::

  Warning: clang/LLVM/Polly need to be in sync (compiled from the same
  revision).

Make Polly available from Clang
===============================

Polly is available through clang, opt, and bugpoint, if Polly was checked out
into tools/polly before compilation. No further configuration is needed.

Optimizing with Polly
=====================

Optimizing with Polly is as easy as adding -O3 -mllvm -polly to your compiler
flags (Polly is not available unless optimizations are enabled, such as
-O1,-O2,-O3; Optimizing for size with -Os or -Oz is not recommended).

.. code-block:: console

  clang -O3 -mllvm -polly file.c

Automatic OpenMP code generation
================================

To automatically detect parallel loops and generate OpenMP code for them you
also need to add -mllvm -polly-parallel -lgomp to your CFLAGS.

.. code-block:: console

  clang -O3 -mllvm -polly -mllvm -polly-parallel -lgomp file.c

Switching the OpenMP backend
----------------------------

The following CL switch allows to choose Polly's OpenMP-backend:

       -polly-omp-backend[=BACKEND]
              choose the OpenMP backend; BACKEND can be 'GNU' (the default) or 'LLVM';

The OpenMP backends can be further influenced using the following CL switches:


       -polly-num-threads[=NUM]
              set the number of threads to use; NUM may be any positive integer (default: 0, which equals automatic/OMP runtime);

       -polly-scheduling[=SCHED]
              set the OpenMP scheduling type; SCHED can be 'static', 'dynamic', 'guided' or 'runtime' (the default);

       -polly-scheduling-chunksize[=CHUNK]
              set the chunksize (for the selected scheduling type); CHUNK may be any strictly positive integer (otherwise it will default to 1);

Note that at the time of writing, the GNU backend may only use the
`polly-num-threads` and `polly-scheduling` switches, where the latter also has
to be set to "runtime".

Example: Use alternative backend with dynamic scheduling, four threads and
chunksize of one (additional switches).

.. code-block:: console

  -mllvm -polly-omp-backend=LLVM -mllvm -polly-num-threads=4
  -mllvm -polly-scheduling=dynamic -mllvm -polly-scheduling-chunksize=1

Automatic Vector code generation
================================

Automatic vector code generation can be enabled by adding -mllvm
-polly-vectorizer=stripmine to your CFLAGS.

.. code-block:: console

  clang -O3 -mllvm -polly -mllvm -polly-vectorizer=stripmine file.c

Isolate the Polly passes
========================

Polly's analysis and transformation passes are run with many other
passes of the pass manager's pipeline.  Some of passes that run before
Polly are essential for its working, for instance the canonicalization
of loop.  Therefore Polly is unable to optimize code straight out of
clang's -O0 output.

To get the LLVM-IR that Polly sees in the optimization pipeline, use the
command:

.. code-block:: console

  clang file.c -c -O3 -mllvm -polly -mllvm -polly-dump-before-file=before-polly.ll

This writes a file 'before-polly.ll' containing the LLVM-IR as passed to
polly, after SSA transformation, loop canonicalization, inlining and
other passes.

Thereafter, any Polly pass can be run over 'before-polly.ll' using the
'opt' tool.  To found out which Polly passes are active in the standard
pipeline, see the output of

.. code-block:: console

  clang file.c -c -O3 -mllvm -polly -mllvm -debug-pass=Arguments

The Polly's passes are those between '-polly-detect' and
'-polly-codegen'. Analysis passes can be omitted.  At the time of this
writing, the default Polly pass pipeline is:

.. code-block:: console

  opt before-polly.ll -polly-simplify -polly-optree -polly-delicm -polly-simplify -polly-prune-unprofitable -polly-opt-isl -polly-codegen

Note that this uses LLVM's old/legacy pass manager.

For completeness, here are some other methods that generates IR
suitable for processing with Polly from C/C++/Objective C source code.
The previous method is the recommended one.

The following generates unoptimized LLVM-IR ('-O0', which is the
default) and runs the canonicalizing passes on it
('-polly-canonicalize'). This does /not/ include all the passes that run
before Polly in the default pass pipeline.  The '-disable-O0-optnone'
option is required because otherwise clang adds an 'optnone' attribute
to all functions such that it is skipped by most optimization passes.
This is meant to stop LTO builds to optimize these functions in the
linking phase anyway.

.. code-block:: console

  clang file.c -c -O0 -Xclang -disable-O0-optnone -emit-llvm -S -o - | opt -polly-canonicalize -S

The option '-disable-llvm-passes' disables all LLVM passes, even those
that run at -O0.  Passing -O1 (or any optimization level other than -O0)
avoids that the 'optnone' attribute is added.

.. code-block:: console

  clang file.c -c -O1 -Xclang -disable-llvm-passes -emit-llvm -S -o - | opt -polly-canonicalize -S

As another alternative, Polly can be pushed in front of the pass
pipeline, and then its output dumped.  This implicitly runs the
'-polly-canonicalize' passes.

.. code-block:: console

  clang file.c -c -O3 -mllvm -polly -mllvm -polly-position=early -mllvm -polly-dump-before-file=before-polly.ll

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
for data-locality and parallelism using the Pluto algorithm.
To disable the optimizer entirely use the option -polly-optimizer=none.

Disable tiling in the optimizer
-------------------------------

By default both optimizers perform tiling, if possible. In case this is not
wanted the option -polly-tiling=false can be used to disable it. (This option
disables tiling for both optimizers).

Import / Export
---------------

The flags -polly-import and -polly-export allow the export and reimport of the
polyhedral representation. By exporting, modifying and reimporting the
polyhedral representation externally calculated transformations can be
applied. This enables external optimizers or the manual optimization of
specific SCoPs.

Viewing Polly Diagnostics with opt-viewer
-----------------------------------------

The flag -fsave-optimization-record will generate .opt.yaml files when compiling
your program. These yaml files contain information about each emitted remark.
Ensure that you have Python 2.7 with PyYaml and Pygments Python Packages.
To run opt-viewer:

.. code-block:: console

   llvm/tools/opt-viewer/opt-viewer.py -source-dir /path/to/program/src/ \
      /path/to/program/src/foo.opt.yaml \
      /path/to/program/src/bar.opt.yaml \
      -o ./output

Include all yaml files (use \*.opt.yaml when specifying which yaml files to view)
to view all diagnostics from your program in opt-viewer. Compile with `PGO
<https://clang.llvm.org/docs/UsersManual.html#profiling-with-instrumentation>`_ to view
Hotness information in opt-viewer. Resulting html files can be viewed in an internet browser.
