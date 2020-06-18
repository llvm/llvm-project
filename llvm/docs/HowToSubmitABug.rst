================================
How to submit an LLVM bug report
================================

Introduction - Got bugs?
========================


If you're working with LLVM and run into a bug, we definitely want to know
about it.  This document describes what you can do to increase the odds of
getting it fixed quickly.

Basically you have to do two things at a minimum.  First, decide whether
the bug `crashes the compiler`_ (or an LLVM pass), or if the
compiler is `miscompiling`_ the program (i.e., the
compiler successfully produces an executable, but it doesn't run right).
Based on what type of bug it is, follow the instructions in the linked
section to narrow down the bug so that the person who fixes it will be able
to find the problem more easily.

Once you have a reduced test-case, go to `the LLVM Bug Tracking System
<https://bugs.llvm.org/enter_bug.cgi>`_ and fill out the form with the
necessary details (note that you don't need to pick a category, just use
the "new-bugs" category if you're not sure).  The bug description should
contain the following information:

* All information necessary to reproduce the problem.
* The reduced test-case that triggers the bug.
* The location where you obtained LLVM (if not from our Git
  repository).

Thanks for helping us make LLVM better!

.. _crashes the compiler:

Crashing Bugs
=============

More often than not, bugs in the compiler cause it to crash---often due to
an assertion failure of some sort. The most important piece of the puzzle
is to figure out if it is crashing in the Clang front-end or if it is one of
the LLVM libraries (e.g. the optimizer or code generator) that has
problems.

To figure out which component is crashing (the front-end, optimizer or code
generator), run the ``clang`` command line as you were when the crash
occurred, but with the following extra command line options:

* ``-O0 -emit-llvm``: If ``clang`` still crashes when passed these
  options (which disable the optimizer and code generator), then the crash
  is in the front-end.  Jump ahead to the section on :ref:`front-end bugs
  <front-end>`.

* ``-emit-llvm``: If ``clang`` crashes with this option (which disables
  the code generator), you found an optimizer bug.  Jump ahead to
  `compile-time optimization bugs`_.

* Otherwise, you have a code generator crash. Jump ahead to `code
  generator bugs`_.

.. _front-end bug:
.. _front-end:

Front-end bugs
--------------

If the problem is in the front-end, you should re-run the same ``clang``
command that resulted in the crash, but add the ``-save-temps`` option.
The compiler will crash again, but it will leave behind a ``foo.i`` file
(containing preprocessed C source code) and possibly ``foo.s`` for each
compiled ``foo.c`` file. Send us the ``foo.i`` file, along with the options
you passed to ``clang``, and a brief description of the error it caused.

The `delta <http://delta.tigris.org/>`_ tool helps to reduce the
preprocessed file down to the smallest amount of code that still replicates
the problem. You're encouraged to use delta to reduce the code to make the
developers' lives easier. `This website
<http://gcc.gnu.org/wiki/A_guide_to_testcase_reduction>`_ has instructions
on the best way to use delta.

.. _compile-time optimization bugs:

Compile-time optimization bugs
------------------------------

If you find that a bug crashes in the optimizer, compile your test-case to a
``.bc`` file by passing "``-emit-llvm -O1 -Xclang -disable-llvm-passes -c -o
foo.bc``".  Then run:

.. code-block:: bash

   opt -O3 -debug-pass=Arguments foo.bc -disable-output

This command should do two things: it should print out a list of passes, and
then it should crash in the same way as clang.  If it doesn't crash, please
follow the instructions for a `front-end bug`_.

If this does crash, then you should be able to debug this with the following
bugpoint command:

.. code-block:: bash

   bugpoint foo.bc <list of passes printed by opt>

Please run this, then file a bug with the instructions and reduced .bc
files that bugpoint emits.  If something goes wrong with bugpoint, please
submit the "foo.bc" file and the list of passes printed by ``opt``.

.. _code generator bugs:

Code generator bugs
-------------------

If you find a bug that crashes clang in the code generator, compile your
source file to a .bc file by passing "``-emit-llvm -c -o foo.bc``" to
clang (in addition to the options you already pass).  Once your have
foo.bc, one of the following commands should fail:

#. ``llc foo.bc``
#. ``llc foo.bc -relocation-model=pic``
#. ``llc foo.bc -relocation-model=static``

If none of these crash, please follow the instructions for a `front-end
bug`_.  If one of these do crash, you should be able to reduce this with
one of the following bugpoint command lines (use the one corresponding to
the command above that failed):

#. ``bugpoint -run-llc foo.bc``
#. ``bugpoint -run-llc foo.bc --tool-args -relocation-model=pic``
#. ``bugpoint -run-llc foo.bc --tool-args -relocation-model=static``

Please run this, then file a bug with the instructions and reduced .bc file
that bugpoint emits.  If something goes wrong with bugpoint, please submit
the "foo.bc" file and the option that llc crashes with.

.. _miscompiling:

Miscompilations
===============

If clang successfully produces an executable, but that executable
doesn't run right, this is either a bug in the code or a bug in the
compiler.  The first thing to check is to make sure it is not using
undefined behavior (e.g. reading a variable before it is defined). In
particular, check to see if the program `valgrind
<http://valgrind.org/>`_'s clean, passes purify, or some other memory
checker tool. Many of the "LLVM bugs" that we have chased down ended up
being bugs in the program being compiled, not LLVM.

Once you determine that the program itself is not buggy, you should choose
which code generator you wish to compile the program with (e.g. LLC or the JIT)
and optionally a series of LLVM passes to run.  For example:

.. code-block:: bash

   bugpoint -run-llc [... optzn passes ...] file-to-test.bc --args -- [program arguments]

bugpoint will try to narrow down your list of passes to the one pass that
causes an error, and simplify the bitcode file as much as it can to assist
you. It will print a message letting you know how to reproduce the
resulting error.

Incorrect code generation
=========================

Similarly to debugging incorrect compilation by mis-behaving passes, you
can debug incorrect code generation by either LLC or the JIT, using
``bugpoint``. The process ``bugpoint`` follows in this case is to try to
narrow the code down to a function that is miscompiled by one or the other
method, but since for correctness, the entire program must be run,
``bugpoint`` will compile the code it deems to not be affected with the C
Backend, and then link in the shared object it generates.

To debug the JIT:

.. code-block:: bash

   bugpoint -run-jit -output=[correct output file] [bitcode file]  \
            --tool-args -- [arguments to pass to lli]              \
            --args -- [program arguments]

Similarly, to debug the LLC, one would run:

.. code-block:: bash

   bugpoint -run-llc -output=[correct output file] [bitcode file]  \
            --tool-args -- [arguments to pass to llc]              \
            --args -- [program arguments]

**Special note:** if you are debugging MultiSource or SPEC tests that
already exist in the ``llvm/test`` hierarchy, there is an easier way to
debug the JIT, LLC, and CBE, using the pre-written Makefile targets, which
will pass the program options specified in the Makefiles:

.. code-block:: bash

   cd llvm/test/../../program
   make bugpoint-jit

At the end of a successful ``bugpoint`` run, you will be presented
with two bitcode files: a *safe* file which can be compiled with the C
backend and the *test* file which either LLC or the JIT
mis-codegenerates, and thus causes the error.

To reproduce the error that ``bugpoint`` found, it is sufficient to do
the following:

#. Regenerate the shared object from the safe bitcode file:

   .. code-block:: bash

      llc -march=c safe.bc -o safe.c
      gcc -shared safe.c -o safe.so

#. If debugging LLC, compile test bitcode native and link with the shared
   object:

   .. code-block:: bash

      llc test.bc -o test.s
      gcc test.s safe.so -o test.llc
      ./test.llc [program options]

#. If debugging the JIT, load the shared object and supply the test
   bitcode:

   .. code-block:: bash

      lli -load=safe.so test.bc [program options]
