=====================
LLVM test-suite Guide
=====================

.. contents::
   :local:

Overview
========

This document describes the features of the Makefile-based LLVM
test-suite as well as the cmake based replacement. This way of interacting
with the test-suite is deprecated in favor of running the test-suite using LNT,
but may continue to prove useful for some users. See the Testing
Guide's :ref:`test-suite Quickstart <test-suite-quickstart>` section for more
information.

Test suite Structure
====================

The ``test-suite`` module contains a number of programs that can be
compiled with LLVM and executed. These programs are compiled using the
native compiler and various LLVM backends. The output from the program
compiled with the native compiler is assumed correct; the results from
the other programs are compared to the native program output and pass if
they match.

When executing tests, it is usually a good idea to start out with a
subset of the available tests or programs. This makes test run times
smaller at first and later on this is useful to investigate individual
test failures. To run some test only on a subset of programs, simply
change directory to the programs you want tested and run ``gmake``
there. Alternatively, you can run a different test using the ``TEST``
variable to change what tests or run on the selected programs (see below
for more info).

In addition for testing correctness, the ``test-suite`` directory also
performs timing tests of various LLVM optimizations. It also records
compilation times for the compilers and the JIT. This information can be
used to compare the effectiveness of LLVM's optimizations and code
generation.

``test-suite`` tests are divided into three types of tests: MultiSource,
SingleSource, and External.

-  ``test-suite/SingleSource``

   The SingleSource directory contains test programs that are only a
   single source file in size. These are usually small benchmark
   programs or small programs that calculate a particular value. Several
   such programs are grouped together in each directory.

-  ``test-suite/MultiSource``

   The MultiSource directory contains subdirectories which contain
   entire programs with multiple source files. Large benchmarks and
   whole applications go here.

-  ``test-suite/External``

   The External directory contains Makefiles for building code that is
   external to (i.e., not distributed with) LLVM. The most prominent
   members of this directory are the SPEC 95 and SPEC 2000 benchmark
   suites. The ``External`` directory does not contain these actual
   tests, but only the Makefiles that know how to properly compile these
   programs from somewhere else. The presence and location of these
   external programs is configured by the test-suite ``configure``
   script.

Each tree is then subdivided into several categories, including
applications, benchmarks, regression tests, code that is strange
grammatically, etc. These organizations should be relatively self
explanatory.

Some tests are known to fail. Some are bugs that we have not fixed yet;
others are features that we haven't added yet (or may never add). In the
regression tests, the result for such tests will be XFAIL (eXpected
FAILure). In this way, you can tell the difference between an expected
and unexpected failure.

The tests in the test suite have no such feature at this time. If the
test passes, only warnings and other miscellaneous output will be
generated. If a test fails, a large <program> FAILED message will be
displayed. This will help you separate benign warnings from actual test
failures.

Running the test suite via CMake
================================

To run the test suite, you need to use the following steps:

#. The test suite uses the lit test runner to run the test-suite,
   you need to have lit installed first.  Check out LLVM and install lit:
   
   .. code-block:: bash

       % svn co http://llvm.org/svn/llvm-project/llvm/trunk llvm
       % cd llvm/utils/lit
       % sudo python setup.py install # Or without sudo, install in virtual-env.
       running install
       running bdist_egg
       running egg_info
       writing lit.egg-info/PKG-INFO
       ...
       % lit --version
       lit 0.5.0dev

#. Check out the ``test-suite`` module with:

   .. code-block:: bash

       % svn co http://llvm.org/svn/llvm-project/test-suite/trunk test-suite

#. Use CMake to configure the test suite in a new directory. You cannot build
   the test suite in the source tree.

   .. code-block:: bash
   
       % mkdir test-suite-build
       % cd test-suite-build
       % cmake ../test-suite

#. Build the benchmarks, using the makefiles CMake generated.

.. code-block:: bash

    % make
    Scanning dependencies of target timeit-target
    [  0%] Building C object tools/CMakeFiles/timeit-target.dir/timeit.c.o
    [  0%] Linking C executable timeit-target
    [  0%] Built target timeit-target
    Scanning dependencies of target fpcmp-host
    [  0%] [TEST_SUITE_HOST_CC] Building host executable fpcmp
    [  0%] Built target fpcmp-host
    Scanning dependencies of target timeit-host
    [  0%] [TEST_SUITE_HOST_CC] Building host executable timeit
    [  0%] Built target timeit-host

    
#. Run the tests with lit:

.. code-block:: bash

    % lit -v -j 1 . -o results.json
    -- Testing: 474 tests, 1 threads --
    PASS: test-suite :: MultiSource/Applications/ALAC/decode/alacconvert-decode.test (1 of 474)
    ********** TEST 'test-suite :: MultiSource/Applications/ALAC/decode/alacconvert-decode.test' RESULTS **********
    compile_time: 0.2192 
    exec_time: 0.0462 
    hash: "59620e187c6ac38b36382685ccd2b63b" 
    size: 83348 
    **********
    PASS: test-suite :: MultiSource/Applications/ALAC/encode/alacconvert-encode.test (2 of 474)


Running the test suite via Makefiles (deprecated)
=================================================

First, all tests are executed within the LLVM object directory tree.
They *are not* executed inside of the LLVM source tree. This is because
the test suite creates temporary files during execution.

To run the test suite, you need to use the following steps:

#. ``cd`` into the ``llvm/projects`` directory in your source tree.
#. Check out the ``test-suite`` module with:

   .. code-block:: bash

       % svn co http://llvm.org/svn/llvm-project/test-suite/trunk test-suite

   This will get the test suite into ``llvm/projects/test-suite``.

#. Configure and build ``llvm``.

#. Configure and build ``llvm-gcc``.

#. Install ``llvm-gcc`` somewhere.

#. *Re-configure* ``llvm`` from the top level of each build tree (LLVM
   object directory tree) in which you want to run the test suite, just
   as you do before building LLVM.

   During the *re-configuration*, you must either: (1) have ``llvm-gcc``
   you just built in your path, or (2) specify the directory where your
   just-built ``llvm-gcc`` is installed using
   ``--with-llvmgccdir=$LLVM_GCC_DIR``.

   You must also tell the configure machinery that the test suite is
   available so it can be configured for your build tree:

   .. code-block:: bash

       % cd $LLVM_OBJ_ROOT ; $LLVM_SRC_ROOT/configure [--with-llvmgccdir=$LLVM_GCC_DIR]

   [Remember that ``$LLVM_GCC_DIR`` is the directory where you
   *installed* llvm-gcc, not its src or obj directory.]

#. You can now run the test suite from your build tree as follows:

   .. code-block:: bash

       % cd $LLVM_OBJ_ROOT/projects/test-suite
       % make

Note that the second and third steps only need to be done once. After
you have the suite checked out and configured, you don't need to do it
again (unless the test code or configure script changes).

Configuring External Tests
--------------------------

In order to run the External tests in the ``test-suite`` module, you
must specify *--with-externals*. This must be done during the
*re-configuration* step (see above), and the ``llvm`` re-configuration
must recognize the previously-built ``llvm-gcc``. If any of these is
missing or neglected, the External tests won't work.

* *--with-externals*

* *--with-externals=<directory>*

This tells LLVM where to find any external tests. They are expected to
be in specifically named subdirectories of <``directory``>. If
``directory`` is left unspecified, ``configure`` uses the default value
``/home/vadve/shared/benchmarks/speccpu2000/benchspec``. Subdirectory
names known to LLVM include:

* spec95

* speccpu2000

* speccpu2006

* povray31

Others are added from time to time, and can be determined from
``configure``.

Running different tests
-----------------------

In addition to the regular "whole program" tests, the ``test-suite``
module also provides a mechanism for compiling the programs in different
ways. If the variable TEST is defined on the ``gmake`` command line, the
test system will include a Makefile named
``TEST.<value of TEST variable>.Makefile``. This Makefile can modify
build rules to yield different results.

For example, the LLVM nightly tester uses ``TEST.nightly.Makefile`` to
create the nightly test reports. To run the nightly tests, run
``gmake TEST=nightly``.

There are several TEST Makefiles available in the tree. Some of them are
designed for internal LLVM research and will not work outside of the
LLVM research group. They may still be valuable, however, as a guide to
writing your own TEST Makefile for any optimization or analysis passes
that you develop with LLVM.

Generating test output
----------------------

There are a number of ways to run the tests and generate output. The
most simple one is simply running ``gmake`` with no arguments. This will
compile and run all programs in the tree using a number of different
methods and compare results. Any failures are reported in the output,
but are likely drowned in the other output. Passes are not reported
explicitly.

Somewhat better is running ``gmake TEST=sometest test``, which runs the
specified test and usually adds per-program summaries to the output
(depending on which sometest you use). For example, the ``nightly`` test
explicitly outputs TEST-PASS or TEST-FAIL for every test after each
program. Though these lines are still drowned in the output, it's easy
to grep the output logs in the Output directories.

Even better are the ``report`` and ``report.format`` targets (where
``format`` is one of ``html``, ``csv``, ``text`` or ``graphs``). The
exact contents of the report are dependent on which ``TEST`` you are
running, but the text results are always shown at the end of the run and
the results are always stored in the ``report.<type>.format`` file (when
running with ``TEST=<type>``). The ``report`` also generate a file
called ``report.<type>.raw.out`` containing the output of the entire
test run.

Writing custom tests for the test suite
---------------------------------------

Assuming you can run the test suite, (e.g.
"``gmake TEST=nightly report``" should work), it is really easy to run
optimizations or code generator components against every program in the
tree, collecting statistics or running custom checks for correctness. At
base, this is how the nightly tester works, it's just one example of a
general framework.

Lets say that you have an LLVM optimization pass, and you want to see
how many times it triggers. First thing you should do is add an LLVM
`statistic <ProgrammersManual.html#Statistic>`_ to your pass, which will
tally counts of things you care about.

Following this, you can set up a test and a report that collects these
and formats them for easy viewing. This consists of two files, a
"``test-suite/TEST.XXX.Makefile``" fragment (where XXX is the name of
your test) and a "``test-suite/TEST.XXX.report``" file that indicates
how to format the output into a table. There are many example reports of
various levels of sophistication included with the test suite, and the
framework is very general.

If you are interested in testing an optimization pass, check out the
"libcalls" test as an example. It can be run like this:

.. code-block:: bash

    % cd llvm/projects/test-suite/MultiSource/Benchmarks  # or some other level
    % make TEST=libcalls report

This will do a bunch of stuff, then eventually print a table like this:

::

    Name                                  | total | #exit |
    ...
    FreeBench/analyzer/analyzer           | 51    | 6     |
    FreeBench/fourinarow/fourinarow       | 1     | 1     |
    FreeBench/neural/neural               | 19    | 9     |
    FreeBench/pifft/pifft                 | 5     | 3     |
    MallocBench/cfrac/cfrac               | 1     | *     |
    MallocBench/espresso/espresso         | 52    | 12    |
    MallocBench/gs/gs                     | 4     | *     |
    Prolangs-C/TimberWolfMC/timberwolfmc  | 302   | *     |
    Prolangs-C/agrep/agrep                | 33    | 12    |
    Prolangs-C/allroots/allroots          | *     | *     |
    Prolangs-C/assembler/assembler        | 47    | *     |
    Prolangs-C/bison/mybison              | 74    | *     |
    ...

This basically is grepping the -stats output and displaying it in a
table. You can also use the "TEST=libcalls report.html" target to get
the table in HTML form, similarly for report.csv and report.tex.

The source for this is in ``test-suite/TEST.libcalls.*``. The format is
pretty simple: the Makefile indicates how to run the test (in this case,
"``opt -simplify-libcalls -stats``"), and the report contains one line
for each column of the output. The first value is the header for the
column and the second is the regex to grep the output of the command
for. There are lots of example reports that can do fancy stuff.
