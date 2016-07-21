opt - LLVM optimizer
====================

SYNOPSIS
--------

:program:`opt` [*options*] [*filename*]

DESCRIPTION
-----------

The :program:`opt` command is the modular LLVM optimizer and analyzer.  It
takes LLVM source files as input, runs the specified optimizations or analyses
on it, and then outputs the optimized file or the analysis results.  The
function of :program:`opt` depends on whether the `-analyze` option is
given.

When `-analyze` is specified, :program:`opt` performs various analyses
of the input source.  It will usually print the results on standard output, but
in a few cases, it will print output to standard error or generate a file with
the analysis output, which is usually done when the output is meant for another
program.

While `-analyze` is *not* given, :program:`opt` attempts to produce an
optimized output file.  The optimizations available via :program:`opt` depend
upon what libraries were linked into it as well as any additional libraries
that have been loaded with the :option:`-load` option.  Use the :option:`-help`
option to determine what optimizations you can use.

If ``filename`` is omitted from the command line or is "``-``", :program:`opt`
reads its input from standard input.  Inputs can be in either the LLVM assembly
language format (``.ll``) or the LLVM bitcode format (``.bc``).

If an output filename is not specified with the :option:`-o` option,
:program:`opt` writes its output to the standard output.

OPTIONS
-------

.. option:: -f

 Enable binary output on terminals.  Normally, :program:`opt` will refuse to
 write raw bitcode output if the output stream is a terminal.  With this option,
 :program:`opt` will write raw bitcode regardless of the output device.

.. option:: -help

 Print a summary of command line options.

.. option:: -o <filename>

 Specify the output filename.

.. option:: -S

 Write output in LLVM intermediate language (instead of bitcode).

.. option:: -{passname}

 :program:`opt` provides the ability to run any of LLVM's optimization or
 analysis passes in any order.  The :option:`-help` option lists all the passes
 available.  The order in which the options occur on the command line are the
 order in which they are executed (within pass constraints).

.. option:: -disable-inlining

 This option simply removes the inlining pass from the standard list.

.. option:: -disable-opt

 This option is only meaningful when `-std-link-opts` is given.  It
 disables most passes.

.. option:: -strip-debug

 This option causes opt to strip debug information from the module before
 applying other optimizations.  It is essentially the same as `-strip`
 but it ensures that stripping of debug information is done first.

.. option:: -verify-each

 This option causes opt to add a verify pass after every pass otherwise
 specified on the command line (including `-verify`).  This is useful
 for cases where it is suspected that a pass is creating an invalid module but
 it is not clear which pass is doing it.

.. option:: -stats

 Print statistics.

.. option:: -time-passes

 Record the amount of time needed for each pass and print it to standard
 error.

.. option:: -debug

 If this is a debug build, this option will enable debug printouts from passes
 which use the ``DEBUG()`` macro.  See the `LLVM Programmer's Manual
 <../ProgrammersManual.html>`_, section ``#DEBUG`` for more information.

.. option:: -load=<plugin>

 Load the dynamic object ``plugin``.  This object should register new
 optimization or analysis passes.  Once loaded, the object will add new command
 line options to enable various optimizations or analyses.  To see the new
 complete list of optimizations, use the :option:`-help` and :option:`-load`
 options together.  For example:

 .. code-block:: sh

     opt -load=plugin.so -help

.. option:: -p

 Print module after each transformation.

EXIT STATUS
-----------

If :program:`opt` succeeds, it will exit with 0.  Otherwise, if an error
occurs, it will exit with a non-zero value.

