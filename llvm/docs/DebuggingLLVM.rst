==============
Debugging LLVM
==============

This document is a collection of tips and tricks for debugging LLVM
using a source-level debugger. The assumption is that you are trying to
figure out the root cause of a miscompilation in the program that you
are compiling.

Extract and rerun the compile command
=====================================

Extract the Clang command that produces the buggy code. The way to do
this depends on the build system used by your program.

- For Ninja-based build systems, you can pass ``-t commands`` to Ninja
  and filter the output by the targeted source file name. For example:
  ``ninja -t commands myprogram | grep path/to/file.cpp``.

- For Bazel-based build systems using Bazel 9 or newer (not released yet
  as of this writing), you can pass ``--output=commands`` to the ``bazel
  aquery`` subcommand for a similar result. For example: ``bazel aquery
  --output=commands 'deps(//myprogram)' | grep path/to/file.cpp``. Build
  commands must generally be run from a subdirectory of the source
  directory named ``bazel-$PROJECTNAME``. Bazel typically makes the target
  paths of ``-o`` and ``-MF`` read-only when running commands outside
  of a build, so it may be necessary to change or remove these flags.

- A method that should work with any build system is to build your program
  under `Bear <https://github.com/rizsotto/Bear>`_ and look for the
  compile command in the resulting ``compile_commands.json`` file.

Once you have the command you can use the following steps to debug
it. Note that any flags mentioned later in this document are LLVM flags
so they must be prefixed with ``-mllvm`` when passed to the Clang driver,
e.g. ``-mllvm -print-after-all``.

Understanding the source of the issue
=====================================

If you have a miscompilation introduced by a pass, it is
frequently possible to identify the pass where things go wrong
by searching a pass-by-pass printout, which is enabled using the
``-print-after-all`` flag. Pipe stderr into ``less`` (append ``2>&1 |
less`` to command line) and use text search to move between passes
(e.g. type ``/Dump After<Enter>``, ``n`` to move to next pass,
``N`` to move to previous pass). If the name of the function
containing the buggy IR is known, you can filter the output by passing
``-filter-print-funcs=functionname``. You can sometimes pass ``-debug`` to
get useful details about what passes are doing. See also  `PrintPasses.cpp
<https://github.com/llvm/llvm-project/blob/main/llvm/lib/IR/PrintPasses.cpp>`_
for more useful options.

Creating a debug build of LLVM
==============================

The subsequent debugging steps require a debug build of LLVM. Pass the
``-DCMAKE_BUILD_TYPE=Debug`` to CMake in a separate build tree to create
a debug build.

Understanding where an instruction came from
============================================

A common debugging task involves understanding which part of the code
introduced a buggy instruction. The pass-by-pass dump is sometimes enough,
but for complex or unfamiliar passes, more information is often required.

The first step is to record a run of the debug build of Clang under `rr
<https://rr-project.org>`_ passing the LLVM flag ``-print-inst-addrs``
together with ``-print-after-all`` and any desired filters. This will
cause each instruction printed by LLVM to be suffixed with a comment
showing the address of the ``Instruction`` object. You can then replay
the run of Clang with ``rr replay``. Because ``rr`` is deterministic,
the instruction will receive the same address during the replay, so
you can break on the instruction's construction using a conditional
breakpoint that checks for the address printed by LLVM, with commands
such as the following:

.. code-block:: text

    b Instruction::Instruction if this == 0x12345678

When the breakpoint is hit, you will likely be at the location where
the instruction was created, so you can unwind the stack with ``bt``
to see the stack trace. It is also possible that an instruction was
created multiple times at the same address, so you may need to continue
until reaching the desired location, but in the author's experience this
is unlikely to occur.

Identifying the source locations of instructions
================================================

To identify the source location that caused a particular instruction
to be created, you can pass the LLVM flag ``-print-inst-debug-locs``
and each instruction printed by LLVM is suffixed with the file and line
number of the instruction according to the debug information. Note that
this requires debug information to be enabled (e.g. pass ``-g`` to Clang).

LLDB Data Formatters
====================

A handful of `LLDB data formatters
<https://lldb.llvm.org/resources/dataformatters.html>`__ are
provided for some of the core LLVM libraries. To use them, execute the
following (or add it to your ``~/.lldbinit``)::

  command script import /path/to/llvm/utils/lldbDataFormatters.py

GDB pretty printers
===================

A handful of `GDB pretty printers
<https://sourceware.org/gdb/onlinedocs/gdb/Pretty-Printing.html>`__ are
provided for some of the core LLVM libraries. To use them, execute the
following (or add it to your ``~/.gdbinit``)::

  source /path/to/llvm/utils/gdb-scripts/prettyprinters.py

It also might be handy to enable the `print pretty
<https://sourceware.org/gdb/current/onlinedocs/gdb.html/Print-Settings.html>`__
option to avoid data structures being printed as a big block of text.
