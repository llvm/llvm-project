# Debugging LLVM

This document is a collection of tips and tricks for debugging LLVM
using a source-level debugger. The assumption is that you are trying to
figure out the root cause of a miscompilation in the program that you
are compiling.

## Extract and rerun the compile command

Extract the Clang command that produces the buggy code. The way to do
this depends on the build system used by your program.

- For Ninja-based build systems, you can pass `-t commands` to Ninja
  and filter the output by the targeted source file name. For example:
  `ninja -t commands myprogram | grep path/to/file.cpp`.

- For Bazel-based build systems using Bazel 9 or newer, you can pass
  `--output=commands` to the `bazel aquery` subcommand for
  a similar result. For example: `bazel aquery --output=commands
  'deps(//myprogram)' | grep path/to/file.cpp`. Build commands must
  generally be run from a subdirectory of the source directory named
  `bazel-$PROJECTNAME`. Bazel typically makes the target paths of
  `-o` and `-MF` read-only when running commands outside of a build,
  so it may be necessary to change or remove these flags.

- A method that should work with any build system is to build your program
  under [Bear] and look for the
  compile command in the resulting `compile_commands.json` file.

Once you have the command you can use the following steps to debug
it. Note that any flags mentioned later in this document are LLVM flags
so they must be prefixed with `-mllvm` when passed to the Clang driver,
e.g. `-mllvm -print-after-all`.

## Understanding the source of the issue

If you have a miscompilation introduced by a pass, it is
frequently possible to identify the pass where things go wrong
by searching a pass-by-pass printout. The following options control when IR is
printed:

- `-print-before-all` and `-print-after-all` print IR before or after every
  pass.
- `-print-before=<pass-name>` and `-print-after=<pass-name>` print IR around
  selected passes. Multiple pass names can be separated by commas.
- `-print-pass-numbers` prints the pass names and their ordinals.
  `-print-before-pass-number=<number>` and
  `-print-after-pass-number=<number>` can then select particular pass
  invocations. Multiple numbers can be separated by commas.
- `-print-changed` prints IR only after passes that change it. Its optional
  modes include `quiet`, `diff`, `diff-quiet`, `cdiff`, `cdiff-quiet`,
  `dot-cfg`, and `dot-cfg-quiet`. `-print-before-changed` also prints the IR
  before each detected change.

The following options reduce or organize the output:

- `-filter-print-funcs=<function-name>` limits output to the listed functions.
  Multiple names can be separated by commas.
- `-filter-print-source-locs=<file>:<line-list>` limits output to IR units that
  contain a matching debug location. A line list can contain individual lines
  and inclusive ranges, for example
  `-filter-print-source-locs=example.cpp:12,20-25`. The file may be a basename,
  a complete path, or a path suffix. Inlined locations are also considered.
  Repeat the option to select locations from more than one file. This option
  requires debug information, such as that produced by Clang's `-g` option. It
  selects whole functions, loops, or other IR units containing a match; it does
  not remove individual nonmatching instructions from them.
- `-filter-passes=<pass-name>` limits `-print-changed` output to the listed
  passes. Multiple names can be separated by commas.
- `-print-module-scope` prints the whole module instead of the IR unit on which
  a pass runs. `-print-loop-func-scope` prints the containing function for loop
  passes.
- `-ir-dump-directory=<directory>` writes `-print-before` and `-print-after`
  output to files instead of standard error.

For example, the following command prints the IR after every pass, but only for
functions containing instructions associated with the selected source lines:

```text
clang -O2 -g -mllvm -print-after-all \
  -mllvm -filter-print-source-locs=example.cpp:12,20-25 example.cpp -c
```

By default, dumps are written to standard error. Pipe standard error into
`less` (append `2>&1 | less` to the command line) and use text search to move
between passes (for example, type `/Dump After<Enter>`, `n` to move to the next
pass, and `N` to move to the previous pass).

Metadata IDs remain stable across repeated pass and debug dumps, which makes
nodes easier to follow between snapshots. These IDs may contain gaps or appear
in a different order from normal complete-module LLVM IR output, where metadata
IDs are renumbered into a contiguous canonical sequence.

You can sometimes pass `-debug` to get useful details about what passes are
doing. See also [PrintPasses.cpp] for the option definitions.

## Creating a debug build of LLVM

The subsequent debugging steps require a debug build of LLVM. Pass the
`-DCMAKE_BUILD_TYPE=Debug` to CMake in a separate build tree to create
a debug build.

## Understanding where an instruction came from

A common debugging task involves understanding which part of the code
introduced a buggy instruction. The pass-by-pass dump is sometimes enough,
but for complex or unfamiliar passes, more information is often required.

The first step is to record a run of the debug build of Clang under [rr]
passing the LLVM flag `-print-inst-addrs`
together with `-print-after-all` and any desired filters. This will
cause each instruction printed by LLVM to be suffixed with a comment
showing the address of the `Instruction` object. You can then replay
the run of Clang with `rr replay`. Because `rr` is deterministic,
the instruction will receive the same address during the replay, so
you can break on the instruction's construction using a conditional
breakpoint that checks for the address printed by LLVM, with commands
such as the following:

```text
b Instruction::Instruction if this == 0x12345678
```

When the breakpoint is hit, you will likely be at the location where
the instruction was created, so you can unwind the stack with `bt`
to see the stack trace. It is also possible that an instruction was
created multiple times at the same address, so you may need to continue
until reaching the desired location, but in the author's experience this
is unlikely to occur.

Similar flags exist for the backend: `-print-sdnode-addrs` for
printing `SDNode` addresses, and `-print-mi-addrs` for printing
`MachineInstr` addresses.

## Identifying the source locations of instructions

To identify the source location that caused a particular instruction
to be created, you can pass the LLVM flag `-print-inst-debug-locs`
and each instruction printed by LLVM is suffixed with the file and line
number of the instruction according to the debug information. Note that
this requires debug information to be enabled (e.g. pass `-g` to Clang).

## LLDB Data Formatters

A handful of [LLDB data formatters] are
provided for some of the core LLVM libraries. To use them, execute the
following (or add it to your `~/.lldbinit`):

```text
command script import /path/to/llvm/utils/lldbDataFormatters.py
```

## GDB pretty printers

A handful of [GDB pretty printers] are
provided for some of the core LLVM libraries. To use them, execute the
following (or add it to your `~/.gdbinit`):

```text
source /path/to/llvm/utils/gdb-scripts/prettyprinters.py
```

It also might be handy to enable the [print pretty]
option to avoid data structures being printed as a big block of text.

[Bear]: https://github.com/rizsotto/Bear
[PrintPasses.cpp]: https://github.com/llvm/llvm-project/blob/main/llvm/lib/IR/PrintPasses.cpp
[rr]: https://rr-project.org
[LLDB data formatters]: https://lldb.llvm.org/resources/dataformatters.html
[GDB pretty printers]: https://sourceware.org/gdb/onlinedocs/gdb/Pretty-Printing.html
[print pretty]: https://sourceware.org/gdb/current/onlinedocs/gdb.html/Print-Settings.html
