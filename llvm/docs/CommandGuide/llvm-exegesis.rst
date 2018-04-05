llvm-exegesis - LLVM Machine Instruction Benchmark
==================================================

SYNOPSIS
--------

:program:`llvm-exegesis` [*options*]

DESCRIPTION
-----------

:program:`llvm-exegesis` is a benchmarking tool that uses information available
in LLVM to measure host machine instruction characteristics like latency or port
decomposition.

Given an LLVM opcode name and a benchmarking mode, :program:`llvm-exegesis`
generates a code snippet that makes execution as serial (resp. as parallel) as
possible so that we can measure the latency (resp. uop decomposition) of the
instruction.
The code snippet is jitted and executed on the host subtarget. The time taken
(resp. resource usage) is measured using hardware performance counters. The
result is printed out as YAML to the standard output.

The main goal of this tool is to automatically (in)validate the LLVM's TableDef
scheduling models.

OPTIONS
-------

.. option:: -help

 Print a summary of command line options.

.. option:: -opcode-index=<LLVM opcode index>

 Specify the opcode to measure, by index.
 Either `opcode-index` or `opcode-name` must be set.

.. option:: -opcode-name=<LLVM opcode name>

 Specify the opcode to measure, by name.
 Either `opcode-index` or `opcode-name` must be set.

.. option:: -benchmark-mode=[Latency|Uops]

 Specify which characteristic of the opcode to measure.

.. option:: -num-repetitions=<Number of repetition>

 Specify the number of repetitions of the asm snippet.
 Higher values lead to more accurate measurements but lengthen the benchmark.


EXIT STATUS
-----------

:program:`llvm-exegesis` returns 0 on success. Otherwise, an error message is
printed to standard error, and the tool returns a non 0 value.
