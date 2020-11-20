llvm-profgen - LLVM SPGO profile generation tool
=================================

.. program:: llvm-profgen

SYNOPSIS
--------

:program:`llvm-profgen` [*commands*] [*options*]

DESCRIPTION
-----------

The :program:`llvm-profgen` utility generates a profile data file
from given perf script data files for sample-based profile guided
optimization(SPGO).

COMMANDS
--------
At least one of the following commands are required:

.. option:: --perfscript=<string[,string,...]>

  Path of perf-script trace created by Linux perf tool with `script`
  command(the raw perf.data should be profiled with -b).

.. option:: --output=<string>

  Path of the output profile file.

OPTIONS
-------
:program:`llvm-profgen` supports the following options:

.. option:: --binary=<string[,string,...]>

  Path of the input profiled binary files. If no file path is specified, the
  path of the actual profiled binaries will be used instead.

.. option:: --show-mmap-events

  Print mmap events.
