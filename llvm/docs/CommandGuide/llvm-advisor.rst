llvm-advisor - LLVM compilation analysis tool
=============================================

.. program:: llvm-advisor

SYNOPSIS
--------

:program:`llvm-advisor` [*options*] *compiler* [*compiler-args...*]

:program:`llvm-advisor` **view** [*options*] *compiler* [*compiler-args...*]

DESCRIPTION
-----------

The :program:`llvm-advisor` tool is a compilation analysis utility that acts as
a wrapper around compiler commands to collect detailed information about the
compilation process. It captures compilation data, optimization information, 
and diagnostic details that can be used to analyze and improve build performance 
and code optimization. The tool requires no external dependencies beyond a 
standard LLVM/Clang installation and Python 3 for the web viewer.

:program:`llvm-advisor` intercepts compiler invocations and instruments them
to extract and collect data including:

* LLVM IR, assembly, and preprocessed source code output
* AST dumps in both text and JSON formats  
* Include dependency trees and macro expansion information
* Compiler diagnostics, warnings, and static analysis results
* Debug information and DWARF data analysis
* Compilation timing reports (via ``-ftime-report``)
* Runtime profiling and coverage data (when profiler is enabled)
* Binary size analysis and symbol information
* Optimization remarks from compiler passes

The tool supports two primary modes of operation:

**Data Collection Mode**: The default mode where :program:`llvm-advisor` wraps
the compiler command, collects analysis data, and stores it in a
hierarchically organized output directory for later analysis.

**View Mode**: When invoked with the **view** subcommand, :program:`llvm-advisor`
performs data collection and then automatically launches a web-based interface
with interactive visualization and analysis capabilities. The web viewer provides
a REST API for programmatic access to the collected data.

All collected data is stored in a timestamped, structured format within the output
directory (default: ``.llvm-advisor``) and can be analyzed using the built-in web
viewer or external tools.

OPTIONS
-------

.. option:: --config <file>

 Specify a configuration file to customize :program:`llvm-advisor` behavior.
 The configuration file uses JSON format and can override default settings
 for output directory, verbosity, and other options.

.. option:: --output-dir <directory>

 Specify the directory where compilation analysis data will be stored.
 If the directory doesn't exist, it will be created. The default output
 directory is ``.llvm-advisor`` in the current working directory.

.. option:: --verbose

 Enable verbose output to display detailed information about the analysis
 process, including the compiler command being executed and the location
 of collected data.

.. option:: --keep-temps

 Preserve temporary files created during the analysis process. By default,
 temporary files are cleaned up automatically. This option is useful for
 debugging or when you need to examine intermediate analysis results.

.. option:: --no-profiler

 Disable the automatic addition of compiler profiling flags during compilation.
 By default, :program:`llvm-advisor` adds flags like ``-fprofile-instr-generate``
 and ``-fcoverage-mapping`` to collect runtime profiling data. This option
 disables that behavior, reducing compilation overhead but limiting the
 coverage and profiling data available for analysis.

.. option:: --port <port>

 Specify the port number for the web server when using the **view** command.
 The default port is 8000. The web viewer will be accessible at
 ``http://localhost:<port>``.

.. option:: --help, -h

 Display usage information and available options.

COMMANDS
--------

:program:`llvm-advisor` supports the following commands:

Data Collection (Default)
~~~~~~~~~~~~~~~~~~~~~~~~~~

When no subcommand is specified, :program:`llvm-advisor` operates in data
collection mode:

.. code-block:: console

  llvm-advisor [options] <compiler> [compiler-args...]

This mode wraps the specified compiler command, collects analysis data during
compilation, and stores the results in the output directory.

View Mode
~~~~~~~~~

The **view** subcommand combines data collection with automatic web viewer
launch:

.. code-block:: console

  llvm-advisor view [options] <compiler> [compiler-args...]

In this mode, :program:`llvm-advisor` first performs compilation with data
collection, then launches a web server providing an interactive interface
to analyze the collected data. The web viewer remains active until manually
terminated.

EXAMPLES
--------

Basic Usage
~~~~~~~~~~~

Analyze a simple C compilation:

.. code-block:: console

  llvm-advisor clang -O2 -g main.c -o main

This command will compile ``main.c`` using clang with ``-O2`` optimization
and debug information, while collecting analysis data in the 
``.llvm-advisor`` directory. The output will be organized as:

.. code-block:: text

  .llvm-advisor/
  └── main/
      └── main_20250825_143022/  # Timestamped compilation session
          ├── ir/main.ll         # LLVM IR output
          ├── assembly/main.s    # Assembly output  
          ├── ast/main.ast       # AST dump
          ├── diagnostics/       # Compiler warnings/errors
          └── ...               # Additional analysis data

Complex C++ Project
~~~~~~~~~~~~~~~~~~~

Analyze a C++ compilation with custom output directory:

.. code-block:: console

  llvm-advisor --output-dir analysis-results clang++ -O3 -std=c++17 app.cpp lib.cpp -o app

Compile with maximum optimization and store analysis results in the
``analysis-results`` directory.

Interactive Analysis
~~~~~~~~~~~~~~~~~~~~

Compile and immediately launch the web viewer:

.. code-block:: console

  llvm-advisor view --port 8080 clang -O2 main.c

This will compile ``main.c``, collect analysis data, and launch a web interface
accessible at ``http://localhost:8080`` for interactive analysis.

Configuration File Usage
~~~~~~~~~~~~~~~~~~~~~~~~~

Use a custom configuration file:

.. code-block:: console

  llvm-advisor --config custom-config.json --verbose clang++ -O1 project.cpp

Example configuration file (``custom-config.json``):

.. code-block:: json

  {
    "outputDir": "compilation-analysis",
    "verbose": true,
    "keepTemps": false,
    "runProfiler": true,
    "timeout": 120
  }

Integration with Build Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:program:`llvm-advisor` can be integrated into existing build systems by
substituting the compiler command:

.. code-block:: console

  # Instead of: make CC=clang CXX=clang++
  make CC="llvm-advisor clang" CXX="llvm-advisor clang++"

  # For CMake projects:
  cmake -DCMAKE_C_COMPILER="llvm-advisor clang" \
        -DCMAKE_CXX_COMPILER="llvm-advisor clang++" \
        ..

Accessing Historical Data
~~~~~~~~~~~~~~~~~~~~~~~~~

The timestamped directory structure allows you to analyze compilation trends
over time:

.. code-block:: console

  # View most recent compilation results
  llvm-advisor view --output-dir .llvm-advisor

  # Each unit directory contains multiple timestamped runs
  ls .llvm-advisor/myproject/
  # Output: myproject_20250825_140512  myproject_20250825_143022

The web viewer automatically uses the most recent compilation run for analysis,
but all historical data remains accessible in the timestamped directories.

CONFIGURATION
-------------

:program:`llvm-advisor` can be configured using a JSON configuration file
specified with the :option:`--config` option. The configuration file supports
the following options:

**outputDir** (string)
  Default output directory for analysis data.

**verbose** (boolean)
  Enable verbose output by default.

**keepTemps** (boolean)
  Preserve temporary files by default.

**runProfiler** (boolean)
  Enable performance profiling during compilation.

**timeout** (integer)
  Timeout in seconds for compilation analysis (default: 60).

OUTPUT FORMAT
-------------

:program:`llvm-advisor` generates analysis data in a structured format within
the output directory. The tool organizes data hierarchically by compilation unit
and timestamp, allowing multiple compilation sessions to be tracked over time.

The typical output structure includes:

.. code-block:: text

  .llvm-advisor/
  └── {compilation-unit}/           # One directory per compilation unit
      └── {unit-name}_{timestamp}/  # Timestamped compilation runs
          ├── ir/                   # LLVM IR files (.ll)
          ├── assembly/             # Assembly output (.s)
          ├── ast/                  # AST dumps (.ast) and JSON (.ast.json)
          ├── preprocessed/         # Preprocessed source (.i/.ii)
          ├── include-tree/         # Include hierarchy information
          ├── dependencies/         # Dependency analysis (.deps.txt)
          ├── debug/                # Debug information and DWARF data
          ├── static-analyzer/      # Static analysis results
          ├── diagnostics/          # Compiler diagnostics and warnings
          ├── coverage/             # Code coverage data
          ├── time-trace/           # Compilation time traces
          ├── runtime-trace/        # Runtime tracing information
          ├── binary-analysis/      # Binary size and symbol analysis
          ├── pgo/                  # Profile-guided optimization data
          ├── ftime-report/         # Compilation timing reports
          ├── version-info/         # Compiler version information
          └── sources/              # Source file copies and metadata

Each compilation run creates a new timestamped directory, preserving the history
of compilation sessions. The most recent run is automatically used by the web
viewer for analysis.

EXIT STATUS
-----------

:program:`llvm-advisor` returns the same exit status as the wrapped compiler
command. If the compilation succeeds, it returns 0. If the compilation fails
or :program:`llvm-advisor` encounters an internal error, it returns a non-zero
exit status.

:program:`llvm-advisor` returns exit code 1 for various error conditions including:

* Invalid command line arguments or missing compiler command
* Configuration file parsing errors  
* Output directory creation failures
* Web viewer launch failures (view mode only)
* Data collection or extraction errors

SEE ALSO
--------

:manpage:`clang(1)`, :manpage:`opt(1)`, :manpage:`llc(1)`
