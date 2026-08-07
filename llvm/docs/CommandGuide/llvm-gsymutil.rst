llvm-gsymutil - GSYM dumping, searching and creating utility
============================================================

.. program:: llvm-gsymutil

SYNOPSIS
--------

:program:`llvm-gsymutil` [*options*] [*gsym-files...*]

DESCRIPTION
-----------

:program:`llvm-gsymutil` is a tool for dumping, searching, and creating GSYM
files.

GSYM is a compact file format for debug information, optimized for fast
lookups. It can represent address ranges, line tables, and inline info.

The tool has three main modes of operation:

#. **Dump Mode**: If one or more GSYM files are specified as arguments without
   any lookup options, the tool dumps all the information contained in the
   specified files.

#. **Lookup Mode**: If a single GSYM file is specified along with one or more
   :option:`--address` options (or :option:`--addresses-from-stdin`), the tool
   performs lookups for the specified addresses in the GSYM file.

#. **Convert Mode**: If the :option:`--convert` option is specified, the tool
   converts the specified ELF or Mach-O file into GSYM format.

OPTIONS
-------

.. option:: --help, -h

  Display information on the various flags.

.. option:: --version, -v

  Display the version of the tool.

.. option:: --verbose

  Enable verbose logging and encoding details.

.. option:: --convert=<file>

  Convert the specified file to the GSYM format. Supported files include ELF and
  Mach-O files. The tool will convert their debug info (DWARF) and symbol table.

.. option:: --symtab-file=<file>

  Specify a separate file to read the symbol table from during GSYM conversion.
  Use when the symbol table and debug info are in separate files. Matching
  architectures are selected automatically for universal binaries.

.. option:: --merged-functions

  * When used with :option:`--convert`, encodes merged function information for
    functions in debug info that have matching address ranges. Without this
    option, one function per unique address range will be emitted.
  * When used with :option:`--address` or :option:`--addresses-from-stdin`, all
    merged functions for a particular address will be displayed. Without this
    option, only one function will be displayed.

.. option:: --dwarf-callsites

  Load call site info from DWARF, if available. This flag only has an impact
  when converting to gsym. When using llvm-gsymutil to lookup addresses,
  any callsite information will automatically be displayed without this flag.

.. option:: --arch=<arch>

  Process debug information for the specified CPU architecture only.
  Architectures may be specified by name or by number. This option can be
  specified multiple times, once for each desired architecture.

.. option:: --out-file=<file>, -o <file>

  Specify the path where the converted GSYM file will be saved. When not
  specified, a '.gsym' extension will be appended to the file name specified in
  the :option:`--convert` option.

.. option:: --verify

  Verify the generated GSYM file against the information in the file that was
  converted.

.. option:: --num-threads=<n>

  Specify the maximum number (n) of simultaneous threads to use when converting
  files to GSYM. Defaults to the number of cores on the current machine.

.. option:: --segment-size=<size>

  Specify the size in bytes of the size the final GSYM file should be segmented
  into. This allows GSYM files to be split across multiple files.

.. option:: --quiet

  Do not output warnings about the debug information.

.. option:: --address=<addr>

  Lookup an address in a GSYM file. Can be specified multiple times.

.. option:: --addresses-from-stdin

  Lookup addresses in a GSYM file that are read from stdin. Each input line is
  expected to be of the following format: ``<addr> <gsym-path>``.

.. option:: --json-summary-file=<file>

  Output a categorized summary of errors into the JSON file specified.

.. option:: --merged-functions-filter=<regex>

  When used with :option:`--address` or :option:`--addresses-from-stdin` and
  :option:`--merged-functions`, filters the merged functions output to only
  show functions matching any of the specified regex patterns. Can be
  specified multiple times.

.. option:: --output-version=<version>

  Set the GSYM output version (1 or 2). Default: 1.

.. option:: --statistics[=<format>]

  Print the size of each section in the input GSYM file(s). Format can be
  ``text`` (default), ``json``, or ``pretty-json``. Calling this option without
  arguments is equivalent to ``--statistics=text``.

EXAMPLES
--------

Convert an ELF file with debug info to GSYM format:

.. code-block:: console

  $ llvm-gsymutil --convert=input.elf -o input.gsym

Lookup addresses in a GSYM file:

.. code-block:: console

  $ llvm-gsymutil --address=0x400391 --address=0x4004cd input.gsym

Lookup addresses from standard input:

.. code-block:: console

  $ cat addrs.txt
  0x400391 input.gsym
  0x4004cd input.gsym
  $ cat addrs.txt | llvm-gsymutil --addresses-from-stdin

Dump the contents of a GSYM file:

.. code-block:: console

  $ llvm-gsymutil input.gsym
