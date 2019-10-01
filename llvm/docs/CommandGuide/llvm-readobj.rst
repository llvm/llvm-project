llvm-readobj - LLVM Object Reader
=================================

SYNOPSIS
--------

:program:`llvm-readobj` [*options*] [*input...*]

DESCRIPTION
-----------

The :program:`llvm-readobj` tool displays low-level format-specific information
about one or more object files. The tool and its output is primarily designed
for use in FileCheck-based tests.

OPTIONS
-------

If ``input`` is "``-``" or omitted, :program:`llvm-readobj` reads from standard
input. Otherwise, it will read from the specified ``filenames``.

.. option:: -help

 Print a summary of command line options.

.. option:: -version

 Display the version of this program

.. option:: -file-headers, -h

 Display file headers.

.. option:: -sections, -s

 Display all sections.

.. option:: -section-data, -sd

 When used with ``-sections``, display section data for each section shown.

.. option:: -section-relocations, -sr

 When used with ``-sections``, display relocations for each section shown.

.. option:: -section-symbols, -st

 When used with ``-sections``, display symbols for each section shown.

.. option:: -relocations, -r

 Display the relocation entries in the file.

.. option:: -symbols, -t

 Display the symbol table.

.. option:: -dyn-symbols

 Display the dynamic symbol table (only for ELF object files).

.. option:: -unwind, -u

 Display unwind information.

.. option:: -expand-relocs

 When used with ``-relocations``, display each relocation in an expanded
 multi-line format.

.. option:: -dynamic-table

 Display the ELF .dynamic section table (only for ELF object files).

.. option:: -needed-libs

 Display the needed libraries (only for ELF object files).

.. option:: -program-headers

 Display the ELF program headers (only for ELF object files).

.. option:: -elf-section-groups, -g

 Display section groups (only for ELF object files).

.. option:: -demangle, -C

 Print demangled symbol names in the output.

EXIT STATUS
-----------

:program:`llvm-readobj` returns 0.
