llvm-readelf - GNU-style LLVM Object Reader
===========================================

.. program:: llvm-readelf

SYNOPSIS
--------

:program:`llvm-readelf` [*options*] [*input...*]

DESCRIPTION
-----------

The :program:`llvm-readelf` tool displays low-level format-specific information
about one or more object files.

If ``input`` is "``-``" or omitted, :program:`llvm-readelf` reads from standard
input. Otherwise, it will read from the specified ``filenames``.

OPTIONS
-------

.. option:: --all

 Equivalent to specifying all the main display options.

.. option:: --addrsig

 Display the address-significance table.

.. option:: --arch-specific, -A

 Display architecture-specific information, e.g. the ARM attributes section on ARM.

.. option:: --color

 Use colors in the output for warnings and errors.

.. option:: --demangle, -C

 Display demangled symbol names in the output.

.. option:: --dyn-relocations

 Display the dynamic relocation entries.

.. option:: --dyn-symbols, --dyn-syms

 Display the dynamic symbol table.

.. option:: --dynamic-table, --dynamic, -d

 Display the dynamic table.

.. option:: --elf-cg-profile

 Display the callgraph profile section.

.. option:: --elf-hash-histogram, --histogram, -I

 Display a bucket list histogram for dynamic symbol hash tables.

.. option:: --elf-linker-options

 Display the linker options section.

.. option:: --elf-output-style=<value>

 Format ELF information in the specified style. Valid options are ``LLVM`` and
 ``GNU``. ``LLVM`` output is an expanded and structured format, whilst ``GNU``
 (the default) output mimics the equivalent GNU :program:`readelf` output.

.. option:: --elf-section-groups, --section-groups, -g

 Display section groups.

.. option:: --expand-relocs

 When used with :option:`--relocations`, display each relocation in an expanded
 multi-line format.

.. option:: --file-headers, -h

 Display file headers.

.. option:: --gnu-hash-table

 Display the GNU hash table for dynamic symbols.

.. option:: --hash-symbols

 Display the expanded hash table with dynamic symbol data.

.. option:: --hash-table

 Display the hash table for dynamic symbols.

.. option:: --headers, -e

 Equivalent to setting: :option:`--file-headers`, :option:`--program-headers`,
 and :option:`--sections`.

.. option:: --help

 Display a summary of command line options.

.. option:: --help-list

 Display an uncategorized summary of command line options.

.. option:: --hex-dump=<section[,section,...]>, -x

 Display the specified section(s) as hexadecimal bytes. ``section`` may be a
 section index or section name.

.. option:: --needed-libs

 Display the needed libraries.

.. option:: --notes, -n

 Display all notes.

.. option:: --program-headers, --segments, -l

 Display the program headers.

.. option:: --raw-relr

 Do not decode relocations in RELR relocation sections when displaying them.

.. option:: --relocations, --relocs, -r

 Display the relocation entries in the file.

.. option:: --sections, --section-headers, -S

 Display all sections.

.. option:: --section-data

 When used with :option:`--sections`, display section data for each section
 shown. This option has no effect for GNU style output.

.. option:: --section-mapping

 Display the section to segment mapping.

.. option:: --section-relocations

 When used with :option:`--sections`, display relocations for each section
 shown. This option has no effect for GNU style output.

.. option:: --section-symbols

 When used with :option:`--sections`, display symbols for each section shown.
 This option has no effect for GNU style output.

.. option:: --stackmap

 Display contents of the stackmap section.

.. option:: --stack-sizes

 Display the contents of the stack sizes section(s), i.e. pairs of function
 names and the size of their stack frames. Currently only implemented for GNU
 style output.

.. option:: --string-dump=<section[,section,...]>, -p

 Display the specified section(s) as a list of strings. ``section`` may be a
 section index or section name.

.. option:: --symbols, --syms, -s

 Display the symbol table.

.. option:: --unwind, -u

 Display unwind information.

.. option:: --version

 Display the version of the :program:`llvm-readelf` executable.

.. option:: --version-info, -V

 Display version sections.

.. option:: @<FILE>

 Read command-line options from response file `<FILE>`.

EXIT STATUS
-----------

:program:`llvm-readelf` returns 0 under normal operation. It returns a non-zero
exit code if there were any errors.

SEE ALSO
--------

:manpage:`llvm-nm(1)`, :manpage:`llvm-objdump(1)`, :manpage:`llvm-readobj(1)`
