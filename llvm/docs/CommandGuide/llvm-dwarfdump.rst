llvm-dwarfdump - dump and verify DWARF debug information
========================================================

.. program:: llvm-dwarfdump

SYNOPSIS
--------

:program:`llvm-dwarfdump` [*options*] [*filename ...*]

DESCRIPTION
-----------

:program:`llvm-dwarfdump` parses DWARF sections in object files,
archives, and `.dSYM` bundles and prints their contents in
human-readable form. Only the .debug_info section is printed unless one of
the section-specific options or :option:`--all` is specified.

If no input file is specified, `a.out` is used instead. If `-` is used as the
input file, :program:`llvm-dwarfdump` reads the input from its standard input
stream.

OPTIONS
-------

.. option:: -a, --all

            Disassemble all supported DWARF sections.

.. option:: --arch=<arch>

            Dump DWARF debug information for the specified CPU architecture.
            Architectures may be specified by name or by number.  This
            option can be specified multiple times, once for each desired
            architecture.  All CPU architectures will be printed by
            default.

.. option:: -c, --show-children

            Show a debug info entry's children when selectively printing with
            the `=<offset>` argument of :option:`--debug-info`, or options such
            as :option:`--find` or :option:`--name`.

.. option:: --color

            Use colors in output.

.. option:: -f <name>, --find=<name>

            Search for the exact text <name> in the accelerator tables
            and print the matching debug information entries.
            When there is no accelerator tables or the name of the DIE
            you are looking for is not found in the accelerator tables,
            try using the slower but more complete :option:`--name` option.

.. option:: -F, --show-form

            Show DWARF form types after the DWARF attribute types.

.. option:: -h, --help

            Show help and usage for this command.

.. option:: --help-list

            Show help and usage for this command without grouping the options
            into categories.

.. option:: -i, --ignore-case

            Ignore case distinctions when searching.

.. option:: -n <pattern>, --name=<pattern>

            Find and print all debug info entries whose name
            (`DW_AT_name` attribute) matches the exact text in
            <pattern>. Use the :option:`--regex` option to have
            <pattern> become a regular expression for more flexible
            pattern matching.

.. option:: --lookup=<address>

            Lookup <address> in the debug information and print out the file,
            function, block, and line table details.

.. option:: -o <path>

            Redirect output to a file specified by <path>, where `-` is the
            standard output stream.

.. option:: -p, --show-parents

            Show a debug info entry's parents when selectively printing with
            the `=<offset>` argument of :option:`--debug-info`, or options such
            as :option:`--find` or :option:`--name`.

.. option:: --parent-recurse-depth=<N>

            Only recurse to a maximum depth of <N> when displaying parents of
            debug info entries.

.. option:: --quiet

            Use with :option:`--verify` to not emit to `STDOUT`.

.. option:: -r <N>, --recurse-depth=<N>

            Only recurse to a maximum depth of <N> when dumping debug info
            entries.

.. option:: --statistics

            Collect debug info quality metrics and print the results
            as machine-readable single-line JSON output.

.. option:: --summarize-types

            Abbreviate the description of type unit entries.

.. option:: -x, --regex

            Treat any <pattern> strings as regular expressions when searching
            instead of just as an exact string match.

.. option:: -u, --uuid

            Show the UUID for each architecture.

.. option:: --diff

            Dump the output in a format that is more friendly for comparing
            DWARF output from two different files.

.. option:: -v, --verbose

            Display verbose information when dumping. This can help to debug
            DWARF issues.

.. option:: --verify

            Verify the structure of the DWARF information by verifying the
            compile unit chains, DIE relationships graph, address
            ranges, and more.

.. option:: --version

            Display the version of the tool.

.. option:: --debug-abbrev, --debug-addr, --debug-aranges, --debug-cu-index, --debug-frame [=<offset>], --debug-gnu-pubnames, --debug-gnu-pubtypes, --debug-info [=<offset>], --debug-line [=<offset>], --debug-line-str, --debug-loc [=<offset>], --debug-loclists [=<offset>], --debug-macro, --debug-names, --debug-pubnames, --debug-pubtypes, --debug-ranges, --debug-rnglists, --debug-str, --debug-str-offsets, --debug-tu-index, --debug-types, --eh-frame [=<offset>], --gdb-index, --apple-names, --apple-types, --apple-namespaces, --apple-objc

            Dump the specified DWARF section by name. Only the
            `.debug_info` section is shown by default. Some entries
            support adding an `=<offset>` as a way to provide an
            optional offset of the exact entry to dump within the
            respective section. When an offset is provided, only the
            entry at that offset will be dumped, else the entire
            section will be dumped.

.. option:: @<FILE>

            Read command-line options from `<FILE>`.

EXIT STATUS
-----------

:program:`llvm-dwarfdump` returns 0 if the input files were parsed and dumped
successfully. Otherwise, it returns 1.

SEE ALSO
--------

:manpage:`dsymutil(1)`
