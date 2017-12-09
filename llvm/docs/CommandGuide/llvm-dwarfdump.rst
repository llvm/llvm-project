llvm-dwarfdump - dump and verify DWARF debug information
========================================================

SYNOPSIS
--------

:program:`llvm-dwarfdump` [*options*] [*filename ...*]

DESCRIPTION
-----------

:program:`llvm-dwarfdump` parses DWARF sections in object files,
archives, and `.dSYM` bundles and prints their contents in
human-readable form. Only the .debug_info section is printed unless one of
the section-specific options or :option:`--all` is specified.

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

            Show a debug info entry's children when using
            the :option:`--debug-info`, :option:`--find`,
            and :option:`--name` options.

.. option:: --diff

            Emit the output in a diff-friendly way by omitting offsets and
            addresses.

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

.. option:: -i, --ignore-case

            Ignore case distinctions in when searching entries by name
            or by regular expression.

.. option:: -n <pattern>, --name=<pattern>

            Find and print all debug info entries whose name
            (`DW_AT_name` attribute) matches the exact text in
            <pattern>. Use the :option:`--regex` option to have
            <pattern> become a regular expression for more flexible
            pattern matching.

.. option:: --lookup=<address>

            Lookup <address> in the debug information and print out the file,
            function, block, and line table details.

.. option:: -o <path>, --out-file=<path>

            Redirect output to a file specified by <path>.

.. option:: -p, --show-parents

            Show a debug info entry's parent objects when using the
            :option:`--debug-info`, :option:`--find`, and
            :option:`--name` options.

.. option:: -r <n>, --recurse-depth=<n>

            Only recurse to a maximum depth of <n> when dumping debug info
            entries.

.. option:: --statistics

            Collect debug info quality metrics and print the results
            as machine-readable single-line JSON output.

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

.. option:: --debug-abbrev, --debug-aranges, --debug-cu-index, --debug-frame [=<offset>], --debug-gnu-pubnames, --debug-gnu-pubtypes, --debug-info [=<offset>], --debug-line [=<offset>], --debug-loc [=<offset>], --debug-macro, --debug-pubnames, --debug-pubtypes, --debug-ranges, --debug-str, --debug-str-offsets, --debug-tu-index, --debug-types, --eh-frame, --gdb-index, --apple-names, --apple-types, --apple-namespaces, --apple-objc

            Dump the specified DWARF section by name. Only the
            `.debug_info` section is shown by default. Some entries
            support adding an `=<offset>` as a way to provide an
            optional offset of the exact entry to dump within the
            respective section. When an offset is provided, only the
            entry at that offset will be dumped, else the entire
            section will be dumped. Children of items at a specific
            offset can be dumped by also using the
            :option:`--show-children` option where applicable.

EXIT STATUS
-----------

:program:`llvm-dwarfdump` returns 0 if the input files were parsed and dumped
successfully. Otherwise, it returns 1.

SEE ALSO
--------

:manpage:`dsymutil(1)`
