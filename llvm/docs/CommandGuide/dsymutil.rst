dsymutil - manipulate archived DWARF debug symbol files
=======================================================

.. program:: dsymutil

SYNOPSIS
--------

| :program:`dsymutil` [*options*] *executable*

DESCRIPTION
-----------

:program:`dsymutil` links the DWARF debug information found in the object files
for an executable *executable* by using debug symbols information contained in
its symbol table. By default, the linked debug information is placed in a
``.dSYM`` bundle with the same name as the executable.

OPTIONS
-------
.. option:: --arch=<arch>

 Link DWARF debug information only for specified CPU architecture types.
 Architectures may be specified by name. When using this option, an error will
 be returned if any architectures can not be properly linked.  This option can
 be specified multiple times, once for each desired architecture. All CPU
 architectures will be linked by default and any architectures that can't be
 properly linked will cause :program:`dsymutil` to return an error.

.. option:: --dump-debug-map

 Dump the *executable*'s debug-map (the list of the object files containing the
 debug information) in YAML format and exit. Not DWARF link will take place.

.. option:: -f, --flat

 Produce a flat dSYM file. A ``.dwarf`` extension will be appended to the
 executable name unless the output file is specified using the -o option.

.. option:: -z, --minimize

 When used when creating a dSYM file, this option will suppress the emission of
 the .debug_inlines, .debug_pubnames, and .debug_pubtypes sections since
 dsymutil currently has better equivalents: .apple_names and .apple_types. When
 used in conjunction with --update option, this option will cause redundant
 accelerator tables to be removed.

.. option:: --no-odr

 Do not use ODR (One Definition Rule) for uniquing C++ types.

.. option:: --no-output

 Do the link in memory, but do not emit the result file.

.. option:: --no-swiftmodule-timestamp

 Don't check the timestamp for swiftmodule files.

.. option:: -j <n>, --num-threads=<n>

 Specifies the maximum number (``n``) of simultaneous threads to use when
 linking multiple architectures.

.. option:: -o <filename>

 Specifies an alternate ``path`` to place the dSYM bundle. The default dSYM
 bundle path is created by appending ``.dSYM`` to the executable name.

.. option:: --oso-prepend-path=<path>

 Specifies a ``path`` to prepend to all debug symbol object file paths.

.. option:: --object-prefix-map=<prefix=remapped>

 Remap object file paths (but no source paths) before processing.  Use
 this for Clang objects where the module cache location was remapped using
 ``-fdebug-prefix-map``; to help dsymutil find the Clang module cache.
 
.. option:: --papertrail

 When running dsymutil as part of your build system, it can be desirable for
 warnings to be part of the end product, rather than just being emitted to the
 output stream. When enabled warnings are embedded in the linked DWARF debug
 information.

.. option:: -s, --symtab

 Dumps the symbol table found in *executable* or object file(s) and exits.

.. option:: --toolchain

 Embed the toolchain in the dSYM bundle's property list.

.. option:: -u, --update

 Update an existing dSYM file to contain the latest accelerator tables and
 other DWARF optimizations. This option will rebuild the '.apple_names' and
 '.apple_types' hashed accelerator tables.

.. option:: -v, --verbose

 Display verbose information when linking.

.. option:: --version

 Display the version of the tool.

.. option:: -y

 Treat *executable* as a YAML debug-map rather than an executable.

EXIT STATUS
-----------

:program:`dsymutil` returns 0 if the DWARF debug information was linked
successfully. Otherwise, it returns 1.

SEE ALSO
--------

:manpage:`llvm-dwarfdump(1)`
