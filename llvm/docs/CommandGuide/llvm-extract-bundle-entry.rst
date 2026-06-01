llvm-extract-bundle-entry - extract an offload bundle entry
===========================================================

.. program:: llvm-extract-bundle-entry

SYNOPSIS
--------

:program:`llvm-extract-bundle-entry` [*options*] URI

DESCRIPTION
-----------

:program:`llvm-extract-offload-entry` is a tool thet takes a URI argument and 
generates a code object file by extracting an offload bundle entry specified
by the URI given.

The URI syntax is defined as:

-- code-block::bnf
 <code_object_uri> ::== <file_uri> | <memory_uri>
 <file_uri>        ::== "file://"<extract_file><range_specifier>
 <memory_uri>      ::== "memory://"<process_id><range_specifier>
 <range_specifier> ::== [ "#" | "?" ]"offset="<number>"&size="<number>
 <extract_file>    ::== URI_ENCODED_OS_FILE_PATH
 <process_id>      ::== DECIMAL_NUMBER
 <number>          ::== DECIMAL_NUMBER
 
The output is always written to a file, whose name is generated from the URI input given.

OPTIONS
-------

The following options are either agnostic of the file format or apply to
multiple file formats.

.. option:: --help, -h

 Print a summary of command line options.

.. option::  -o <file>

 Write output to <file>. Multiple input files cannot be used in combination
 with -o.

.. option:: --version, -V

 Display the version of the :program:`llvm-extract-bundle-entry` executable.

EXIT STATUS
-----------

:program:`llvm-extract-bundle-entry` exits with a non-zero exit code if there is an error.
Otherwise, it exits with code 0.

BUGS
----

To report bugs, please visit <https://github.com/llvm/llvm-project/issues?q=state%3Aopen%20label%3Allvm-extract-bundle-entry>.
