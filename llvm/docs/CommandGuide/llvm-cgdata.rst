llvm-cgdata - LLVM CodeGen Data Tool
====================================

.. program:: llvm-cgdata

SYNOPSIS
--------

:program:`llvm-cgdata` [**commands**] [**options**] (<binaries>|<.cgdata>)

DESCRIPTION
-----------

The :program:llvm-cgdata utility parses raw codegen data embedded in compiled
binary files and merges them into a single .cgdata file. It can also inspect
and manipulate .cgdata files. Currently, the tool supports saving and restoring
outlined hash trees and stable function maps, allowing for more efficient
function outlining and function merging across modules in subsequent
compilations. The design is extensible, allowing for the incorporation of
additional codegen summaries and optimization techniques.

COMMANDS
--------

At least one of the following commands are required:

.. option:: --convert

  Convert a .cgdata file from one format to another.

.. option:: --merge

  Merge multiple raw codgen data in binaries into a single .cgdata file.

.. option:: --show

  Show summary information about a .cgdata file.

OPTIONS
-------

:program:`llvm-cgdata` supports the following options:

.. option:: --format=[text|binary]

  Specify the format of the output .cgdata file.

.. option:: --output=<string>

  Specify the output file name.

.. option:: --cgdata-version

  Print the version of the llvm-cgdata tool.

EXAMPLES
--------

To convert a .cgdata file from binary to text format:
    $ llvm-cgdata --convert --format=text input.cgdata --output=output.data

To merge multiple raw codegen data in object files into a single .cgdata file:
    $ llvm-cgdata --merge file1.o file2.o --output=merged.cgdata

To show summary information about a .cgdata file:
    $ llvm-cgdata --show input.cgdata
