llvm-libtool-darwin - LLVM tool for creating libraries for Darwin
=================================================================

.. program:: llvm-libtool-darwin

SYNOPSIS
--------

:program:`llvm-libtool-darwin` [*options*] *<input files>*

DESCRIPTION
-----------

:program:`llvm-libtool-darwin` is a tool for creating static and dynamic
libraries for Darwin.

For most scenarios, it works as a drop-in replacement for cctools'
:program:`libtool`.

OPTIONS
--------
:program:`llvm-libtool-darwin` supports the following options:

.. option:: -h, -help

  Show help and usage for this command.

.. option:: -help-list

  Show help and usage for this command without grouping the options
  into categories.

.. option:: -color

  Use colors in output.

.. option:: -version

  Display the version of this program.

.. option:: -D

 Use zero for timestamps and UIDs/GIDs. This is set by default.

.. option:: -U

 Use actual timestamps and UIDs/GIDs.

.. option:: -o <filename>

  Specify the output file name. Must be specified exactly once.

.. option:: -static

 Produces a static library from the input files.

.. option:: -filelist <listfile[,dirname]>

 Read input file names from `<listfile>`. File names are specified in `<listfile>`
 one per line, separated only by newlines. Whitespace on a line is assumed
 to be part of the filename. If the directory name, `dirname`, is also
 specified then it is prepended to each file name in the `<listfile>`.

EXIT STATUS
-----------

:program:`llvm-libtool-darwin` exits with a non-zero exit code if there is an error.
Otherwise, it exits with code 0.

BUGS
----

To report bugs, please visit <https://bugs.llvm.org/>.

SEE ALSO
--------

:manpage:`llvm-ar(1)`
