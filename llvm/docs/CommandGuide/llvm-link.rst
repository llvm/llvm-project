llvm-link - LLVM bitcode linker
===============================

SYNOPSIS
--------

:program:`llvm-link` [*options*] *filename ...*

DESCRIPTION
-----------

:program:`llvm-link` takes several LLVM bitcode files and links them together
into a single LLVM bitcode file.  It writes the output file to standard output,
unless the :option:`-o` option is used to specify a filename.

OPTIONS
-------

.. option:: -f

 Enable binary output on terminals.  Normally, :program:`llvm-link` will refuse
 to write raw bitcode output if the output stream is a terminal. With this
 option, :program:`llvm-link` will write raw bitcode regardless of the output
 device.

.. option:: -o filename

 Specify the output file name.  If ``filename`` is "``-``", then
 :program:`llvm-link` will write its output to standard output.

.. option:: -S

 Write output in LLVM intermediate language (instead of bitcode).

.. option:: -d

 If specified, :program:`llvm-link` prints a human-readable version of the
 output bitcode file to standard error.

.. option:: -help

 Print a summary of command line options.

.. option:: -v

 Verbose mode.  Print information about what :program:`llvm-link` is doing.
 This typically includes a message for each bitcode file linked in and for each
 library found.

EXIT STATUS
-----------

If :program:`llvm-link` succeeds, it will exit with 0.  Otherwise, if an error
occurs, it will exit with a non-zero value.


