llvm-install-name-tool - LLVM tool for manipulating install-names and rpaths
============================================================================

.. program:: llvm-install-name-tool

SYNOPSIS
--------

:program:`llvm-install-name-tool` [*options*] *input*

DESCRIPTION
-----------

:program:`llvm-install-name-tool` is a tool to manipulate dynamic shared library
install names and rpaths listed in a Mach-O binary.

For most scenarios, it works as a drop-in replacement for Apple's
:program:`install_name_tool`.

OPTIONS
--------
At least one of the following options are required, and some options can be
combined with other options:

.. option:: -add_rpath <rpath>

 Add an rpath named ``<rpath>`` to the specified binary. Can be specified multiple
 times to add multiple rpaths. Throws an error if ``<rpath>`` is already listed in
 the binary.

.. option:: -delete_rpath <rpath>

 Delete an rpath named ``<rpath>`` from the specified binary. Can be specified multiple
 times to delete multiple rpaths. Throws an error if ``<rpath>`` is not listed in
 the binary.

EXIT STATUS
-----------

:program:`llvm-install-name-tool` exits with a non-zero exit code of 1 if there is an error.
Otherwise, it exits with code 0.

BUGS
----

To report bugs, please visit <https://bugs.llvm.org/>.

SEE ALSO
--------

:manpage:`llvm-objcopy(1)`
