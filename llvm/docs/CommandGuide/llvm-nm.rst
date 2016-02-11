llvm-nm - list LLVM bitcode and object file's symbol table
==========================================================

SYNOPSIS
--------

:program:`llvm-nm` [*options*] [*filenames...*]

DESCRIPTION
-----------

The :program:`llvm-nm` utility lists the names of symbols from the LLVM bitcode
files, object files, or :program:`ar` archives containing them, named on the
command line.  Each symbol is listed along with some simple information about
its provenance.  If no file name is specified, or *-* is used as a file name,
:program:`llvm-nm` will process a file on its standard input stream.

:program:`llvm-nm`'s default output format is the traditional BSD :program:`nm`
output format.  Each such output record consists of an (optional) 8-digit
hexadecimal address, followed by a type code character, followed by a name, for
each symbol.  One record is printed per line; fields are separated by spaces.
When the address is omitted, it is replaced by 8 spaces.

Type code characters currently supported, and their meanings, are as follows:

U

 Named object is referenced but undefined in this bitcode file

C

 Common (multiple definitions link together into one def)

W

 Weak reference (multiple definitions link together into zero or one definitions)

t

 Local function (text) object

T

 Global function (text) object

d

 Local data object

D

 Global data object

?

 Something unrecognizable

Because LLVM bitcode files typically contain objects that are not considered to
have addresses until they are linked into an executable image or dynamically
compiled "just-in-time", :program:`llvm-nm` does not print an address for any
symbol in an LLVM bitcode file, even symbols which are defined in the bitcode
file.

OPTIONS
-------

.. program:: llvm-nm

.. option:: -B    (default)

 Use BSD output format.  Alias for :option:`--format=bsd`.

.. option:: -P

 Use POSIX.2 output format.  Alias for :option:`--format=posix`.

.. option:: --debug-syms, -a

 Show all symbols, even debugger only.

.. option:: --defined-only

 Print only symbols defined in this file (as opposed to
 symbols which may be referenced by objects in this file, but not
 defined in this file.)

.. option:: --dynamic, -D

 Display dynamic symbols instead of normal symbols.

.. option:: --extern-only, -g

 Print only symbols whose definitions are external; that is, accessible
 from other files.

.. option:: --format=format, -f format

 Select an output format; *format* may be *sysv*, *posix*, or *bsd*.  The default
 is *bsd*.

.. option:: -help

 Print a summary of command-line options and their meanings.

.. option:: --no-sort, -p

 Shows symbols in order encountered.

.. option:: --numeric-sort, -n, -v

 Sort symbols by address.

.. option:: --print-file-name, -A, -o

 Precede each symbol with the file it came from.

.. option:: --print-size, -S

 Show symbol size instead of address.

.. option:: --size-sort

 Sort symbols by size.

.. option:: --undefined-only, -u

 Print only symbols referenced but not defined in this file.

.. option:: --radix=RADIX, -t

 Specify the radix of the symbol address(es). Values accepted d(decimal),
 x(hexadecomal) and o(octal).

BUGS
----

 * :program:`llvm-nm` cannot demangle C++ mangled names, like GNU :program:`nm`
   can.

 * :program:`llvm-nm` does not support the full set of arguments that GNU
   :program:`nm` does.

EXIT STATUS
-----------

:program:`llvm-nm` exits with an exit code of zero.

SEE ALSO
--------

llvm-dis, ar(1), nm(1)
