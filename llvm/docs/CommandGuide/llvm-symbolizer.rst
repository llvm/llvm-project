llvm-symbolizer - convert addresses into source code locations
==============================================================

SYNOPSIS
--------

:program:`llvm-symbolizer` [options]

DESCRIPTION
-----------

:program:`llvm-symbolizer` reads object file names and addresses from standard
input and prints corresponding source code locations to standard output.
If object file is specified in command line, :program:`llvm-symbolizer` reads
only addresses from standard input. This
program uses debug info sections and symbol table in the object files.

EXAMPLE
--------

.. code-block:: console

  $ cat addr.txt
  a.out 0x4004f4
  /tmp/b.out 0x400528
  /tmp/c.so 0x710
  /tmp/mach_universal_binary:i386 0x1f84
  /tmp/mach_universal_binary:x86_64 0x100000f24
  $ llvm-symbolizer < addr.txt
  main
  /tmp/a.cc:4
  
  f(int, int)
  /tmp/b.cc:11

  h_inlined_into_g
  /tmp/header.h:2
  g_inlined_into_f
  /tmp/header.h:7
  f_inlined_into_main
  /tmp/source.cc:3
  main
  /tmp/source.cc:8

  _main
  /tmp/source_i386.cc:8

  _main
  /tmp/source_x86_64.cc:8
  $ cat addr2.txt
  0x4004f4
  0x401000
  $ llvm-symbolizer -obj=a.out < addr2.txt
  main
  /tmp/a.cc:4

  foo(int)
  /tmp/a.cc:12
  $cat addr.txt
  0x40054d
  $llvm-symbolizer -inlining -print-address -pretty-print -obj=addr.exe < addr.txt
  0x40054d: inc at /tmp/x.c:3:3
   (inlined by) main at /tmp/x.c:9:0
  $llvm-symbolizer -inlining -pretty-print -obj=addr.exe < addr.txt
  inc at /tmp/x.c:3:3
   (inlined by) main at /tmp/x.c:9:0

OPTIONS
-------

.. option:: -obj

  Path to object file to be symbolized.

.. option:: -functions=[none|short|linkage]

  Specify the way function names are printed (omit function name,
  print short function name, or print full linkage name, respectively).
  Defaults to ``linkage``.

.. option:: -use-symbol-table

 Prefer function names stored in symbol table to function names
 in debug info sections. Defaults to true.

.. option:: -demangle

 Print demangled function names. Defaults to true.

.. option:: -inlining 

 If a source code location is in an inlined function, prints all the
 inlnied frames. Defaults to true.

.. option:: -default-arch

 If a binary contains object files for multiple architectures (e.g. it is a
 Mach-O universal binary), symbolize the object file for a given architecture.
 You can also specify architecture by writing ``binary_name:arch_name`` in the
 input (see example above). If architecture is not specified in either way,
 address will not be symbolized. Defaults to empty string.

.. option:: -dsym-hint=<path/to/file.dSYM>

 (Darwin-only flag). If the debug info for a binary isn't present in the default
 location, look for the debug info at the .dSYM path provided via the
 ``-dsym-hint`` flag. This flag can be used multiple times.

.. option:: -print-address
 Print address before the source code location. Defaults to false.

.. option:: -pretty-print
 Print human readable output. If ``-inlining`` is specified, enclosing scope is
 prefixed by (inlined by). Refer to listed examples.

EXIT STATUS
-----------

:program:`llvm-symbolizer` returns 0. Other exit codes imply internal program error.
