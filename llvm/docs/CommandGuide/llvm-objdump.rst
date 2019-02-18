llvm-objdump - LLVM's object file dumper
========================================

SYNOPSIS
--------

:program:`llvm-objdump` [*commands*] [*options*] [*filenames...*]

DESCRIPTION
-----------
The :program:`llvm-objdump` utility prints the contents of object files and
final linked images named on the command line. If no file name is specified,
:program:`llvm-objdump` will attempt to read from *a.out*. If *-* is used as a
file name, :program:`llvm-objdump` will process a file on its standard input
stream.

COMMANDS
--------
At least one of the following commands are required, and some commands can be combined with other commands:

.. option:: -disassemble

  Display assembler mnemonics for the machine instructions
 
.. option:: -help

  Display usage information and exit. Does not stack with other commands.

.. option:: -r

  Display the relocation entries in the file.

.. option:: -s

  Display the content of each section.

.. option:: -section-headers

  Display summaries of the headers for each section.

.. option:: -t

  Display the symbol table.

.. option:: -version

  Display the version of this program. Does not stack with other commands.
  
OPTIONS
-------
:program:`llvm-objdump` supports the following options:

.. option:: -arch=<architecture>

  Specify the architecture to disassemble. see -version for available
  architectures.

.. option:: -cfg

  Create a CFG for every symbol in the object file and write it to a graphviz
  file (Mach-O-only).

.. option:: -dsym=<string>

  Use .dSYM file for debug info.

.. option:: -g

  Print line information from debug info if available.

.. option:: -macho

  Use Mach-O specific object file parser.

.. option:: -mattr=<a1,+a2,-a3,...>

  Target specific attributes.
  
.. option:: -mc-x86-disable-arith-relaxation

  Disable relaxation of arithmetic instruction for X86.

.. option:: -stats

  Enable statistics output from program.
  
.. option:: -triple=<string>

  Target triple to disassemble for, see -version for available targets.
  
.. option:: -x86-asm-syntax=<style>

  When used with the ``-disassemble`` option, choose style of code to emit from
  X86 backend. Supported values are:

   .. option:: att
   
    AT&T-style assembly
   
   .. option:: intel
   
    Intel-style assembly

   
  The default disassembly style is **att**. 

BUGS
----

To report bugs, please visit <http://llvm.org/bugs/>.

SEE ALSO
--------

:manpage:`llvm-nm(1)`
