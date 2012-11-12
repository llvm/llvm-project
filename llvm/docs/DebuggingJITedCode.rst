.. _debugging-jited-code:

==============================
Debugging JIT-ed Code With GDB
==============================

.. sectionauthor:: Reid Kleckner and Eli Bendersky

Background
==========

Without special runtime support, debugging dynamically generated code with
GDB (as well as most debuggers) can be quite painful.  Debuggers generally
read debug information from the object file of the code, but for JITed
code, there is no such file to look for.

In order to communicate the necessary debug info to GDB, an interface for
registering JITed code with debuggers has been designed and implemented for
GDB and LLVM MCJIT.  At a high level, whenever MCJIT generates new machine code,
it does so in an in-memory object file that contains the debug information in
DWARF format.  MCJIT then adds this in-memory object file to a global list of
dynamically generated object files and calls a special function
(``__jit_debug_register_code``) marked noinline that GDB knows about.  When
GDB attaches to a process, it puts a breakpoint in this function and loads all
of the object files in the global list.  When MCJIT calls the registration
function, GDB catches the breakpoint signal, loads the new object file from
the inferior's memory, and resumes the execution.  In this way, GDB can get the
necessary debug information.

GDB Version
===========

In order to debug code JIT-ed by LLVM, you need GDB 7.0 or newer, which is
available on most modern distributions of Linux.  The version of GDB that
Apple ships with Xcode has been frozen at 6.3 for a while.  LLDB may be a
better option for debugging JIT-ed code on Mac OS X.


Debugging MCJIT-ed code
=======================

The emerging MCJIT component of LLVM allows full debugging of JIT-ed code with
GDB.  This is due to MCJIT's ability to use the MC emitter to provide full
DWARF debugging information to GDB.

Note that lli has to be passed the ``-use-mcjit`` flag to JIT the code with
MCJIT instead of the old JIT.

Example
-------

Consider the following C code (with line numbers added to make the example
easier to follow):

..
   FIXME:
   Sphinx has the ability to automatically number these lines by adding
   :linenos: on the line immediately following the `.. code-block:: c`, but
   it looks like garbage; the line numbers don't even line up with the
   lines. Is this a Sphinx bug, or is it a CSS problem?

.. code-block:: c

   1   int compute_factorial(int n)
   2   {
   3       if (n <= 1)
   4           return 1;
   5
   6       int f = n;
   7       while (--n > 1)
   8           f *= n;
   9       return f;
   10  }
   11
   12
   13  int main(int argc, char** argv)
   14  {
   15      if (argc < 2)
   16          return -1;
   17      char firstletter = argv[1][0];
   18      int result = compute_factorial(firstletter - '0');
   19
   20      // Returned result is clipped at 255...
   21      return result;
   22  }

Here is a sample command line session that shows how to build and run this
code via ``lli`` inside GDB:

.. code-block:: bash

   $ $BINPATH/clang -cc1 -O0 -g -emit-llvm showdebug.c
   $ gdb --quiet --args $BINPATH/lli -use-mcjit showdebug.ll 5
   Reading symbols from $BINPATH/lli...done.
   (gdb) b showdebug.c:6
   No source file named showdebug.c.
   Make breakpoint pending on future shared library load? (y or [n]) y
   Breakpoint 1 (showdebug.c:6) pending.
   (gdb) r
   Starting program: $BINPATH/lli -use-mcjit showdebug.ll 5
   [Thread debugging using libthread_db enabled]

   Breakpoint 1, compute_factorial (n=5) at showdebug.c:6
   6	    int f = n;
   (gdb) p n
   $1 = 5
   (gdb) p f
   $2 = 0
   (gdb) n
   7	    while (--n > 1)
   (gdb) p f
   $3 = 5
   (gdb) b showdebug.c:9
   Breakpoint 2 at 0x7ffff7ed404c: file showdebug.c, line 9.
   (gdb) c
   Continuing.

   Breakpoint 2, compute_factorial (n=1) at showdebug.c:9
   9	    return f;
   (gdb) p f
   $4 = 120
   (gdb) bt
   #0  compute_factorial (n=1) at showdebug.c:9
   #1  0x00007ffff7ed40a9 in main (argc=2, argv=0x16677e0) at showdebug.c:18
   #2  0x3500000001652748 in ?? ()
   #3  0x00000000016677e0 in ?? ()
   #4  0x0000000000000002 in ?? ()
   #5  0x0000000000d953b3 in llvm::MCJIT::runFunction (this=0x16151f0, F=0x1603020, ArgValues=...) at /home/ebenders_test/llvm_svn_rw/lib/ExecutionEngine/MCJIT/MCJIT.cpp:161
   #6  0x0000000000dc8872 in llvm::ExecutionEngine::runFunctionAsMain (this=0x16151f0, Fn=0x1603020, argv=..., envp=0x7fffffffe040)
       at /home/ebenders_test/llvm_svn_rw/lib/ExecutionEngine/ExecutionEngine.cpp:397
   #7  0x000000000059c583 in main (argc=4, argv=0x7fffffffe018, envp=0x7fffffffe040) at /home/ebenders_test/llvm_svn_rw/tools/lli/lli.cpp:324
   (gdb) finish
   Run till exit from #0  compute_factorial (n=1) at showdebug.c:9
   0x00007ffff7ed40a9 in main (argc=2, argv=0x16677e0) at showdebug.c:18
   18	    int result = compute_factorial(firstletter - '0');
   Value returned is $5 = 120
   (gdb) p result
   $6 = 23406408
   (gdb) n
   21	    return result;
   (gdb) p result
   $7 = 120
   (gdb) c
   Continuing.

   Program exited with code 0170.
   (gdb)
