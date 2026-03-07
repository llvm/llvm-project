llubi - LLVM UB-aware Interpreter
=================================

.. program:: llubi

SYNOPSIS
--------

:program:`llubi` [*options*] [*filename*] [*program args*]

DESCRIPTION
-----------

:program:`llubi` directly executes programs in LLVM bitcode format and tracks values in LLVM IR semantics.
Unlike :program:`lli`, :program:`llubi` is designed to be aware of undefined behaviors during execution.
It detects immediate undefined behaviors such as integer division by zero, and respects poison generating flags
like `nsw` and `nuw`. As it captures most of the guardable undefined behaviors, it is highly suitable for
constructing an interesting-ness test for miscompilation bugs.

If `filename` is not specified, then :program:`llubi` reads the LLVM bitcode for the
program from standard input.

The optional *args* specified on the command line are passed to the program as
arguments.

GENERAL OPTIONS
---------------

.. option:: -fake-argv0=executable

 Override the ``argv[0]`` value passed into the executing program.

.. option:: -entry-function=function

 Specify the name of the function to execute as the program's entry point.
 By default, :program:`llubi` uses the function named ``main``.

.. option:: -help

 Print a summary of command line options.

.. option:: -verbose

 Print results for each instruction executed.

.. option:: -version

 Print out the version of :program:`llubi` and exit without doing anything else.

INTERPRETER OPTIONS
-------------------

.. option:: -max-mem=N

  Limit the amount of memory (in bytes) that can be allocated by the program, including
  stack, heap, and global variables. If the limit is exceeded, execution will be terminated.
  By default, there is no limit (N = 0).

.. option:: -max-stack-depth=N

  Limit the maximum stack depth to N. If the limit is exceeded, execution will be terminated.
  The default limit is 256. Set N to 0 to disable the limit.

.. option:: -max-steps=N

  Limit the number of instructions executed to N. If the limit is reached, execution will
  be terminated. By default, there is no limit (N = 0).

.. option:: -vscale=N

  Set the value of `llvm.vscale` to N. The default value is 4.

EXIT STATUS
-----------

If :program:`llubi` fails to load the program, or an error occurs during execution (e.g, an immediate undefined
behavior is triggered), it will exit with an exit code of 1.
If the return type of entry function is not an integer type, it will return 0.
Otherwise, it will return the exit code of the program.
