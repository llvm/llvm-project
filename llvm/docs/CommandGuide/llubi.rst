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

.. option:: -seed=N

  Set the seed for random number generator to N. By default, the seed is 0.

.. option:: -undef-behavior=mode

  Set the behavior for undefined values (e.g., load from uninitialized memory or freeze a poison value).
  The options for `mode` are:

  * `nondet`: Each load from the same uninitialized byte yields a freshly random value. This is the default behavior.
  * `zero`: Uninitialized values are treated as zero.

.. option:: -nan-behavior=mode

  Set the behavior for payload preserving behavior of floating-point NaN values.
  The options for `mode` are:

  * `nondet`: The actual behavior is randomly chosen from the modes below. This is the default behavior.
  * `preferred`: The quiet bit is set and the payload is all-zero.
  * `quieting`: The quiet bit is set and the payload is copied from any input operand that is a NaN.
  * `unchanged`: The quiet bit and payload are copied from any input operand that is a NaN.
  * `target-specific`: The quiet bit is set and the payload is picked from a known target-specific set of extra possible NaN payloads.

.. option:: -deterministic

  Disable interpreter-introduced non-determinism (off by default).
  This option implies '``-undef-behavior=zero``' and '``-nan-behavior=preferred``'.

.. option:: -fuse-fmuladd

  Treat '``llvm.fmuladd.*``' as '``llvm.fma.*``'. It is the default behavior.
  Otherwise, it is expanded into a \* b + c.

.. option:: -disable-verify

  Disable the validation of the input LLVM IR. The user is responsible for the validity of inputs.
  The verifier is executed by default.

EXIT STATUS
-----------

If :program:`llubi` fails to load the program, or an error occurs during execution (e.g, an immediate undefined
behavior is triggered), it will exit with an exit code of 1.
If the return type of entry function is not an integer type, it will return 0.
Otherwise, it will return the exit code of the program.
