============================
Running Tools in Daemon Mode
============================

.. contents::
   :local:

Introduction
============

Process creation incurs a significant overhead, especially on Windows. This
document describes the "daemon mode" execution model for LLVM tools, where
tools run as persistent daemon processes that can perform many tasks, with
different arguments and input, in the same process. The original motivation
for this work is to reduce Lit testing time by replacing tool invocations with
commands for the daemon - this means that the daemonized tool must be able to
produce the exact same output as the regular version of the tool.

Adding Daemon Support to a Tool
===============================

Tools can support daemon mode by implementing the ``LLVMTool`` interface and
calling ``runWithDaemonSupport(Tool)`` in the main function.

First, the tool's ``main`` function must be refactored into a class implementing
the ``LLVMTool`` interface. This has two functions, ``run`` and ``resetState``.
The majority of the body of main (everything aside from ``InitLLVM`` and other
one-time initialization) must be moved to ``run``. This is the function that is
called by the daemon for each invocation, and it is also invoked once when the
tool is run normally. The handling of standard input must also be reworked so that
the ``StandardInputSource`` provided to ``run`` is respected as standard input.
For example, ``MemoryBuffer::getSTDIN`` should be replaced by
``StandardInputSource::getInput()`` and ``MemoryBuffer::getFileOrStdin`` should
be replaced by ``StandardInputSource::getFileOrInput()``. This is how the daemon
provides input to the tool, as the input cannot be read from stdin normally
because the pipe never closes.

``resetState`` must also be implemented, which is responsible for resetting any
application state, for example command line options, statistics and debug
counters, for the next invocation. This is not called when the process is run
normally. Any persistent state which may affect the output and is not reset by
``resetState`` will cause flaky tests.

Finally, the contents of main that were refactored into ``run`` can be replaced
by ``runWithDaemonSupport(Tool)``, which will detect the daemon argument and
either run the tool in daemon mode or run it normally by deferring to ``run``.

Daemon IPC Protocol
===================

The communication protocol between the daemon and the Lit tester uses four
pipes: the daemon's ``stdin``, ``stdout`` and ``stderr`` and another pipe,
called the `status pipe`, for the daemon to communicate responses to Lit.

When the daemon first starts it will send a ``ready`` response to indicate that
it has initialized correctly and is ready to receive commands.

The daemon accepts `commands` on ``stdin``. Commands must be separated by a
newline (\n; \r is ignored). The following commands are accepted:

* ``run``: This command takes no arguments. Upon receiving this command, the
  daemon will run the tool, whose output will be sent on ``stdout`` and ``stderr``
  as usual (unless ``stderr`` is redirected). When the tool returns, a
  ``returned`` response indicating the exit code is sent.

* ``arg``: This command is followed by an integer indicating a number of
  bytes. The daemon will read this many bytes from ``stdin``; this string will
  be appended to the list of arguments for the next invocation. Arguments are
  framed in this way to avoid having to worry about escaping a separator char.

* ``input_string``: This command is followed by an integer indicating a number of
  bytes. The daemon will read this many bytes from ``stdin``; this string will
  be used as standard input for the next invocation.

* ``input_file``: This command is followed by a file name. The content of this
  file will be used as standard input for the next invocation.

* ``cd``: This command is followed by a directory name. The current directory
  for the daemon is changed to the provided directory until the next invocation
  finishes. This controls the working directory for the tool, and following
  ``input_file`` and ``cd`` commands until the next ``run``. start from this
  directory too.

* ``redirect_stderr_to_stdout``: This causes writes to ``stderr`` to be
  directed through ``stdout`` for the next invocation.

* ``exit``: The daemon will exit.

The daemon may send the following `responses` along the status
pipe:

* ``ready``: This is sent when the daemon finishes initialization, to
  indicate it is ready to receive commands.

* ``error``: This is sent when the daemon encounters an error, for example a
  badly formed command. The daemon will exit after encountering an error. The
  managing process should re-raise the error, as this indicates incorrect use of
  the daemon. The response is followed by a string describing the error.

* ``returned``. This is sent when a task finishes executing. The response is
  followed by an integer representing the exit code from the task.

If the daemon exits unexpectedly while running the tool, this means that the
tool itself caused the process to exit. The exit code from the daemon should be
taken as the exit code for the task, and the daemon should be restarted.

Each command and response has a space before its argument. An example exchange
may look like:

* Command: ``arg 3\\n`` followed by ``opt``

* Command: ``arg 2\\n`` followed by ``-S``

* Command: ``arg 20\\n`` followed by ``--passes=instcombine``

* Command: ``input_file llvm/test/Transforms/InstCombine/range-check.ll\\n``

* Command: ``run\\n``

* (``stdout`` and ``stderr`` are sent as usual.)

* Response: ``returned 0\\n``

* Command: ``bad command\\n``

* Response: ``error Unexpected command: bad command\\n``

Running a Daemon
================

Tools are invoked in daemon mode by passing ``--daemon`` as the first command
line argument. Additionally, ``--daemon-status-pipe`` argument must be provided
to set the file descriptor or Windows file handle on which the status messages
are sent. This may take the form ``fd:{N}`` indicating a Unix/CRT file descriptor
or ``handle:{N}`` indicating a Windows file handle.
