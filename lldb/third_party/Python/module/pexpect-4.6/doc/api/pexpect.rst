Core pexpect components
=======================

.. automodule:: pexpect

spawn class
-----------

.. autoclass:: spawn

   .. automethod:: __init__
   .. automethod:: expect
   .. automethod:: expect_exact
   .. automethod:: expect_list
   .. automethod:: compile_pattern_list
   .. automethod:: send
   .. automethod:: sendline
   .. automethod:: write
   .. automethod:: writelines
   .. automethod:: sendcontrol
   .. automethod:: sendeof
   .. automethod:: sendintr
   .. automethod:: read
   .. automethod:: readline
   .. automethod:: read_nonblocking
   .. automethod:: eof
   .. automethod:: interact

   .. attribute:: logfile
                  logfile_read
                  logfile_send

      Set these to a Python file object (or :data:`sys.stdout`) to log all
      communication, data read from the child process, or data sent to the child
      process.

      .. note::

         With :class:`spawn` in bytes mode, the log files should be open for
         writing binary data. In unicode mode, they should
         be open for writing unicode text. See :ref:`unicode`.

Controlling the child process
`````````````````````````````

.. class:: spawn

   .. automethod:: kill
   .. automethod:: terminate
   .. automethod:: isalive
   .. automethod:: wait
   .. automethod:: close
   .. automethod:: getwinsize
   .. automethod:: setwinsize
   .. automethod:: getecho
   .. automethod:: setecho
   .. automethod:: waitnoecho

   .. attribute:: pid

      The process ID of the child process.

   .. attribute:: child_fd

      The file descriptor used to communicate with the child process.

.. _unicode:

Handling unicode
````````````````

By default, :class:`spawn` is a bytes interface: its read methods return bytes,
and its write/send and expect methods expect bytes. If you pass the *encoding*
parameter to the constructor, it will instead act as a unicode interface:
strings you send will be encoded using that encoding, and bytes received will
be decoded before returning them to you. In this mode, patterns for
:meth:`~spawn.expect` and :meth:`~spawn.expect_exact` should also be unicode.

.. versionchanged:: 4.0

   :class:`spawn` provides both the bytes and unicode interfaces. In Pexpect
   3.x, the unicode interface was provided by a separate ``spawnu`` class.

For backwards compatibility, some Unicode is allowed in bytes mode: the
send methods will encode arbitrary unicode as UTF-8 before sending it to the
child process, and its expect methods can accept ascii-only unicode strings.

.. note::

   Unicode handling with pexpect works the same way on Python 2 and 3, despite
   the difference in names. I.e.:

   - Bytes mode works with ``str`` on Python 2, and :class:`bytes` on Python 3,
   - Unicode mode works with ``unicode`` on Python 2, and :class:`str` on Python 3.

run function
------------

.. autofunction:: run

Exceptions
----------

.. autoclass:: EOF

.. autoclass:: TIMEOUT

.. autoclass:: ExceptionPexpect

Utility functions
-----------------

.. autofunction:: which

.. autofunction:: split_command_line
