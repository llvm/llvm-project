API Overview
============

Pexpect can be used for automating interactive applications such as ssh, ftp,
mencoder, passwd, etc. The Pexpect interface was designed to be easy to use.

Here is an example of Pexpect in action::

    # This connects to the openbsd ftp site and
    # downloads the recursive directory listing.
    import pexpect
    child = pexpect.spawn('ftp ftp.openbsd.org')
    child.expect('Name .*: ')
    child.sendline('anonymous')
    child.expect('Password:')
    child.sendline('noah@example.com')
    child.expect('ftp> ')
    child.sendline('lcd /tmp')
    child.expect('ftp> ')
    child.sendline('cd pub/OpenBSD')
    child.expect('ftp> ')
    child.sendline('get README')
    child.expect('ftp> ')
    child.sendline('bye')

Obviously you could write an ftp client using Python's own :mod:`ftplib` module,
but this is just a demonstration. You can use this technique with any application.
This is especially handy if you are writing automated test tools.

There are two important methods in Pexpect -- :meth:`~pexpect.spawn.expect` and
:meth:`~pexpect.spawn.send` (or :meth:`~pexpect.spawn.sendline` which is
like :meth:`~pexpect.spawn.send` with a linefeed). The :meth:`~pexpect.spawn.expect`
method waits for the child application to return a given string. The string you
specify is a regular expression, so you can match complicated patterns. The
:meth:`~pexpect.spawn.send` method writes a string to the child application.
From the child's point of view it looks just like someone typed the text from a
terminal. After each call to :meth:`~pexpect.spawn.expect` the ``before`` and ``after``
properties will be set to the text printed by child application. The ``before``
property will contain all text up to the expected string pattern. The ``after``
string will contain the text that was matched by the expected pattern.
The match property is set to the `re match object <http://docs.python.org/3/library/re#match-objects>`_.

An example of Pexpect in action may make things more clear. This example uses
ftp to login to the OpenBSD site; list files in a directory; and then pass
interactive control of the ftp session to the human user::

    import pexpect
    child = pexpect.spawn ('ftp ftp.openbsd.org')
    child.expect ('Name .*: ')
    child.sendline ('anonymous')
    child.expect ('Password:')
    child.sendline ('noah@example.com')
    child.expect ('ftp> ')
    child.sendline ('ls /pub/OpenBSD/')
    child.expect ('ftp> ')
    print child.before   # Print the result of the ls command.
    child.interact()     # Give control of the child to the user.

Special EOF and TIMEOUT patterns
--------------------------------

There are two special patterns to match the End Of File (:class:`~pexpect.EOF`)
or a Timeout condition (:class:`~pexpect.TIMEOUT`). You can pass these
patterns to :meth:`~pexpect.spawn.expect`. These patterns are not regular
expressions. Use them like predefined constants.

If the child has died and you have read all the child's output then ordinarily
:meth:`~pexpect.spawn.expect` will raise an :class:`~pexpect.EOF` exception.
You can read everything up to the EOF without generating an exception by using
the EOF pattern expect. In this case everything the child has output will be
available in the ``before`` property.

The pattern given to :meth:`~pexpect.spawn.expect` may be a regular expression
or it may also be a list of regular expressions. This allows you to match
multiple optional responses. The :meth:`~pexpect.spawn.expect` method returns
the index of the pattern that was matched. For example, say you wanted to login
to a server. After entering a password you could get various responses from the
server -- your password could be rejected; or you could be allowed in and asked
for your terminal type; or you could be let right in and given a command prompt.
The following code fragment gives an example of this::

    child.expect('password:')
    child.sendline(my_secret_password)
    # We expect any of these three patterns...
    i = child.expect (['Permission denied', 'Terminal type', '[#\$] '])
    if i==0:
        print('Permission denied on host. Can\'t login')
        child.kill(0)
    elif i==1:
        print('Login OK... need to send terminal type.')
        child.sendline('vt100')
        child.expect('[#\$] ')
    elif i==2:
        print('Login OK.')
        print('Shell command prompt', child.after)

If nothing matches an expected pattern then :meth:`~pexpect.spawn.expect` will
eventually raise a :class:`~pexpect.TIMEOUT` exception. The default time is 30
seconds, but you can change this by passing a timeout argument to
:meth:`~pexpect.spawn.expect`::

    # Wait no more than 2 minutes (120 seconds) for password prompt.
    child.expect('password:', timeout=120)

Find the end of line -- CR/LF conventions
-----------------------------------------

Pexpect matches regular expressions a little differently than what you might be
used to.

The :regexp:`$` pattern for end of line match is useless. The :regexp:`$`
matches the end of string, but Pexpect reads from the child one character at a
time, so each character looks like the end of a line. Pexpect can't do a
look-ahead into the child's output stream. In general you would have this
situation when using regular expressions with any stream.

.. note::

  Pexpect does have an internal buffer, so reads are faster than one character
  at a time, but from the user's perspective the regex patterns test happens
  one character at a time.

The best way to match the end of a line is to look for the newline: ``"\r\n"``
(CR/LF). Yes, that does appear to be DOS-style. It may surprise some UNIX people
to learn that terminal TTY device drivers (dumb, vt100, ANSI, xterm, etc.) all
use the CR/LF combination to signify the end of line. Pexpect uses a Pseudo-TTY
device to talk to the child application, so when the child app prints ``"\n"``
you actually see ``"\r\n"``.

UNIX uses just linefeeds to end lines of text, but not when it comes to TTY
devices! TTY devices are more like the Windows world. Each line of text ends
with a CR/LF combination. When you intercept data from a UNIX command from a
TTY device you will find that the TTY device outputs a CR/LF combination. A
UNIX command may only write a linefeed (``\n``), but the TTY device driver
converts it to CR/LF. This means that your terminal will see lines end with
CR/LF (hex ``0D 0A``). Since Pexpect emulates a terminal, to match ends of
lines you have to expect the CR/LF combination::

    child.expect('\r\n')

If you just need to skip past a new line then ``expect('\n')`` by itself will
work, but if you are expecting a specific pattern before the end of line then
you need to explicitly look for the ``\r``. For example the following expects a
word at the end of a line::

    child.expect('\w+\r\n')

But the following would both fail::

    child.expect('\w+\n')

And as explained before, trying to use :regexp:`$` to match the end of line
would not work either::

    child.expect ('\w+$')

So if you need to explicitly look for the END OF LINE, you want to look for the
CR/LF combination -- not just the LF and not the $ pattern.

This problem is not limited to Pexpect. This problem happens any time you try
to perform a regular expression match on a stream. Regular expressions need to
look ahead. With a stream it is hard to look ahead because the process
generating the stream may not be finished. There is no way to know if the
process has paused momentarily or is finished and waiting for you. Pexpect must
implicitly always do a NON greedy match (minimal) at the end of a input.

Pexpect compiles all regular expressions with the :data:`re.DOTALL` flag.
With the :data:`~re.DOTALL` flag, a ``"."`` will match a newline.

Beware of + and * at the end of patterns
----------------------------------------

Remember that any time you try to match a pattern that needs look-ahead that
you will always get a minimal match (non greedy). For example, the following
will always return just one character::

    child.expect ('.+')

This example will match successfully, but will always return no characters::

    child.expect ('.*')

Generally any star * expression will match as little as possible.

One thing you can do is to try to force a non-ambiguous character at the end of
your :regexp:`\\d+` pattern. Expect that character to delimit the string. For
example, you might try making the end of your pattern be :regexp:`\\D+` instead
of :regexp:`\\D*`. Number digits alone would not satisfy the :regexp:`(\\d+)\\D+`
pattern. You would need some numbers and at least one non-number at the end.


Debugging
---------

If you get the string value of a :class:`pexpect.spawn` object you will get lots
of useful debugging information. For debugging it's very useful to use the
following pattern::

    try:
        i = child.expect ([pattern1, pattern2, pattern3, etc])
    except:
        print("Exception was thrown")
        print("debug information:")
        print(str(child))

It is also useful to log the child's input and out to a file or the screen. The
following will turn on logging and send output to stdout (the screen)::

    child = pexpect.spawn(foo)
    child.logfile = sys.stdout

Exceptions
----------

:class:`~pexpect.EOF`

Note that two flavors of EOF Exception may be thrown. They are virtually
identical except for the message string. For practical purposes you should have
no need to distinguish between them, but they do give a little extra information
about what type of platform you are running. The two messages are:

- "End Of File (EOF) in read(). Exception style platform."
- "End Of File (EOF) in read(). Empty string style platform."

Some UNIX platforms will throw an exception when you try to read from a file
descriptor in the EOF state. Other UNIX platforms instead quietly return an
empty string to indicate that the EOF state has been reached.

If you wish to read up to the end of the child's output without generating an
:class:`~pexpect.EOF` exception then use the ``expect(pexpect.EOF)`` method.

:class:`~pexpect.TIMEOUT`

The :meth:`~pexpect.spawn.expect` and :meth:`~pexpect.spawn.read` methods will
also timeout if the child does not generate any output for a given amount of
time. If this happens they will raise a :class:`~pexpect.TIMEOUT` exception.
You can have these methods ignore timeout and block indefinitely by passing
``None`` for the timeout parameter::

    child.expect(pexpect.EOF, timeout=None)

.. _windows:

Pexpect on Windows
------------------

.. versionadded:: 4.0
   Windows support

Pexpect can be used on Windows to wait for a pattern to be produced by a child
process, using :class:`pexpect.popen_spawn.PopenSpawn`, or a file descriptor,
using :class:`pexpect.fdpexpect.fdspawn`.

:class:`pexpect.spawn` and :func:`pexpect.run` are *not* available on Windows,
as they rely on Unix pseudoterminals (ptys). Cross platform code must not use
these.

``PopenSpawn`` is not a direct replacement for ``spawn``. Many programs only
offer interactive behaviour if they detect that they are running in a terminal.
When run by ``PopenSpawn``, they may behave differently.

.. seealso::

   `winpexpect <https://pypi.python.org/pypi/winpexpect>`__ and `wexpect <https://gist.github.com/anthonyeden/8488763>`__
     Two unmaintained pexpect-like modules for Windows, which work with a
     hidden console.
