FAQ
===

**Q: Where can I get help with pexpect?  Is there a mailing list?**

A: You can use the `pexpect tag on Stackoverflow <http://stackoverflow.com/questions/tagged/pexpect>`__
to ask questions specifically related to Pexpect. For more general Python
support, there's the python-list_ mailing list, and the `#python`_
IRC channel.  Please refrain from using github for general
python or systems scripting support.

.. _python-list: https://mail.python.org/mailman/listinfo/python-list
.. _#python: https://www.python.org/community/irc/

**Q: Why don't shell pipe and redirect (| and >) work when I spawn a command?**

A: Remember that Pexpect does NOT interpret shell meta characters such as
redirect, pipe, or wild cards (``>``, ``|``, or ``*``). That's done by a shell not
the command you are spawning. This is a common mistake. If you want to run a
command and pipe it through another command then you must also start a shell.
For example::

    child = pexpect.spawn('/bin/bash -c "ls -l | grep LOG > log_list.txt"')
    child.expect(pexpect.EOF)

The second form of spawn (where you pass a list of arguments) is useful in
situations where you wish to spawn a command and pass it its own argument list.
This can make syntax more clear. For example, the following is equivalent to the
previous example::

    shell_cmd = 'ls -l | grep LOG > log_list.txt'
    child = pexpect.spawn('/bin/bash', ['-c', shell_cmd])
    child.expect(pexpect.EOF)

**Q: The `before` and `after` properties sound weird.**

A: This is how the -B and -A options in grep works, so that made it
easier for me to remember. Whatever makes my life easier is what's best.
Originally I was going to model Pexpect after Expect, but then I found
that I didn't actually like the way Expect did some things. It was more
confusing. The `after` property can be a little confusing at first,
because it will actually include the matched string. The `after` means
after the point of match, not after the matched string.

**Q: Why not just use Expect?**

A: I love it. It's great. I has bailed me out of some real jams, but I
wanted something that would do 90% of what I need from Expect; be 10% of
the size; and allow me to write my code in Python instead of TCL.
Pexpect is not nearly as big as Expect, but Pexpect does everything I
have ever used Expect for.

.. _whynotpipe:

**Q: Why not just use a pipe (popen())?**

A: A pipe works fine for getting the output to non-interactive programs.
If you just want to get the output from ls, uname, or ping then this
works. Pipes do not work very well for interactive programs and pipes
will almost certainly fail for most applications that ask for passwords
such as telnet, ftp, or ssh.

There are two reasons for this.

* First an application may bypass stdout and print directly to its
  controlling TTY. Something like SSH will do this when it asks you for
  a password. This is why you cannot redirect the password prompt because
  it does not go through stdout or stderr.

* The second reason is because most applications are built using the C
  Standard IO Library (anything that uses ``#include <stdio.h>``). One
  of the features of the stdio library is that it buffers all input and
  output. Normally output is line buffered when a program is printing to
  a TTY (your terminal screen). Everytime the program prints a line-feed
  the currently buffered data will get printed to your screen. The
  problem comes when you connect a pipe. The stdio library is smart and
  can tell that it is printing to a pipe instead of a TTY. In that case
  it switches from line buffer mode to block buffered. In this mode the
  currently buffered data is flushed when the buffer is full. This
  causes most interactive programs to deadlock. Block buffering is more
  efficient when writing to disks and pipes. Take the situation where a
  program prints a message ``"Enter your user name:\n"`` and then waits
  for you type type something. In block buffered mode, the stdio library
  will not put the message into the pipe even though a linefeed is
  printed. The result is that you never receive the message, yet the
  child application will sit and wait for you to type a response. Don't
  confuse the stdio lib's buffer with the pipe's buffer. The pipe buffer
  is another area that can cause problems. You could flush the input
  side of a pipe, whereas you have no control over the stdio library buffer.

More information: the Standard IO library has three states for a
``FILE *``. These are: _IOFBF for block buffered; _IOLBF for line buffered;
and _IONBF for unbuffered. The STDIO lib will use block buffering when
talking to a block file descriptor such as a pipe. This is usually not
helpful for interactive programs. Short of recompiling your program to
include fflush() everywhere or recompiling a custom stdio library there
is not much a controlling application can do about this if talking over
a pipe.

The program may have put data in its output that remains unflushed
because the output buffer is not full; then the program will go and
deadlock while waiting for input -- because you never send it any
because you are still waiting for its output (still stuck in the STDIO's
output buffer).

The answer is to use a pseudo-tty. A TTY device will force line
buffering (as opposed to block buffering). Line buffering means that you
will get each line when the child program sends a line feed. This
corresponds to the way most interactive programs operate -- send a line
of output then wait for a line of input.

I put "answer" in quotes because it's ugly solution and because there is
no POSIX standard for pseudo-TTY devices (even though they have a TTY
standard...). What would make more sense to me would be to have some way
to set a mode on a file descriptor so that it will tell the STDIO to be
line-buffered. I have investigated, and I don't think there is a way to
set the buffered state of a child process. The STDIO Library does not
maintain any external state in the kernel or whatnot, so I don't think
there is any way for you to alter it. I'm not quite sure how this
line-buffered/block-buffered state change happens internally in the
STDIO library. I think the STDIO lib looks at the file descriptor and
decides to change behavior based on whether it's a TTY or a block file
(see isatty()).

I hope that this qualifies as helpful. Don't use a pipe to control
another application.

**Q: Can I do screen scraping with this thing?**

A: That depends. If your application just does line-oriented output then
this is easy. If a program emits many terminal sequences, from video
attributes to screen addressing, such as programs using curses, then
it may become very difficult to ascertain what text is displayed on a screen.

We suggest using the `pyte <https://github.com/selectel/pyte>`_ library to
screen-scrape.  The module :mod:`pexpect.ANSI` released with previous versions
of pexpect is now marked deprecated and may be removed in the future.

**Q: I get strange behavior with pexect and gevent**

A: Pexpect uses fork(2), exec(2), select(2), waitpid(2), and implements its
own selector in expect family of calls. pexpect has been known to misbehave
when paired with gevent.  A solution might be to isolate your pexpect
dependent code from any frameworks that manipulate event selection behavior
by running it in an another process entirely.
