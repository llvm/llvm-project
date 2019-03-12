Common problems
===============

Threads
-------

On Linux (RH 8) you cannot spawn a child from a different thread and pass the
handle back to a worker thread. The child is successfully spawned but you can't
interact with it. The only way to make it work is to spawn and interact with the
child all in the same thread. [Adam Kerrison]

Timing issue with send() and sendline()
---------------------------------------

This problem has been addressed and should not affect most users.

It is sometimes possible to read an echo of the string sent with
:meth:`~pexpect.spawn.send` and :meth:`~pexpect.spawn.sendline`. If you call
:meth:`~pexpect.spawn.send` and then immediately call :meth:`~pexpect.spawn.readline`,
you may get part of your output echoed back. You may read back what you just
wrote even if the child application does not explicitly echo it. Timing is
critical. This could be a security issue when talking to an application that
asks for a password; otherwise, this does not seem like a big deal. But why do
TTYs do this?

People usually report this when they are trying to control SSH or some other
login. For example, if your code looks something like this::

    child.expect ('[pP]assword:')
    child.sendline (my_password)


1. SSH prints "password:" prompt to the user.
2. SSH turns off echo on the TTY device.
3. SSH waits for user to enter a password.

When scripting with Pexpect what can happen is that Pexpect will respond to the
"password:" prompt before SSH has had time to turn off TTY echo. In other words,
Pexpect sends the password between steps 1. and 2., so the password gets echoed
back to the TTY. I would call this an SSH bug.

Pexpect now automatically adds a short delay before sending data to a child
process. This more closely mimics what happens in the usual human-to-app
interaction. The delay can be tuned with the ``delaybeforesend`` attribute of the
spawn class. In general, this fixes the problem for everyone and so this should
not be an issue for most users. For some applications you might with to turn it
off::

    child = pexpect.spawn ("ssh user@example.com")
    child.delaybeforesend = None

Truncated output just before child exits
----------------------------------------

So far I have seen this only on older versions of Apple's MacOS X. If the child
application quits it may not flush its output buffer. This means that your
Pexpect application will receive an EOF even though it should have received a
little more data before the child died. This is not generally a problem when
talking to interactive child applications. One example where it is a problem is
when trying to read output from a program like *ls*. You may receive most of the
directory listing, but the last few lines will get lost before you receive an EOF.
The reason for this is that *ls* runs; completes its task; and then exits. The
buffer is not flushed before exit so the last few lines are lost. The following
example demonstrates the problem::

    child = pexpect.spawn('ls -l')
    child.expect(pexpect.EOF)
    print child.before       

Controlling SSH on Solaris
--------------------------

Pexpect does not yet work perfectly on Solaris. One common problem is that SSH
sometimes will not allow TTY password authentication. For example, you may
expect SSH to ask you for a password using code like this::

    child = pexpect.spawn('ssh user@example.com')
    child.expect('password')
    child.sendline('mypassword')

You may see the following error come back from a spawned child SSH::

    Permission denied (publickey,keyboard-interactive). 

This means that SSH thinks it can't access the TTY to ask you for your password.
The only solution I have found is to use public key authentication with SSH.
This bypasses the need for a password. I'm not happy with this solution. The
problem is due to poor support for Solaris Pseudo TTYs in the Python Standard
Library.

child does not receive full input, emits BEL
--------------------------------------------

You may notice when running for example cat(1) or base64(1), when sending a
very long input line, that it is not fully received, and the BEL ('\a') may
be found in output.

By default the child terminal matches the parent, which is often in "canonical
mode processing". You may wish to disable this mode. The exact limit of a line
varies by operating system, and details of disabling canonical mode may be
found in the docstring of :meth:`~pexpect.spawn.send`.
