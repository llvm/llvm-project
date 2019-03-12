History
=======

Releases
--------

Version 4.6
```````````

* The :meth:`.pxssh.login` method now supports an ``ssh_config`` parameter,
  which can be used to specify a file path to an SSH config file
  (:ghpull:`490`).
* Improved compatability for the ``crlf`` parameter of :class:`~.PopenSpawn`
  (:ghpull:`493`)
* Fixed an issue in read timeout handling when using :class:`~.spawn` and
  :class:`~.fdspawn` with the ``use_poll`` parameter (:ghpull:`492`).

Version 4.5
```````````

* :class:`~.spawn` and :class:`~.fdspawn` now have a ``use_poll`` parameter.
  If this is True, they will use :func:`select.poll` instead of :func:`select.select`.
  ``poll()`` allows file descriptors above 1024, but it must be explicitly
  enabled due to compatibility concerns (:ghpull:`474`).
* The :meth:`.pxssh.login` method has several new and changed options:

  * The option ``password_regex`` allows changing
    the password prompt regex, for servers that include ``password:`` in a banner
    before reaching a prompt (:ghpull:`468`).
  * :meth:`~.pxssh.login` now allows for setting up SSH tunnels to be requested once
    logged in to the remote server. This option is ``ssh_tunnels`` (:ghpull:`473`).
    The structure should be like this::

          {
            'local': ['2424:localhost:22'],   # Local SSH tunnels
            'remote': ['2525:localhost:22'],  # Remote SSH tunnels
            'dynamic': [8888],                # Dynamic/SOCKS tunnels
          }

  * The option ``spawn_local_ssh=False`` allows subsequent logins from the
    remote session and treats the session as if it was local (:ghpull:`472`).
  * Setting ``sync_original_prompt=False`` will prevent changing the prompt to
    something unique, in case the remote server is sensitive to new lines at login
    (:ghpull:`468`).
  * If ``ssh_key=True`` is passed, the SSH client forces forwarding the authentication
    agent to the remote server instead of providing a key (:ghpull:`473`).

Version 4.4
```````````

* :class:`~.PopenSpawn` now has a ``preexec_fn`` parameter, like :class:`~.spawn`
  and :class:`subprocess.Popen`, for a function to be called in the child
  process before executing the new command. Like in ``Popen``, this works only
  in POSIX, and can cause issues if your application also uses threads
  (:ghpull:`460`).
* Significant performance improvements when processing large amounts of data
  (:ghpull:`464`).
* Ensure that ``spawn.closed`` gets set by :meth:`~.spawn.close`, and improve
  an example for passing ``SIGWINCH`` through to a child process (:ghpull:`466`).

Version 4.3.1
`````````````

* When launching bash for :mod:`pexpect.replwrap`, load the system ``bashrc``
  from a couple of different common locations (:ghpull:`457`), and then unset
  the ``PROMPT_COMMAND`` environment variable, which can interfere with the
  prompt we're expecting (:ghpull:`459`).

Version 4.3
```````````

* The ``async=`` parameter to integrate with asyncio has become ``async_=``
  (:ghpull:`431`), as *async* is becoming a Python keyword from Python 3.6.
  Pexpect will still recognise ``async`` as an alternative spelling.
* Similarly, the module ``pexpect.async`` became ``pexpect._async``
  (:ghpull:`450`). This module is not part of the public API.
* Fix problems with asyncio objects closing file descriptors during garbage
  collection (:ghissue:`347`, :ghpull:`376`).
* Set the ``.pid`` attribute of a :class:`~.PopenSpawn` object (:ghpull:`417`).
* Fix passing Windows paths to :class:`~.PopenSpawn` (:ghpull:`446`).
* :class:`~.PopenSpawn` on Windows can pass string commands through to ``Popen``
  without splitting them into a list (:ghpull:`447`).
* Stop ``shlex`` trying to read from stdin when :class:`~.PopenSpawn` is
  passed ``cmd=None`` (:ghissue:`433`, :ghpull:`434`).
* Ensure that an error closing a Pexpect spawn object raises a Pexpect error,
  rather than a Ptyprocess error (:ghissue:`383`, :ghpull:`386`).
* Cleaned up invalid backslash escape sequences in strings (:ghpull:`430`,
  :ghpull:`445`).
* The pattern for a password prompt in :mod:`pexpect.pxssh` changed from
  ``password`` to ``password:`` (:ghpull:`452`).
* Correct docstring for using unicode with spawn (:ghpull:`395`).
* Various other improvements to documentation.

Version 4.2.1
`````````````

* Fix to allow running ``env`` in replwrap-ed bash.
* Raise more informative exception from pxssh if it fails to connect.
* Change ``passmass`` example to not log passwords entered.

Version 4.2
```````````

* Change: When an ``env`` parameter is specified to the :class:`~.spawn` or
  :class:`~.run` family of calls containing a value for ``PATH``, its value is
  used to discover the target executable from a relative path, rather than the
  current process's environment ``PATH``.  This mirrors the behavior of
  :func:`subprocess.Popen` in the standard library (:ghissue:`348`).

* Regression: Re-introduce capability for :meth:`read_nonblocking` in class
  :class:`fdspawn` as previously supported in version 3.3 (:ghissue:`359`).

Version 4.0
```````````

* Integration with :mod:`asyncio`: passing ``async=True`` to :meth:`~.spawn.expect`,
  :meth:`~.spawn.expect_exact` or :meth:`~.spawn.expect_list` will make them return a
  coroutine. You can get the result using ``yield from``, or wrap it in an
  :class:`asyncio.Task`. This allows the event loop to do other things while
  waiting for output that matches a pattern.
* Experimental support for Windows (with some caveats)—see :ref:`windows`.
* Enhancement: allow method as callbacks of argument ``events`` for
  :func:`pexpect.run` (:ghissue:`176`).
* It is now possible to call :meth:`~.spawn.wait` multiple times, or after a process
  is already determined to be terminated without raising an exception
  (:ghpull:`211`).
* New :class:`pexpect.spawn` keyword argument, ``dimensions=(rows, columns)``
  allows setting terminal screen dimensions before launching a program
  (:ghissue:`122`).
* Fix regression that prevented executable, but unreadable files from
  being found when not specified by absolute path -- such as
  /usr/bin/sudo (:ghissue:`104`).
* Fixed regression when executing pexpect with some prior releases of
  the multiprocessing module where stdin has been closed (:ghissue:`86`).

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Deprecated ``pexpect.screen`` and ``pexpect.ANSI``. Please use other packages
  such as `pyte <https://pypi.python.org/pypi/pyte>`__ to emulate a terminal.
* Removed the independent top-level modules (``pxssh fdpexpect FSM screen ANSI``)
  which were installed alongside Pexpect. These were moved into the Pexpect
  package in 3.0, but the old names were left as aliases.
* Child processes created by Pexpect no longer ignore SIGHUP by default: the
  ``ignore_sighup`` parameter of :class:`pexpect.spawn` defaults to False. To
  get the old behaviour, pass ``ignore_sighup=True``.

Version 3.3
```````````

* Added a mechanism to wrap REPLs, or shells, in an object which can conveniently
  be used to send commands and wait for the output (:mod:`pexpect.replwrap`).
* Fixed issue where pexpect would attempt to execute a directory because
  it has the 'execute' bit set (:ghissue:`37`).
* Removed the ``pexpect.psh`` module. This was never documented, and we found
  no evidence that people use it. The new :mod:`pexpect.replwrap` module
  provides a more flexible alternative.
* Fixed ``TypeError: got <type 'str'> ('\r\n') as pattern`` in :meth:`spawnu.readline`
  method (:ghissue:`67`).
* Fixed issue where EOF was not correctly detected in :meth:`~.interact`, causing
  a repeating loop of output on Linux, and blocking before EOF on BSD and
  Solaris (:ghissue:`49`).
* Several Solaris (SmartOS) bugfixes, preventing :exc:`IOError` exceptions, especially
  when used with cron(1) (:ghissue:`44`).
* Added new keyword argument ``echo=True`` for :class:`spawn`.  On SVR4-like
  systems, the method :meth:`~.isatty` will always return *False*: the child pty
  does not appear as a terminal.  Therefore, :meth:`~.setecho`, :meth:`~.getwinsize`,
  :meth:`~.setwinsize`, and :meth:`~.waitnoecho` are not supported on those platforms.

After this, we intend to start working on a bigger refactoring of the code, to
be released as Pexpect 4. There may be more bugfix 3.x releases, however.

Version 3.2
```````````

* Fix exception handling from :func:`select.select` on Python 2 (:ghpull:`38`).
  This was accidentally broken in the previous release when it was fixed for
  Python 3.
* Removed a workaround for ``TIOCSWINSZ`` on very old systems, which was causing
  issues on some BSD systems (:ghpull:`40`).
* Fixed an issue with exception handling in :mod:`~pexpect.pxssh` (:ghpull:`43`)

The documentation for :mod:`~pexpect.pxssh` was improved.

Version 3.1
```````````

* Fix an issue that prevented importing pexpect on Python 3 when ``sys.stdout``
  was reassigned (:ghissue:`30`).
* Improve prompt synchronisation in :mod:`~pexpect.pxssh` (:ghpull:`28`).
* Fix pickling exception instances (:ghpull:`34`).
* Fix handling exceptions from :func:`select.select` on Python 3 (:ghpull:`33`).

The examples have also been cleaned up somewhat - this will continue in future
releases.

Version 3.0
```````````

The new major version number doesn't indicate any deliberate API incompatibility.
We have endeavoured to avoid breaking existing APIs. However, pexpect is under
new maintenance after a long dormancy, so some caution is warranted.

* A new :ref:`unicode API <unicode>` was introduced.
* Python 3 is now supported, using a single codebase.
* Pexpect now requires at least Python 2.6 or 3.2.
* The modules other than pexpect, such as :mod:`pexpect.fdpexpect` and
  :mod:`pexpect.pxssh`, were moved into the pexpect package. For now, wrapper
  modules are installed to the old locations for backwards compatibility (e.g.
  ``import pxssh`` will still work), but these will be removed at some point in
  the future.
* Ignoring ``SIGHUP`` is now optional - thanks to Kimmo Parviainen-Jalanko for
  the patch.

We also now have `docs on ReadTheDocs <https://pexpect.readthedocs.io/>`_,
and `continuous integration on Travis CI <https://travis-ci.org/pexpect/pexpect>`_.

Version 2.4
```````````

* Fix a bug regarding making the pty the controlling terminal when the process
  spawning it is not, actually, a terminal (such as from cron)

Version 2.3
```````````

* Fixed OSError exception when a pexpect object is cleaned up. Previously, you
  might have seen this exception::

      Exception exceptions.OSError: (10, 'No child processes')
      in <bound method spawn.__del__ of <pexpect.spawn instance at 0xd248c>> ignored

  You should not see that anymore. Thanks to Michael Surette.
* Added support for buffering reads. This greatly improves speed when trying to
  match long output from a child process. When you create an instance of the spawn
  object you can then set a buffer size. For now you MUST do the following to turn
  on buffering -- it may be on by default in future version::

      child = pexpect.spawn ('my_command')
      child.maxread=1000 # Sets buffer to 1000 characters.

* I made a subtle change to the way TIMEOUT and EOF exceptions behave.
  Previously you could either expect these states in which case pexpect
  will not raise an exception, or you could just let pexpect raise an
  exception when these states were encountered. If you expected the
  states then the ``before`` property was set to everything before the
  state was encountered, but if you let pexpect raise the exception then
  ``before`` was not set. Now, the ``before`` property will get set either
  way you choose to handle these states.
* The spawn object now provides iterators for a *file-like interface*.
  This makes Pexpect a more complete file-like object. You can now write
  code like this::

      child = pexpect.spawn ('ls -l')
      for line in child:
          print line

* write and writelines() no longer return a value. Use send() if you need that
  functionality. I did this to make the Spawn object more closely match a
  file-like object.
* Added the attribute ``exitstatus``. This will give the exit code returned
  by the child process. This will be set to ``None`` while the child is still
  alive. When ``isalive()`` returns 0 then ``exitstatus`` will be set.
* Made a few more tweaks to ``isalive()`` so that it will operate more
  consistently on different platforms. Solaris is the most difficult to support.
* You can now put ``TIMEOUT`` in a list of expected patterns. This is just like
  putting ``EOF`` in the pattern list. Expecting for a ``TIMEOUT`` may not be
  used as often as ``EOF``, but this makes Pexpect more consistent.
* Thanks to a suggestion and sample code from Chad J. Schroeder I added the ability
  for Pexpect to operate on a file descriptor that is already open. This means that
  Pexpect can be used to control streams such as those from serial port devices. Now,
  you just pass the integer file descriptor as the "command" when constructing a
  spawn open. For example on a Linux box with a modem on ttyS1::

      fd = os.open("/dev/ttyS1", os.O_RDWR|os.O_NONBLOCK|os.O_NOCTTY)
      m = pexpect.spawn(fd) # Note integer fd is used instead of usual string.
      m.send("+++") # Escape sequence
      m.send("ATZ0\r") # Reset modem to profile 0
      rval = m.expect(["OK", "ERROR"])

* ``read()`` was renamed to ``read_nonblocking()``. Added new ``read()`` method
  that matches file-like object interface. In general, you should not notice
  the difference except that ``read()`` no longer allows you to directly set the
  timeout value. I hope this will not effect any existing code. Switching to
  ``read_nonblocking()`` should fix existing code.
* Changed the name of ``set_echo()`` to ``setecho()``.
* Changed the name of ``send_eof()`` to ``sendeof()``.
* Modified ``kill()`` so that it checks to make sure the pid ``isalive()``.
* modified ``spawn()`` (really called from ``__spawn()``) so that it does not
  raise an exception if ``setwinsize()`` fails. Some platforms such as Cygwin
  do not like setwinsize. This was a constant problem and since it is not a
  critical feature I decided to just silence the error.  Normally I don't like
  to do that, but in this case I'm making an exception.
* Added a method ``close()`` that does what you think. It closes the file
  descriptor of the child application. It makes no attempt to actually kill the
  child or wait for its status.
* Add variables ``__version__`` and ``__revision__`` (from cvs) to the pexpect
  modules.  This is mainly helpful to me so that I can make sure that I'm testing
  with the right version instead of one already installed.
* ``log_open()`` and ``log_close(`` have been removed. Now use ``setlog()``.
  The ``setlog()`` method takes a file object. This is far more flexible than
  the previous log method. Each time data is written to the file object it will
  be flushed. To turn logging off simply call ``setlog()`` with None.
* renamed the ``isAlive()`` method to ``isalive()`` to match the more typical
  naming style in Python. Also the technique used to detect child process
  status has been drastically modified. Previously I did some funky stuff
  with signals which caused indigestion in other Python modules on some
  platforms. It was a big headache. It still is, but I think it works
  better now.
* attribute ``matched`` renamed to ``after``
* new attribute ``match``
* The ``expect_eof()`` method is gone. You can now simply use the
  ``expect()`` method to look for EOF.
* **Pexpect works on OS X**, but the nature of the quirks cause many of the
  tests to fail. See bugs. (Incomplete Child Output). The problem is more
  than minor, but Pexpect is still more than useful for most tasks.
* **Solaris**: For some reason, the *second* time a pty file descriptor is created and
  deleted it never gets returned for use. It does not effect the first time
  or the third time or any time after that. It's only the second time. This
  is weird... This could be a file descriptor leak, or it could be some
  peculiarity of how Solaris recycles them. I thought it was a UNIX requirement
  for the OS to give you the lowest available filedescriptor number. In any case,
  this should not be a problem unless you create hundreds of pexpect instances...
  It may also be a pty module bug.


Moves and forks
---------------

* Pexpect development used to be hosted on Sourceforge.
* In 2011, Thomas Kluyver forked pexpect as 'pexpect-u', to support
  Python 3. He later decided he had taken the wrong approach with this.
* In 2012, Noah Spurrier, the original author of Pexpect, moved the
  project to Github, but was still too busy to develop it much.
* In 2013, Thomas Kluyver and Jeff Quast forked Pexpect again, intending
  to call the new fork Pexpected. Noah Spurrier agreed to let them use
  the name Pexpect, so Pexpect versions 3 and above are based on this
  fork, which now lives `here on Github <https://github.com/pexpect/pexpect>`_.
