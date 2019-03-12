API documentation
=================

.. toctree::
   :maxdepth: 2

   pexpect
   fdpexpect
   popen_spawn
   replwrap
   pxssh

The modules ``pexpect.screen`` and ``pexpect.ANSI`` have been deprecated in
Pexpect version 4. They were separate from the main use cases for Pexpect, and
there are better maintained Python terminal emulator packages, such as
`pyte <https://pypi.python.org/pypi/pyte>`__.
These modules are still present for now, but we don't advise using them in new
code.
