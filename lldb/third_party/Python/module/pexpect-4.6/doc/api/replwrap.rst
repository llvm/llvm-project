replwrap - Control read-eval-print-loops
========================================

.. automodule:: pexpect.replwrap

.. versionadded:: 3.3

.. autoclass:: REPLWrapper

   .. automethod:: run_command

.. data:: PEXPECT_PROMPT

   A string that can be used as a prompt, and is unlikely to be found in output.

Using the objects above, it is easy to wrap a REPL. For instance, to use a
Python shell::

    py = REPLWrapper("python", ">>> ", "import sys; sys.ps1={!r}; sys.ps2={!r}")
    py.run_command("4+7")

Convenience functions are provided for Python and bash shells:

.. autofunction:: python

.. autofunction:: bash
