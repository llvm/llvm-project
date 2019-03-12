pxssh - control an SSH session
==============================

.. automodule:: pexpect.pxssh

.. autoclass:: ExceptionPxssh

pxssh class
-----------

.. autoclass:: pxssh

   .. automethod:: __init__

   .. attribute:: PROMPT

      The regex pattern to search for to find the prompt. If you call :meth:`login`
      with ``auto_prompt_reset=False``, you must set this attribute manually.

   .. attribute:: force_password

      If this is set to True, public key authentication is disabled, forcing the
      server to ask for a password. Note that the sysadmin can disable password
      logins, in which case this won't work.

   .. attribute:: options

      The dictionary of user specified SSH options, eg, ``options = dict(StrictHostKeyChecking="no", UserKnownHostsFile="/dev/null")``

   .. automethod:: login
   .. automethod:: logout
   .. automethod:: prompt
   .. automethod:: sync_original_prompt
   .. automethod:: set_unique_prompt
