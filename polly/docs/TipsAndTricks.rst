==================================================
Tips and Tricks on using and contributing to Polly
==================================================

Commiting to polly trunk
------------------------
    - `General reference to git-svn workflow <https://stackoverflow.com/questions/190431/is-git-svn-dcommit-after-merging-in-git-dangerous>`_


Using bugpoint to track down errors in large files
--------------------------------------------------

    If you know the ``opt`` invocation and have a large ``.ll`` file that causes
    an error, ``bugpoint`` allows one to reduce the size of test cases.

    The general calling pattern is:

    - ``$ bugpoint <file.ll> <pass that causes the crash> -opt-args <opt option flags>``

    An example invocation is:

    - ``$ bugpoint crash.ll -polly-codegen -opt-args  -polly-canonicalize -polly-process-unprofitable``

    For more documentation on bugpoint, `Visit the LLVM manual <http://llvm.org/docs/Bugpoint.html>`_


Understanding which pass makes a particular change
--------------------------------------------------

    If you know that something like `opt -O3 -polly` makes a change, but you wish to
    isolate which pass makes a change, the steps are as follows:

    - ``$ bugpoint -O3 file.ll -opt-args -polly``  will allow bugpoint to track down the pass which causes the crash.

    To do this manually:

    - ``$ opt -O3 -polly -debug-pass=Arguments`` to get all passes that are run by default. ``-debug-pass=Arguments`` will list all passes that have run.
    - Bisect down to the pass that changes it.

