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

    For more documentation on bugpoint, `Visit the LLVM manual <https://llvm.org/docs/Bugpoint.html>`_


Understanding which pass makes a particular change
--------------------------------------------------

    If you know that something like `opt -O3 -polly` makes a change, but you wish to
    isolate which pass makes a change, the steps are as follows:

    - ``$ bugpoint -O3 file.ll -opt-args -polly``  will allow bugpoint to track down the pass which causes the crash.

    To do this manually:

    - ``$ opt -O3 -polly -debug-pass=Arguments`` to get all passes that are run by default. ``-debug-pass=Arguments`` will list all passes that have run.
    - Bisect down to the pass that changes it.


Debugging regressions introduced at some unknown earlier point
--------------------------------------------------------------

In case of a regression in performance or correctness (e.g., an earlier version
of Polly behaved as expected and a later version does not), bisecting over the
version history is the standard approach to identify the commit that introduced
the regression.

LLVM has a single repository that contains all projects. It can be cloned at:
`<https://github.com/llvm/llvm-project>`_. How to bisect on a
git repository is explained here
`<https://www.metaltoad.com/blog/beginners-guide-git-bisect-process-elimination>`_.
The bisect process can also be automated as explained here:
`<https://www.metaltoad.com/blog/mechanizing-git-bisect-bug-hunting-lazy>`_.
An LLVM specific run script is available here:
`<https://gist.github.com/dcci/891cd98d80b1b95352a407d80914f7cf>`_.
