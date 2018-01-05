==================================
Contributing to LLVM
==================================


Thank you for your interest in contributing to LLVM! There are multiple ways to
contribute, and we appreciate all contributions. In case you
have questions, you can either use the `Developer's List (llvm-dev)`_
or the #llvm channel on `irc.oftc.net`_.

If you want to contribute code, please familiarize yourself with the :doc:`DeveloperPolicy`.

.. contents::
  :local:


Ways to Contribute
==================

Bug Reports
-----------
If you are working with LLVM and run into a bug, we definitely want to know
about it. Please let us know and follow the instructions in
:doc:`HowToSubmitABug`  to create a bug report.

Bug Fixes
---------
If you are interested in contributing code to LLVM, bugs labeled with the
`beginner keyword`_ in the `bug tracker`_ are a good way to get familiar with
the code base. If you are interested in fixing a bug, please create an account
for the bug tracker and assign it to yourself, to let people know you are working on
it.

Then try to reproduce and fix the bug with upstream LLVM. Start by building
LLVM from source as described in :doc:`GettingStarted` and
and use the built binaries to reproduce the failure described in the bug. Use
a debug build (`-DCMAKE_BUILD_TYPE=Debug`) or a build with assertions
(`-DLLVM_ENABLE_ASSERTIONS=On`, enabled for Debug builds).

Bigger Pieces of Work
---------------------
In case you are interested in taking on a bigger piece of work, a list of
interesting projects is maintained at the `LLVM's Open Projects page`_. In case
you are interested in working on any of these projects, please send a mail to
the `LLVM Developer's mailing list`_, so that we know the project is being
worked on.


How to Submit a Patch
=====================
Once you have a patch ready, it is time to submit it. The patch should:

* include a small unit test
* conform to the :doc:`CodingStandards`. You can use the `clang-format-diff.py`_ or `git-clang-format`_ tools to automatically format your patch properly.
* not contain any unrelated changes
* be an isolated change. Independent changes should be submitted as separate patches as this makes reviewing easier.

To get a patch accepted, it has to be reviewed by the LLVM community. This can
be done using `LLVM's Phabricator`_ or the llvm-commits mailing list.
Please  follow :ref:`Phabricator#requesting-a-review-via-the-web-interface <phabricator-request-review-web>`
to request a review using Phabricator.

To make sure the right people see your patch, please select suitable reviewers
and add them to your patch when requesting a review. Suitable reviewers are the
code owner (see CODE_OWNERS.txt) and other people doing work in the area your
patch touches. If you are using Phabricator, add them to the `Reviewers` field
when creating a review and if you are using `llvm-commits`, add them to the CC of
your email.

A reviewer may request changes or ask questions during the review. If you are
uncertain on how to provide test cases, documentation, etc., feel free to ask
for guidance during the review. Please address the feedback and re-post an
updated version of your patch. This cycle continues until all requests and comments
have been addressed and a reviewer accepts the patch with a `Looks good to me` or `LGTM`.
Once that is done the change can be committed. If you do not have commit
access, please let people know during the review and someone should commit it
on your behalf.

If you have received no comments on your patch for a week, you can request a
review by 'ping'ing a patch by responding to the email thread containing the
patch, or the Phabricator review with "Ping." The common courtesy 'ping' rate
is once a week. Please remember that you are asking for valuable time from other
professional developers.


Helpful Information About LLVM
==============================
:doc:`LLVM's documentation <index>` provides a wealth of information about LLVM's internals as
well as various user guides. The pages listed below should provide a good overview
of LLVM's high-level design, as well as its internals:

`Intro to LLVM`__
  Book chapter providing a compiler hacker's introduction to LLVM.

  .. __: http://www.aosabook.org/en/llvm.html

:doc:`GettingStarted`
   Discusses how to get up and running quickly with the LLVM infrastructure.
   Everything from unpacking and compilation of the distribution to execution
   of some tools.

:doc:`LangRef`
  Defines the LLVM intermediate representation.

:doc:`ProgrammersManual`
  Introduction to the general layout of the LLVM sourcebase, important classes
  and APIs, and some tips & tricks.

:ref:`index-subsystem-docs`
  A collection of pages documenting various subsystems of LLVM.



.. _Developer's List (llvm-dev): http://lists.llvm.org/mailman/listinfo/llvm-dev
.. _irc.oftc.net: irc://irc.oftc.net/llvm
.. _beginner keyword: https://bugs.llvm.org/buglist.cgi?bug_status=NEW&bug_status=REOPENED&keywords=beginner%2C%20&keywords_type=allwords&list_id=130748&query_format=advanced&resolution=---
.. _bug tracker: https://bugs.llvm.org
.. _clang-format-diff.py: https://reviews.llvm.org/source/clang/browse/cfe/trunk/tools/clang-format/clang-format-diff.py
.. _git-clang-format: https://reviews.llvm.org/source/clang/browse/cfe/trunk/tools/clang-format/git-clang-format
.. _LLVM's Phabricator: https://reviews.llvm.org/
.. _LLVM's Open Projects page: https://llvm.org/OpenProjects.html#what
.. _LLVM Developer's mailing list: http://lists.llvm.org/mailman/listinfo/llvm-dev
