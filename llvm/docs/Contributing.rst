==================================
Contributing to LLVM
==================================


Thank you for your interest in contributing to LLVM! There are multiple ways to
contribute, and we appreciate all contributions. If you have questions,
you can either use the `Forum`_ or, for a more interactive chat, go to our
`Discord server`_.

If you want to contribute code, please familiarize yourself with the :doc:`DeveloperPolicy`.

.. contents::
  :local:


Ways to Contribute
==================

Bug Reports
-----------
If you are working with LLVM and run into a bug, we definitely want to know
about it. Please follow the instructions in
:doc:`HowToSubmitABug`  to create a bug report.

Bug Fixes
---------
If you are interested in contributing code to LLVM, bugs labeled with the
`good first issue`_ keyword in the `bug tracker`_ are a good way to get familiar with
the code base. If you are interested in fixing a bug, please comment on it to
let people know you are working on it.

Then try to reproduce and fix the bug with upstream LLVM. Start by building
LLVM from source as described in :doc:`GettingStarted` and
use the built binaries to reproduce the failure described in the bug. Use
a debug build (`-DCMAKE_BUILD_TYPE=Debug`) or a build with assertions
(`-DLLVM_ENABLE_ASSERTIONS=On`, enabled for Debug builds).

Reporting a Security Issue
--------------------------

There is a separate process to submit security-related bugs, see :ref:`report-security-issue`.

Bigger Pieces of Work
---------------------
If you are interested in taking on a bigger piece of work, a list of
interesting projects is maintained at the `LLVM's Open Projects page`_. If
you are interested in working on any of these projects, please post on the
`Forum`_, so that we know the project is being worked on.

.. _submit_patch:

How to Submit a Patch
=====================
Once you have a patch ready, it is time to submit it. The patch should:

* include a small unit test
* conform to the :doc:`CodingStandards`. You can use the `clang-format-diff.py`_ or `git-clang-format`_ tools to automatically format your patch properly.
* not contain any unrelated changes
* be an isolated change. Independent changes should be submitted as separate patches as this makes reviewing easier.
* have a single commit, up-to-date with the upstream ``origin/main`` branch, and don't have merges.

.. _format patches:

Before sending a patch for review, please also ensure it is
formatted properly. We use ``clang-format`` for this, which has git integration
through the ``git-clang-format`` script. On some systems, it may already be
installed (or be installable via your package manager). If so, you can simply
run it -- the following command will format only the code changed in the most
recent commit:

.. code-block:: console

  % git clang-format HEAD~1

.. note::
  For some patches, formatting them may add changes that obscure the intent of
  the patch. For example, adding to an enum that was not previously formatted
  may result in the entire enum being reformatted. This happens because not all
  of the LLVM Project conforms to LLVM's clang-format style at this time.

  If you think that this might be the case for your changes, or are unsure, we
  recommend that you add the formatting changes as a **separate commit** within
  the Pull Request.

  Reviewers may request that this formatting commit be made into a separate Pull
  Request that will be merged before your actual changes.

  This means that if the formatting changes are the first commit, you will have
  an easier time doing this. If they are not, that is ok too, but you will have
  to do a bit more work to separate it out.

Note that ``git clang-format`` modifies the files, but does not commit them --
you will likely want to run one of the following to add the changes to a commit:

.. code-block:: console

  # To create a new commit.
  % git commit -a
  # To add to the most recent commit.
  % git commit --amend -a

.. note::
  If you don't already have ``clang-format`` or ``git clang-format`` installed
  on your system, the ``clang-format`` binary will be built alongside clang, and
  the git integration can be run from
  ``clang/tools/clang-format/git-clang-format``.

The LLVM project has migrated to GitHub Pull Requests as its review process.
For more information about the workflow of using GitHub Pull Requests see our
:ref:`GitHub <github-reviews>` documentation. We still have a read-only
`LLVM's Phabricator <https://reviews.llvm.org>`_ instance.

To make sure the right people see your patch, please select suitable reviewers
and add them to your patch when requesting a review.

Suitable reviewers are the maintainers of the project you are modifying, and
anyone else working in the area your patch touches. To find maintainers, look for
the ``Maintainers.md`` or ``Maintainers.rst`` file in the root of the project's
sub-directory. For example, LLVM's is ``llvm/Maintainers.md`` and Clang's is
``clang/Maintainers.rst``.

If you are a new contributor, you will not be able to select reviewers in such a
way, in which case you can still get the attention of potential reviewers by CC'ing
them in a comment -- just @name them.

If you have received no comments on your patch for a week, you can request a
review by 'ping'ing the GitHub PR with "Ping" in a comment. The common courtesy 'ping' rate
is once a week. Please remember that you are asking for valuable time from
other professional developers.

After your PR is approved, you can merge it. If you do not have the ability to
merge the PR, ask your reviewers to merge it on your behalf. You must do this
explicitly, as reviewers' default assumption is that you are able to merge your
own PR.

For more information on LLVM's code-review process, please see
:doc:`CodeReview`.

.. _commit_from_git:

For developers to commit changes from Git
-----------------------------------------

.. note::
   See also :ref:`GitHub <github-reviews>` for more details on merging your changes
   into LLVM project monorepo.

Once a pull request is approved, you can select the "Squash and merge" button in the
GitHub web interface.

When pushing directly from the command-line to the ``main`` branch, you will need
to rebase your change. LLVM has a linear-history policy, which means
that merge commits are not allowed, and the ``main`` branch is configured to reject
pushes that include merges.

GitHub will display a message that looks like this:

.. code-block:: console

  remote: Bypassed rule violations for refs/heads/main:
  remote:
  remote: - Required status check “buildkite/github-pull-requests” is expected.

This can seem scary, but this is just an artifact of the GitHub setup: it is
intended as a warning for people merging pull-requests with failing CI. We can't
disable it for people pushing on the command-line.

Please ask for help if you're having trouble with your particular git workflow.

.. _git_pre_push_hook:

Git pre-push hook
^^^^^^^^^^^^^^^^^

We include an optional pre-push hook that runs some sanity checks on the revisions
you are about to push and asks for confirmation if you push multiple commits at once.
You can set it up (on Unix systems) by running from the repository root:

.. code-block:: console

  % ln -sf ../../llvm/utils/git/pre-push.py .git/hooks/pre-push

Helpful Information About LLVM
==============================
:doc:`LLVM's documentation <index>` provides a wealth of information about LLVM's internals as
well as various user guides. The pages listed below should provide a good overview
of LLVM's high-level design, as well as its internals:

:doc:`GettingStarted`
   Discusses how to get up and running quickly with the LLVM infrastructure.
   Everything from unpacking and compilation of the distribution to execution
   of some tools.

:doc:`LangRef`
  Defines the LLVM intermediate representation.

:doc:`ProgrammersManual`
  Introduction to the general layout of the LLVM sourcebase, important classes
  and APIs, and some tips & tricks.

`LLVM for Grad Students`__
  This is an introduction to the LLVM infrastructure by Adrian Sampson. While it
  has been written for grad students, it provides  a good, compact overview of
  LLVM's architecture, LLVM's IR and how to write a new pass.

  .. __: http://www.cs.cornell.edu/~asampson/blog/llvm.html

`Intro to LLVM`__
  Book chapter providing a compiler hacker's introduction to LLVM.

  .. __: http://www.aosabook.org/en/llvm.html

.. _Forum: https://discourse.llvm.org
.. _Discord server: https://discord.gg/xS7Z362
.. _irc.oftc.net: irc://irc.oftc.net/llvm
.. _good first issue: https://github.com/llvm/llvm-project/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22
.. _bug tracker: https://github.com/llvm/llvm-project/issues
.. _clang-format-diff.py: https://github.com/llvm/llvm-project/blob/main/clang/tools/clang-format/clang-format-diff.py
.. _git-clang-format: https://github.com/llvm/llvm-project/blob/main/clang/tools/clang-format/git-clang-format
.. _LLVM's GitHub: https://github.com/llvm/llvm-project
.. _LLVM's Phabricator (read-only): https://reviews.llvm.org/
.. _LLVM's Open Projects page: https://llvm.org/OpenProjects.html#what
