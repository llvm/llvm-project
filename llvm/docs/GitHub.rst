======================
LLVM GitHub User Guide
======================

Introduction
============
The LLVM Project uses `GitHub <https://github.com/>`_ for
`Source Code <https://github.com/llvm/llvm-project>`_,
`Releases <https://github.com/llvm/llvm-project/releases>`_, and
`Issue Tracking <https://github.com/llvm/llvm-project/issues>`_.

This page describes how the LLVM Project users and developers can
participate in the project using GitHub.

Branches
========
Do not create any branches in the llvm/llvm-project repository.  This repository
is reserved for official project branches only.  We may relax this rule in
the future if needed to support "stacked" pull request, but in that case only
branches being used for "stacked" pull requests will be allowed.

Pull Requests
=============
The LLVM Project does not currently accept pull requests for the llvm/llvm-project
repository.  However, there is a
`plan <https://discourse.llvm.org/t/code-review-process-update/63964>`_ to move
to pull requests in the future.  This section documents the pull request
policies LLVM will be adopting once the project starts using them.

Creating Pull Requests
^^^^^^^^^^^^^^^^^^^^^^
For pull requests, please push a branch to your
`fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks>`_
of the llvm-project and
`create a pull request from the fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`_.

Updating Pull Requests
^^^^^^^^^^^^^^^^^^^^^^
When updating a pull request, you should push additional "fix up" commits to
your branch instead of force pushing.  This makes it easier for GitHub to
track the context of previous review comments.

If you do this, you must squash and merge before committing and
you must use the pull request title and description as the commit message.
The default commit message for a squashed pull request is the pull request
description, so this will allow reviewers to review the commit message before
approving the commit.

Releases
========

Backporting Fixes to the Release Branches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can use special comments on issues to make backport requests for the
release branches.  This is done by making a comment containing one of the
following commands on any issue that has been added to one of the "X.Y.Z Release"
milestones.

::

  /cherry-pick <commit> <commit> <...>

This command takes one or more git commit hashes as arguments and will attempt
to cherry-pick the commit(s) to the release branch.  If the commit(s) fail to
apply cleanly, then a comment with a link to the failing job will be added to
the issue.  If the commit(s) do apply cleanly, then a pull request will
be created with the specified commits.

::

  /branch <owner>/<repo>/<branch>

This command will create a pull request against the latest release branch using
the <branch> from the <owner>/<repo> repository.  <branch> cannot contain any
forward slash '/' characters.
