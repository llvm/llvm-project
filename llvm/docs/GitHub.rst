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
