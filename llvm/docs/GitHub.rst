.. _github-reviews:

======================
LLVM GitHub User Guide
======================

Introduction
============
The LLVM Project uses `GitHub <https://github.com/>`_ for
`Source Code <https://github.com/llvm/llvm-project>`_,
`Releases <https://github.com/llvm/llvm-project/releases>`_,
`Issue Tracking <https://github.com/llvm/llvm-project/issues>`_., and
`Code Reviews <https://github.com/llvm/llvm-project/pulls>`_.

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
The LLVM project is using GitHub Pull Requests for Code Reviews. This document
describes the typical workflow of creating a Pull Request and getting it reviewed
and accepted. This is meant as an overview of the GitHub workflow, for compelete
documentation refer to `GitHubs documentation <https://docs.github.com/pull-requests>`_.

GitHub Tools
------------
You can interact with GitHub in several ways: via git command line tools,
the web browser, `GitHub Desktop <https://desktop.github.com/>`_, or the
`GitHub CLI <https://cli.github.com>`_. This guide will cover the git command line
tools and the GitHub CLI. The GitHub CLI (`gh`) will be most like the `arc` workflow and
recommended.

Creating Pull Requests
----------------------
For pull requests, please push a branch to your
`fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks>`_
of the llvm-project and
`create a pull request from the fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`_.

Creating Pull Requests with GitHub CLI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
With the CLI it's enough to create the branch locally and then run:

::

  gh pr create

When promted select to create and use your own fork and follow
the instructions to add more information needed.

.. note::

  When you let the GitHub CLI to create a fork of llvm-project to
  your user, it will change the git "remotes" so that "origin" points
  to your fork and "upstream" points to the main llvm-project repository.

Updating Pull Requests
----------------------
When updating a pull request, you should push additional "fix up" commits to
your branch instead of force pushing.  This makes it easier for GitHub to
track the context of previous review comments.

If you do this, you must squash and merge before committing and
you must use the pull request title and description as the commit message.
The default commit message for a squashed pull request is the pull request
description, so this will allow reviewers to review the commit message before
approving the commit.

When pushing to your branch, make sure you push to the correct fork. Check your
remotes with:

::

  git remote -v

And make sure you push to the remote that's pointing to your fork.

Rebasing Pull Requests and Force Pushes
---------------------------------------
In general, you should avoid rebasing a Pull Request and force pushing to the
branch that's the root of the Pull Request during the review. This action will
make the context of the old changes and comments harder to find and read.

Sometimes, a rebase might be needed to update your branch with a fix for a test
or in some dependent code.

After your PR is reviewed and accepted, you want to rebase your branch to ensure
you won't encounter merge conflicts when landing the PR.

Landing your change
-------------------
When your PR has been accepted you can use the web interface to land your change.
The button that should be used is called `Squash and merge` and after you can
select the option `Delete branch` to delete the branch from your fork.

You can also merge via the CLI by switch to your branch locally and run:

::

  gh pr merge --squash --delete-branch


Checking out another PR locally
-------------------------------
Sometimes you want to review another persons PR on your local machine to run
tests or inspect code in your prefered editor. This is easily done with the
CLI:

::

  gh pr checkout <PR Number>

This is also possible with the web interface and the normal git command line
tools, but the process is a bit more complicated. See GitHubs
`documentation <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/checking-out-pull-requests-locally?platform=linux&tool=webui#modifying-an-inactive-pull-request-locally>`_
on the topic.

Example Pull Request with GitHub CLI
====================================
Here is an example for creating a Pull Request with the GitHub CLI:

::

  # Clone the repo
  gh repo clone llvm/llvm-project

  # Switch to the repo and create a new branch
  cd llvm-project
  git switch -c my_change

  # Create your changes
  $EDITOR file.cpp

  # Don't forget clang-format
  git clang-format

  # Commit, use a good commit message
  git commit file.cpp

  # Create the PR, select to use your own fork when prompted.
  gh pr create

  # If you get any review comments, come back to the branch and
  # adjust them.
  git switch my_change
  $EDITOR file.cpp

  # Commit your changes
  git commit file.cpp -m "Code Review adjustments"

  # Push your changes to your fork branch, be mindful of
  # your remotes here, if you don't remember what points to your
  # fork, use git remote -v to see. Usually origin points to your
  # fork and upstream to llvm/llvm-project
  git push origin my_change

  # When your PR is accepted, you can now rebase it and make sure
  # you have all the latest changes.
  git rebase -i origin/main

  # Now merge it
  gh pr merge --squash --delete

Releases
========

Backporting Fixes to the Release Branches
-----------------------------------------
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
