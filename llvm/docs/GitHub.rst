.. _github-reviews:

======================
LLVM GitHub User Guide
======================

.. contents::
   :local:

Introduction
============
The LLVM Project uses `GitHub <https://github.com/>`_ for
`Source Code <https://github.com/llvm/llvm-project>`_,
`Releases <https://github.com/llvm/llvm-project/releases>`_,
`Issue Tracking <https://github.com/llvm/llvm-project/issues>`_., and
`Code Reviews <https://github.com/llvm/llvm-project/pulls>`_.

This page describes how the LLVM Project users and developers can
participate in the project using GitHub.

Before your first PR
====================

Please ensure that you have set a valid email address in your GitHub account,
see :ref:`github-email-address`.

Pull Requests
=============
The LLVM project is using GitHub Pull Requests for Code Reviews. This document
describes the typical workflow of creating a Pull Request and getting it reviewed
and accepted. This is meant as an overview of the GitHub workflow, for complete
documentation refer to `GitHub's documentation <https://docs.github.com/pull-requests>`_.

.. note::
   If you are using a Pull Request for purposes other than review
   (eg: precommit CI results, convenient web-based reverts, etc)
   add the `skip-precommit-approval <https://github.com/llvm/llvm-project/labels?q=skip-precommit-approval>`_
   label to the PR.

GitHub Tools
------------
You can interact with GitHub in several ways: via git command line tools,
the web browser, `GitHub Desktop <https://desktop.github.com/>`_, or the
`GitHub CLI <https://cli.github.com>`_. This guide will cover the git command line
tools and the GitHub CLI.

Creating Pull Requests
----------------------
Keep in mind that when creating a pull request, it should generally only contain one
self-contained commit initially.
This makes it easier for reviewers to understand the introduced changes and
provide feedback. It also helps maintain a clear and organized commit history
for the project. If you have multiple changes you want to introduce, it's
recommended to create separate pull requests for each change.

Create a local branch per commit you want to submit and then push that branch
to your `fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks>`_
of the llvm-project and
`create a pull request from the fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`_.
As GitHub uses the first line of the commit message truncated to 72 characters
as the pull request title, you may have to edit to reword or to undo this
truncation.

Creating Pull Requests with GitHub CLI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
With the CLI it's enough to create the branch locally and then run:

::

  gh pr create

When prompted select to create and use your own fork and follow
the instructions to add more information needed.

.. note::

  When you let the GitHub CLI create a fork of llvm-project to
  your user, it will change the git "remotes" so that "origin" points
  to your fork and "upstream" points to the main llvm-project repository.

Updating Pull Requests
----------------------
In order to update your pull request, the only thing you need to do is to push
your new commits to the branch in your fork. That will automatically update
the pull request.

When updating a pull request, you should push additional "fix up" commits to
your branch instead of force pushing. This makes it easier for GitHub to
track the context of previous review comments. Consider using the
`built-in support for fixups <https://git-scm.com/docs/git-commit#Documentation/git-commit.txt---fixupamendrewordltcommitgt>`_
in git.

If you do this, you must squash and merge before landing the PR and
you must use the pull request title and description as the commit message.
You can do this manually with an interactive git rebase or with GitHub's
built-in tool. See the section about landing your fix below.

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

.. note::
  This guide assumes that the PR branch only has 1 author. If you are
  collaborating with others on a single branch, be careful how and when you push
  changes. ``--force-with-lease`` may be useful in this situation.

Approvals
---------

Before merging a PR you must have the required approvals. See
:ref:`lgtm_how_a_patch_is_accepted` for more details.


Landing your change
-------------------

After your PR is approved, ensure that:

  * The PR title and description describe the final changes. These will be used
    as the title and message of the final squashed commit. The titles and
    messages of commits in the PR will **not** be used.
  * You have set a valid email address in your GitHub account, see :ref:`github-email-address`.

.. note::
   The LLVM Project monorepo on GitHub is configured to always use "Squash
   and Merge" as the pull request merge option when using the web interface.
   With this option, GitHub uses the PR summary as the default commit
   message.

   Users with write access who can merge PRs have a final opportunity to edit
   the commit title and message before merging. However, this option is not
   available to contributors without write access.

At this point, you can merge your changes. If you do not have write permissions
for the repository, the merge button in GitHub's web interface will be
disabled. If this is the case, continue following the steps here but ask one of
your reviewers to click the merge button on your behalf.

If the PR is a single commit, all you need to do is click the merge button in
GitHub's web interface.

If your PR contains multiple commits, you need to consolidate those commits into
one commit. There are three different ways to do this, shown here with the most
commonly used first:

* Use the button `Squash and merge` in GitHub's web interface, if you do this
  remember to review the commit message when prompted.

  Afterwards you can select the option `Delete branch` to delete the branch
  from your fork.

* `Interactive rebase <https://git-scm.com/docs/git-rebase#_interactive_mode>`_
  with fixups. This is the recommended method since you can control the final
  commit message and check that the final commit looks as you expect. When
  your local state is correct, remember to force-push to your branch and press
  the merge button in GitHub's web interface afterwards.

* Merge using the GitHub command line interface. Switch to your branch locally
  and run:

  ::

    gh pr merge --squash --delete-branch

  If you observe an error message from the above informing you that your pull
  request is not mergeable, then that is likely because upstream has been
  modified since your pull request was authored in a way that now results in a
  merge conflict. You must first resolve this merge conflict in order to merge
  your pull request. In order to do that:

  ::

    git fetch upstream
    git rebase upstream/main

  Then fix the source files causing merge conflicts and make sure to rebuild and
  retest the result. Then:

  ::

    git add <files with resolved merge conflicts>
    git rebase --continue

  Finally, you'll need to force push to your branch one more time before you can
  merge:

  ::

    git push --force
    gh pr merge --squash --delete-branch

  This force push may ask if you intend to push hundreds, or potentially
  thousands of patches (depending on how long it's been since your pull request
  was initially authored vs. when you intended to merge it). Since you're pushing
  to a branch in your fork, this is ok and expected. Github's UI for the pull
  request will understand that you're rebasing just your patches, and display
  this result correctly with a note that a force push did occur.

.. _github_branches:

Branches
========

It is possible to create branches in `llvm/llvm-project/` that start with
`users/<username>/`, however this is intended to be able to support "stacked"
pull-request. Do not create any branches in the `llvm/llvm-project` repository
otherwise, please use a fork (see above). User branches that aren't
associated with a pull-request **will be deleted**.

Stacked Pull Requests
=====================

To separate related changes or to break down a larger PR into smaller, reviewable
pieces, use "stacked pull requests" — this helps make the review process
smoother.

.. note::
   The LLVM Project monorepo on GitHub is configured to always use "Squash and
   Merge" as the pull request merge option. As a result, each PR results in
   exactly one commit being merged into the project.

   This means that stacked pull requests are the only available option for
   landing a series of related changes. In contrast, submitting a PR with
   multiple commits and merging them as-is (without squashing) is not supported
   in LLVM.

While GitHub does not natively support stacked pull requests, there are several
common alternatives.

To illustrate, assume that you are working on two branches in your fork of the
``llvm/llvm-project`` repository, and you want to eventually merge both into
``main``:

- `feature_1`, which contains commit `feature_commit_1`
- `feature_2`, which contains commit `feature_commit_2` and depends on
  `feature_1` (so it also includes `feature_commit_1`)

Your options are as follows:

#. Use user branches in ``llvm/llvm-project``

   Create user branches in the main repository, as described
   :ref:`above<github_branches>`. Then:

   - Open a pull request from `users/<username>/feature_1` → `main`
   - Open another from `users/<username>/feature_2` → `users/<username>/feature_1`

   This approach allows GitHub to display clean, incremental diffs for each PR
   in the stack, making it much easier for reviewers to see what has changed at
   each step. Once `feature_1` is merged, GitHub will automatically rebase and
   re-target your branch `feature_2` to `main`. For more complex stacks, you can
   perform this step using the web interface.

   This approach requires commit access. See how to obtain it
   `here <https://llvm.org/docs/DeveloperPolicy.html#obtaining-commit-access>`_.

#. Two PRs with a dependency note

   Create PR_1 for `feature_1` and PR_2 for `feature_2`. In PR_2, include a
   note in the PR summary indicating that it depends on PR_1 (e.g.,
   “Depends on #PR_1”).

   To make review easier, make it clear which commits are part of the base PR
   and which are new, e.g. "The first N commits are from the base PR". This
   helps reviewers focus only on the incremental changes.

#. Use a stacked PR tool

   Use tools like SPR or Graphite (described below) to automate managing
   stacked PRs. These tools are also based on using user branches
   in ``llvm/llvm-project``.

.. note::
   When not using user branches, GitHub will not display proper diffs for
   subsequent PRs in a stack. Instead, it will show a combined diff that
   includes all commits from earlier PRs.

   As described above, it is the PR author’s responsibility to clearly indicate
   which commits are relevant to the current PR.
   For example: “The first N commits are from the base PR.”

   You can avoid this issue by using user branches directly in the
   ``llvm/llvm-project`` repository.


Using Graphite for stacked Pull Requests
----------------------------------------

`Graphite <https://app.graphite.dev/>`_ is a stacked pull request tool supported
by the LLVM repo (the other being `reviewable.io <https://reviewable.io>`_).

Graphite will want to create branches under ``llvm/llvm-project`` rather than your
private fork, so the guidance above, about branch naming, is critical, otherwise
``gt submit`` (i.e. publish your PRs for review) will fail.

Use ``gt config`` then ``Branch naming settings`` and ``Set a prefix for branch names``.
Include the last ``/``.

If you didn't do the above and Graphite created non-prefixed branches, a simple way to
unblock is to rename (``git -m <old name> <new name>``), and then checkout the branch
and ``gt track``.

Pre-merge Continuous Integration (CI)
-------------------------------------

Multiple checks will be applied on a pull-request, either for linting/formatting
or some build and tests. None of these are perfect and you will encounter
false positive, infrastructure failures (unstable or unavailable worker), or
you will be unlucky and based your change on a broken revision of the main branch.

None of the checks are strictly mandatory: these are tools to help us build a
better codebase and be more productive (by avoiding issues found post-merge and
possible reverts). As a developer you're empowered to exercise your judgement
about bypassing any of the checks when merging code.

The infrastructure can print messages that make it seem like these are mandatory,
but this is just an artifact of GitHub infrastructure and not a policy of the
project.

However, please make sure you do not force-merge any changes that have clear
test failures directly linked to your changes. Our policy is still to keep the
``main`` branch in a good condition, and introducing failures to be fixed later
violates that policy.

Problems After Landing Your Change
==================================

Even though your PR passed the pre-commit checks and is approved by reviewers, it
may cause problems for some configurations after it lands. You will be notified
if this happens and the community is ready to help you fix the problems.

This process is described in detail
:ref:`here <MyFirstTypoFix Issues After Landing Your PR>`.


Checking out another PR locally
-------------------------------
Sometimes you want to review another person's PR on your local machine to run
tests or inspect code in your preferred editor. This is easily done with the
CLI:

::

  gh pr checkout <PR Number>

This is also possible with the web interface and the normal git command line
tools, but the process is a bit more complicated. See GitHub's
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

  # and don't forget running your tests
  ninja check-llvm

  # Commit, use a good commit message
  git commit file.cpp

  # Create the PR, select to use your own fork when prompted.
  # If you don't have a fork, gh will create one for you.
  gh pr create

  # If you get any review comments, come back to the branch and
  # adjust them.
  git switch my_change
  $EDITOR file.cpp

  # Commit your changes
  git commit file.cpp -m "Code Review adjustments"

  # Format changes
  git clang-format HEAD~

  # Recommit if any formatting changes
  git commit -a --amend

  # Push your changes to your fork branch, be mindful of
  # your remotes here, if you don't remember what points to your
  # fork, use git remote -v to see. Usually origin points to your
  # fork and upstream to llvm/llvm-project
  git push origin my_change

Before merging the PR, it is recommended that you rebase locally and re-run test
checks:

::

  # Add upstream as a remote (if you don't have it already)
  git remote add upstream https://github.com/llvm/llvm-project.git

  # Make sure you have all the latest changes
  git fetch upstream && git rebase -i upstream/main

  # Make sure tests pass with latest changes and your change
  ninja check

  # Push the rebased changes to your fork.
  git push origin my_change --force

  # Now merge it
  gh pr merge --squash --delete-branch


See more in-depth information about how to contribute in the following documentation:

* :doc:`Contributing`
* :doc:`MyFirstTypoFix`

Example Pull Request with git
====================================

Instead of using the GitHub CLI to create a PR, you can push your code to a
remote branch on your fork and create the PR to upstream using the GitHub web
interface.

Here is an example of making a PR using git and the GitHub web interface:

First follow the instructions to `fork the repository <https://docs.github.com/en/get-started/quickstart/fork-a-repo?tool=webui#forking-a-repository>`_.

Next follow the instructions to `clone your forked repository <https://docs.github.com/en/get-started/quickstart/fork-a-repo?tool=webui#cloning-your-forked-repository>`_.

Once you've cloned your forked repository,

::

  # Switch to the forked repo
  cd llvm-project

  # Create a new branch
  git switch -c my_change

  # Create your changes
  $EDITOR file.cpp

  # Don't forget clang-format
  git clang-format

  # and don't forget running your tests
  ninja check-llvm

  # Commit, use a good commit message
  git commit file.cpp

  # Push your changes to your fork branch, be mindful of
  # your remotes here, if you don't remember what points to your
  # fork, use git remote -v to see. Usually origin points to your
  # fork and upstream to llvm/llvm-project
  git push origin my_change

Navigate to the URL printed to the console from the git push command in the last step.
Create a pull request from your branch to llvm::main.

::

  # If you get any review comments, come back to the branch and
  # adjust them.
  git switch my_change
  $EDITOR file.cpp

  # Commit your changes
  git commit file.cpp -m "Code Review adjustments"

  # Format changes
  git clang-format HEAD~

  # Recommit if any formatting changes
  git commit -a --amend

  # Re-run tests and make sure nothing broke.
  ninja check

  # Push your changes to your fork branch, be mindful of
  # your remotes here, if you don't remember what points to your
  # fork, use git remote -v to see. Usually origin points to your
  # fork and upstream to llvm/llvm-project
  git push origin my_change

Before merging the PR, it is recommended that you rebase locally and re-run test
checks:

::

  # Add upstream as a remote (if you don't have it already)
  git remote add upstream https://github.com/llvm/llvm-project.git

  # Make sure you have all the latest changes
  git fetch upstream && git rebase -i upstream/main

  # Make sure tests pass with latest changes and your change
  ninja check

  # Push the rebased changes to your fork.
  git push origin my_change --force

Once your PR is approved, rebased, and tests are passing, click `Squash and
Merge` on your PR in the GitHub web interface.

See more in-depth information about how to contribute in the following documentation:

* :doc:`Contributing`
* :doc:`MyFirstTypoFix`

Releases
========

.. _backporting:

Backporting Fixes to the Release Branches
-----------------------------------------
You can use special comments on issues or pull requests to make backport
requests for the release branches.  To do this, after your pull request has been
merged:

1. Edit "Milestone" at the right side of the issue or pull request
   to say "LLVM X.Y Release"

2. Add a comment to it in the following format:

::

  /cherry-pick <commit> <commit> <...>

This command takes one or more git commit hashes as arguments and will attempt
to cherry-pick the commit(s) to the release branch.  If the commit(s) fail to
apply cleanly, then a comment with a link to the failing job will be added to
the issue/pull request.  If the commit(s) do apply cleanly, then a pull request
will be created with the specified commits.

If a commit you want to backport does not apply cleanly, you may resolve
the conflicts locally and then create a pull request against the release
branch.  Just make sure to add the release milestone to the pull request.

Getting admin access to CI infrastructure
=========================================

Any individual who is responsible for setting up and/or maintaining CI
infrastructure for a LLVM project can request to be granted the CI/CD role by
the LLVM infrastructure area team. The request can be made by creating `a
Github issue <https://github.com/llvm/llvm-project/issues/new>`_ and using the
``infrastructure`` label.  Applicants must include a justification for why the
role is being requested. Applications are reviewed on a case-by-case basis by
the LLVM infrastructure area team and the role can be revoked at any point as
the area team sees fit.
