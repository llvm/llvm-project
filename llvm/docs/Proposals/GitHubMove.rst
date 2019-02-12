==============================
Moving LLVM Projects to GitHub
==============================

.. contents:: Table of Contents
  :depth: 4
  :local:

Introduction
============

This is a proposal to move our current revision control system from our own
hosted Subversion to GitHub. Below are the financial and technical arguments as
to why we are proposing such a move and how people (and validation
infrastructure) will continue to work with a Git-based LLVM.

There will be a survey pointing at this document which we'll use to gauge the
community's reaction and, if we collectively decide to move, the time-frame. Be
sure to make your view count.

Additionally, we will discuss this during a BoF at the next US LLVM Developer
meeting (http://llvm.org/devmtg/2016-11/).

What This Proposal is *Not* About
=================================

Changing the development policy.

This proposal relates only to moving the hosting of our source-code repository
from SVN hosted on our own servers to Git hosted on GitHub. We are not proposing
using GitHub's issue tracker, pull-requests, or code-review.

Contributors will continue to earn commit access on demand under the Developer
Policy, except that that a GitHub account will be required instead of SVN
username/password-hash.

Why Git, and Why GitHub?
========================

Why Move At All?
----------------

This discussion began because we currently host our own Subversion server
and Git mirror on a voluntary basis. The LLVM Foundation sponsors the server and
provides limited support, but there is only so much it can do.

Volunteers are not sysadmins themselves, but compiler engineers that happen
to know a thing or two about hosting servers. We also don't have 24/7 support,
and we sometimes wake up to see that continuous integration is broken because
the SVN server is either down or unresponsive.

We should take advantage of one of the services out there (GitHub, GitLab,
and BitBucket, among others) that offer better service (24/7 stability, disk
space, Git server, code browsing, forking facilities, etc) for free.

Why Git?
--------

Many new coders nowadays start with Git, and a lot of people have never used
SVN, CVS, or anything else. Websites like GitHub have changed the landscape
of open source contributions, reducing the cost of first contribution and
fostering collaboration.

Git is also the version control many LLVM developers use. Despite the
sources being stored in a SVN server, these developers are already using Git
through the Git-SVN integration.

Git allows you to:

* Commit, squash, merge, and fork locally without touching the remote server.
* Maintain local branches, enabling multiple threads of development.
* Collaborate on these branches (e.g. through your own fork of llvm on GitHub).
* Inspect the repository history (blame, log, bisect) without Internet access.
* Maintain remote forks and branches on Git hosting services and
  integrate back to the main repository.

In addition, because Git seems to be replacing many OSS projects' version
control systems, there are many tools that are built over Git.
Future tooling may support Git first (if not only).

Why GitHub?
-----------

GitHub, like GitLab and BitBucket, provides free code hosting for open source
projects. Any of these could replace the code-hosting infrastructure that we
have today.

These services also have a dedicated team to monitor, migrate, improve and
distribute the contents of the repositories depending on region and load.

GitHub has one important advantage over GitLab and
BitBucket: it offers read-write **SVN** access to the repository
(https://github.com/blog/626-announcing-svn-support).
This would enable people to continue working post-migration as though our code
were still canonically in an SVN repository.

In addition, there are already multiple LLVM mirrors on GitHub, indicating that
part of our community has already settled there.

On Managing Revision Numbers with Git
-------------------------------------

The current SVN repository hosts all the LLVM sub-projects alongside each other.
A single revision number (e.g. r123456) thus identifies a consistent version of
all LLVM sub-projects.

Git does not use sequential integer revision number but instead uses a hash to
identify each commit.

The loss of a sequential integer revision number has been a sticking point in
past discussions about Git:

- "The 'branch' I most care about is mainline, and losing the ability to say
  'fixed in r1234' (with some sort of monotonically increasing number) would
  be a tragic loss." [LattnerRevNum]_
- "I like those results sorted by time and the chronology should be obvious, but
  timestamps are incredibly cumbersome and make it difficult to verify that a
  given checkout matches a given set of results." [TrickRevNum]_
- "There is still the major regression with unreadable version numbers.
  Given the amount of Bugzilla traffic with 'Fixed in...', that's a
  non-trivial issue." [JSonnRevNum]_
- "Sequential IDs are important for LNT and llvmlab bisection tool." [MatthewsRevNum]_.

However, Git can emulate this increasing revision number:
``git rev-list --count <commit-hash>``. This identifier is unique only
within a single branch, but this means the tuple `(num, branch-name)` uniquely
identifies a commit.

We can thus use this revision number to ensure that e.g. `clang -v` reports a
user-friendly revision number (e.g. `master-12345` or `4.0-5321`), addressing
the objections raised above with respect to this aspect of Git.

What About Branches and Merges?
-------------------------------

In contrast to SVN, Git makes branching easy. Git's commit history is
represented as a DAG, a departure from SVN's linear history. However, we propose
to mandate making merge commits illegal in our canonical Git repository.

Unfortunately, GitHub does not support server side hooks to enforce such a
policy.  We must rely on the community to avoid pushing merge commits.

GitHub offers a feature called `Status Checks`: a branch protected by
`status checks` requires commits to be whitelisted before the push can happen.
We could supply a pre-push hook on the client side that would run and check the
history, before whitelisting the commit being pushed [statuschecks]_.
However this solution would be somewhat fragile (how do you update a script
installed on every developer machine?) and prevents SVN access to the
repository.

What About Commit Emails?
-------------------------

We will need a new bot to send emails for each commit. This proposal leaves the
email format unchanged besides the commit URL.

Straw Man Migration Plan
========================

Step #1 : Before The Move
-------------------------

1. Update docs to mention the move, so people are aware of what is going on.
2. Set up a read-only version of the GitHub project, mirroring our current SVN
   repository.
3. Add the required bots to implement the commit emails, as well as the
   umbrella repository update (if the multirepo is selected) or the read-only
   Git views for the sub-projects (if the monorepo is selected).

Step #2 : Git Move
------------------

4. Update the buildbots to pick up updates and commits from the GitHub
   repository. Not all bots have to migrate at this point, but it'll help
   provide infrastructure testing.
5. Update Phabricator to pick up commits from the GitHub repository.
6. LNT and llvmlab have to be updated: they rely on unique monotonically
   increasing integer across branch [MatthewsRevNum]_.
7. Instruct downstream integrators to pick up commits from the GitHub
   repository.
8. Review and prepare an update for the LLVM documentation.

Until this point nothing has changed for developers, it will just
boil down to a lot of work for buildbot and other infrastructure
owners.

The migration will pause here until all dependencies have cleared, and all
problems have been solved.

Step #3: Write Access Move
--------------------------

9. Collect developers' GitHub account information, and add them to the project.
10. Switch the SVN repository to read-only and allow pushes to the GitHub repository.
11. Update the documentation.
12. Mirror Git to SVN.

Step #4 : Post Move
-------------------

13. Archive the SVN repository.
14. Update links on the LLVM website pointing to viewvc/klaus/phab etc. to
    point to GitHub instead.

One or Multiple Repositories?
=============================

There are two major variants for how to structure our Git repository: The
"multirepo" and the "monorepo".

Multirepo Variant
-----------------

This variant recommends moving each LLVM sub-project to a separate Git
repository. This mimics the existing official read-only Git repositories
(e.g., http://llvm.org/git/compiler-rt.git), and creates new canonical
repositories for each sub-project.

This will allow the individual sub-projects to remain distinct: a
developer interested only in compiler-rt can checkout only this repository,
build it, and work in isolation of the other sub-projects.

A key need is to be able to check out multiple projects (i.e. lldb+clang+llvm or
clang+llvm+libcxx for example) at a specific revision.

A tuple of revisions (one entry per repository) accurately describes the state
across the sub-projects.
For example, a given version of clang would be
*<LLVM-12345, clang-5432, libcxx-123, etc.>*.

Umbrella Repository
^^^^^^^^^^^^^^^^^^^

To make this more convenient, a separate *umbrella* repository will be
provided. This repository will be used for the sole purpose of understanding
the sequence in which commits were pushed to the different repositories and to
provide a single revision number.

This umbrella repository will be read-only and continuously updated
to record the above tuple. The proposed form to record this is to use Git
[submodules]_, possibly along with a set of scripts to help check out a
specific revision of the LLVM distribution.

A regular LLVM developer does not need to interact with the umbrella repository
-- the individual repositories can be checked out independently -- but you would
need to use the umbrella repository to bisect multiple sub-projects at the same
time, or to check-out old revisions of LLVM with another sub-project at a
consistent state.

This umbrella repository will be updated automatically by a bot (running on
notice from a webhook on every push, and periodically) on a per commit basis: a
single commit in the umbrella repository would match a single commit in a
sub-project.

Living Downstream
^^^^^^^^^^^^^^^^^

Downstream SVN users can use the read/write SVN bridges with the following
caveats:

 * Be prepared for a one-time change to the upstream revision numbers.
 * The upstream sub-project revision numbers will no longer be in sync.

Downstream Git users can continue without any major changes, with the minor
change of upstreaming using `git push` instead of `git svn dcommit`.

Git users also have the option of adopting an umbrella repository downstream.
The tooling for the upstream umbrella can easily be reused for downstream needs,
incorporating extra sub-projects and branching in parallel with sub-project
branches.

Multirepo Preview
^^^^^^^^^^^^^^^^^

As a preview (disclaimer: this rough prototype, not polished and not
representative of the final solution), you can look at the following:

  * Repository: https://github.com/llvm-beanz/llvm-submodules
  * Update bot: http://beanz-bot.com:8180/jenkins/job/submodule-update/

Concerns
^^^^^^^^

 * Because GitHub does not allow server-side hooks, and because there is no
   "push timestamp" in Git, the umbrella repository sequence isn't totally
   exact: commits from different repositories pushed around the same time can
   appear in different orders. However, we don't expect it to be the common case
   or to cause serious issues in practice.
 * You can't have a single cross-projects commit that would update both LLVM and
   other sub-projects (something that can be achieved now). It would be possible
   to establish a protocol whereby users add a special token to their commit
   messages that causes the umbrella repo's updater bot to group all of them
   into a single revision.
 * Another option is to group commits that were pushed closely enough together
   in the umbrella repository. This has the advantage of allowing cross-project
   commits, and is less sensitive to mis-ordering commits. However, this has the
   potential to group unrelated commits together, especially if the bot goes
   down and needs to catch up.
 * This variant relies on heavier tooling. But the current prototype shows that
   it is not out-of-reach.
 * Submodules don't have a good reputation / are complicating the command line.
   However, in the proposed setup, a regular developer will seldom interact with
   submodules directly, and certainly never update them.
 * Refactoring across projects is not friendly: taking some functions from clang
   to make it part of a utility in libSupport wouldn't carry the history of the
   code in the llvm repo, preventing recursively applying `git blame` for
   instance. However, this is not very different than how most people are
   Interacting with the repository today, by splitting such change in multiple
   commits.

Workflows
^^^^^^^^^

 * :ref:`Checkout/Clone a Single Project, without Commit Access <workflow-checkout-commit>`.
 * :ref:`Checkout/Clone a Single Project, with Commit Access <workflow-multicheckout-nocommit>`.
 * :ref:`Checkout/Clone Multiple Projects, with Commit Access <workflow-multicheckout-multicommit>`.
 * :ref:`Commit an API Change in LLVM and Update the Sub-projects <workflow-cross-repo-commit>`.
 * :ref:`Branching/Stashing/Updating for Local Development or Experiments <workflow-multi-branching>`.
 * :ref:`Bisecting <workflow-multi-bisecting>`.

Monorepo Variant
----------------

This variant recommends moving all LLVM sub-projects to a single Git repository,
similar to https://github.com/llvm-project/llvm-project.
This would mimic an export of the current SVN repository, with each sub-project
having its own top-level directory.
Not all sub-projects are used for building toolchains. In practice, www/
and test-suite/ will probably stay out of the monorepo.

Putting all sub-projects in a single checkout makes cross-project refactoring
naturally simple:

 * New sub-projects can be trivially split out for better reuse and/or layering
   (e.g., to allow libSupport and/or LIT to be used by runtimes without adding a
   dependency on LLVM).
 * Changing an API in LLVM and upgrading the sub-projects will always be done in
   a single commit, designing away a common source of temporary build breakage.
 * Moving code across sub-project (during refactoring for instance) in a single
   commit enables accurate `git blame` when tracking code change history.
 * Tooling based on `git grep` works natively across sub-projects, allowing to
   easier find refactoring opportunities across projects (for example reusing a
   datastructure initially in LLDB by moving it into libSupport).
 * Having all the sources present encourages maintaining the other sub-projects
   when changing API.

Finally, the monorepo maintains the property of the existing SVN repository that
the sub-projects move synchronously, and a single revision number (or commit
hash) identifies the state of the development across all projects.

.. _build_single_project:

Building a single sub-project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Nobody will be forced to build unnecessary projects.  The exact structure
is TBD, but making it trivial to configure builds for a single sub-project
(or a subset of sub-projects) is a hard requirement.

As an example, it could look like the following::

  mkdir build && cd build
  # Configure only LLVM (default)
  cmake path/to/monorepo
  # Configure LLVM and lld
  cmake path/to/monorepo -DLLVM_ENABLE_PROJECTS=lld
  # Configure LLVM and clang
  cmake path/to/monorepo -DLLVM_ENABLE_PROJECTS=clang

.. _git-svn-mirror:

Read/write sub-project mirrors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With the Monorepo, the existing single-subproject mirrors (e.g.
http://llvm.org/git/compiler-rt.git) with git-svn read-write access would
continue to be maintained: developers would continue to be able to use the
existing single-subproject git repositories as they do today, with *no changes
to workflow*. Everything (git fetch, git svn dcommit, etc.) could continue to
work identically to how it works today. The monorepo can be set-up such that the
SVN revision number matches the SVN revision in the GitHub SVN-bridge.

Living Downstream
^^^^^^^^^^^^^^^^^

Downstream SVN users can use the read/write SVN bridge. The SVN revision
number can be preserved in the monorepo, minimizing the impact.

Downstream Git users can continue without any major changes, by using the
git-svn mirrors on top of the SVN bridge.

Git users can also work upstream with monorepo even if their downstream
fork has split repositories.  They can apply patches in the appropriate
subdirectories of the monorepo using, e.g., `git am --directory=...`, or
plain `diff` and `patch`.

Alternatively, Git users can migrate their own fork to the monorepo.  As a
demonstration, we've migrated the "CHERI" fork to the monorepo in two ways:

 * Using a script that rewrites history (including merges) so that it looks
   like the fork always lived in the monorepo [LebarCHERI]_.  The upside of
   this is when you check out an old revision, you get a copy of all llvm
   sub-projects at a consistent revision.  (For instance, if it's a clang
   fork, when you check out an old revision you'll get a consistent version
   of llvm proper.)  The downside is that this changes the fork's commit
   hashes.

 * Merging the fork into the monorepo [AminiCHERI]_.  This preserves the
   fork's commit hashes, but when you check out an old commit you only get
   the one sub-project.

Monorepo Preview
^^^^^^^^^^^^^^^^^

As a preview (disclaimer: this rough prototype, not polished and not
representative of the final solution), you can look at the following:

  * Full Repository: https://github.com/joker-eph/llvm-project
  * Single sub-project view with *SVN write access* to the full repo:
    https://github.com/joker-eph/compiler-rt

Concerns
^^^^^^^^

 * Using the monolithic repository may add overhead for those contributing to a
   standalone sub-project, particularly on runtimes like libcxx and compiler-rt
   that don't rely on LLVM; currently, a fresh clone of libcxx is only 15MB (vs.
   1GB for the monorepo), and the commit rate of LLVM may cause more frequent
   `git push` collisions when upstreaming. Affected contributors can continue to
   use the SVN bridge or the single-subproject Git mirrors with git-svn for
   read-write.
 * Using the monolithic repository may add overhead for those *integrating* a
   standalone sub-project, even if they aren't contributing to it, due to the
   same disk space concern as the point above. The availability of the
   sub-project Git mirror addresses this, even without SVN access.
 * Preservation of the existing read/write SVN-based workflows relies on the
   GitHub SVN bridge, which is an extra dependency.  Maintaining this locks us
   into GitHub and could restrict future workflow changes.

Workflows
^^^^^^^^^

 * :ref:`Checkout/Clone a Single Project, without Commit Access <workflow-checkout-commit>`.
 * :ref:`Checkout/Clone a Single Project, with Commit Access <workflow-monocheckout-nocommit>`.
 * :ref:`Checkout/Clone Multiple Projects, with Commit Access <workflow-monocheckout-multicommit>`.
 * :ref:`Commit an API Change in LLVM and Update the Sub-projects <workflow-cross-repo-commit>`.
 * :ref:`Branching/Stashing/Updating for Local Development or Experiments <workflow-mono-branching>`.
 * :ref:`Bisecting <workflow-mono-bisecting>`.

Multi/Mono Hybrid Variant
-------------------------

This variant recommends moving only the LLVM sub-projects that are *rev-locked*
to LLVM into a monorepo (clang, lld, lldb, ...), following the multirepo
proposal for the rest.  While neither variant recommends combining sub-projects
like www/ and test-suite/ (which are completely standalone), this goes further
and keeps sub-projects like libcxx and compiler-rt in their own distinct
repositories.

Concerns
^^^^^^^^

 * This has most disadvantages of multirepo and monorepo, without bringing many
   of the advantages.
 * Downstream have to upgrade to the monorepo structure, but only partially. So
   they will keep the infrastructure to integrate the other separate
   sub-projects.
 * All projects that use LIT for testing are effectively rev-locked to LLVM.
   Furthermore, some runtimes (like compiler-rt) are rev-locked with Clang.
   It's not clear where to draw the lines.


Workflow Before/After
=====================

This section goes through a few examples of workflows, intended to illustrate
how end-users or developers would interact with the repository for
various use-cases.

.. _workflow-checkout-commit:

Checkout/Clone a Single Project, without Commit Access
------------------------------------------------------

Except the URL, nothing changes. The possibilities today are::

  svn co http://llvm.org/svn/llvm-project/llvm/trunk llvm
  # or with Git
  git clone http://llvm.org/git/llvm.git

After the move to GitHub, you would do either::

  git clone https://github.com/llvm-project/llvm.git
  # or using the GitHub svn native bridge
  svn co https://github.com/llvm-project/llvm/trunk

The above works for both the monorepo and the multirepo, as we'll maintain the
existing read-only views of the individual sub-projects.

Checkout/Clone a Single Project, with Commit Access
---------------------------------------------------

Currently
^^^^^^^^^

::

  # direct SVN checkout
  svn co https://user@llvm.org/svn/llvm-project/llvm/trunk llvm
  # or using the read-only Git view, with git-svn
  git clone http://llvm.org/git/llvm.git
  cd llvm
  git svn init https://llvm.org/svn/llvm-project/llvm/trunk --username=<username>
  git config svn-remote.svn.fetch :refs/remotes/origin/master
  git svn rebase -l  # -l avoids fetching ahead of the git mirror.

Commits are performed using `svn commit` or with the sequence `git commit` and
`git svn dcommit`.

.. _workflow-multicheckout-nocommit:

Multirepo Variant
^^^^^^^^^^^^^^^^^

With the multirepo variant, nothing changes but the URL, and commits can be
performed using `svn commit` or `git commit` and `git push`::

  git clone https://github.com/llvm/llvm.git llvm
  # or using the GitHub svn native bridge
  svn co https://github.com/llvm/llvm/trunk/ llvm

.. _workflow-monocheckout-nocommit:

Monorepo Variant
^^^^^^^^^^^^^^^^

With the monorepo variant, there are a few options, depending on your
constraints. First, you could just clone the full repository::

  git clone https://github.com/llvm/llvm-projects.git llvm
  # or using the GitHub svn native bridge
  svn co https://github.com/llvm/llvm-projects/trunk/ llvm

At this point you have every sub-project (llvm, clang, lld, lldb, ...), which
:ref:`doesn't imply you have to build all of them <build_single_project>`. You
can still build only compiler-rt for instance. In this way it's not different
from someone who would check out all the projects with SVN today.

You can commit as normal using `git commit` and `git push` or `svn commit`, and
read the history for a single project (`git log libcxx` for example).

Secondly, there are a few options to avoid checking out all the sources.

**Using the GitHub SVN bridge**

The GitHub SVN native bridge allows to checkout a subdirectory directly:

  svn co https://github.com/llvm/llvm-projects/trunk/compiler-rt compiler-rt  —username=...

This checks out only compiler-rt and provides commit access using "svn commit",
in the same way as it would do today.

**Using a Subproject Git Nirror**

You can use *git-svn* and one of the sub-project mirrors::

  # Clone from the single read-only Git repo
  git clone http://llvm.org/git/llvm.git
  cd llvm
  # Configure the SVN remote and initialize the svn metadata
  $ git svn init https://github.com/joker-eph/llvm-project/trunk/llvm —username=...
  git config svn-remote.svn.fetch :refs/remotes/origin/master
  git svn rebase -l

In this case the repository contains only a single sub-project, and commits can
be made using `git svn dcommit`, again exactly as we do today.

**Using a Sparse Checkouts**

You can hide the other directories using a Git sparse checkout::

  git config core.sparseCheckout true
  echo /compiler-rt > .git/info/sparse-checkout
  git read-tree -mu HEAD

The data for all sub-projects is still in your `.git` directory, but in your
checkout, you only see `compiler-rt`.
Before you push, you'll need to fetch and rebase (`git pull --rebase`) as
usual.

Note that when you fetch you'll likely pull in changes to sub-projects you don't
care about. If you are using spasre checkout, the files from other projects
won't appear on your disk. The only effect is that your commit hash changes.

You can check whether the changes in the last fetch are relevant to your commit
by running::

  git log origin/master@{1}..origin/master -- libcxx

This command can be hidden in a script so that `git llvmpush` would perform all
these steps, fail only if such a dependent change exists, and show immediately
the change that prevented the push. An immediate repeat of the command would
(almost) certainly result in a successful push.
Note that today with SVN or git-svn, this step is not possible since the
"rebase" implicitly happens while committing (unless a conflict occurs).

Checkout/Clone Multiple Projects, with Commit Access
----------------------------------------------------

Let's look how to assemble llvm+clang+libcxx at a given revision.

Currently
^^^^^^^^^

::

  svn co http://llvm.org/svn/llvm-project/llvm/trunk llvm -r $REVISION
  cd llvm/tools
  svn co http://llvm.org/svn/llvm-project/clang/trunk clang -r $REVISION
  cd ../projects
  svn co http://llvm.org/svn/llvm-project/libcxx/trunk libcxx -r $REVISION

Or using git-svn::

  git clone http://llvm.org/git/llvm.git
  cd llvm/
  git svn init https://llvm.org/svn/llvm-project/llvm/trunk --username=<username>
  git config svn-remote.svn.fetch :refs/remotes/origin/master
  git svn rebase -l
  git checkout `git svn find-rev -B r258109`
  cd tools
  git clone http://llvm.org/git/clang.git
  cd clang/
  git svn init https://llvm.org/svn/llvm-project/clang/trunk --username=<username>
  git config svn-remote.svn.fetch :refs/remotes/origin/master
  git svn rebase -l
  git checkout `git svn find-rev -B r258109`
  cd ../../projects/
  git clone http://llvm.org/git/libcxx.git
  cd libcxx
  git svn init https://llvm.org/svn/llvm-project/libcxx/trunk --username=<username>
  git config svn-remote.svn.fetch :refs/remotes/origin/master
  git svn rebase -l
  git checkout `git svn find-rev -B r258109`

Note that the list would be longer with more sub-projects.

.. _workflow-multicheckout-multicommit:

Multirepo Variant
^^^^^^^^^^^^^^^^^

With the multirepo variant, the umbrella repository will be used. This is
where the mapping from a single revision number to the individual repositories
revisions is stored.::

  git clone https://github.com/llvm-beanz/llvm-submodules
  cd llvm-submodules
  git checkout $REVISION
  git submodule init
  git submodule update clang llvm libcxx
  # the list of sub-project is optional, `git submodule update` would get them all.

At this point the clang, llvm, and libcxx individual repositories are cloned
and stored alongside each other. There are CMake flags to describe the directory
structure; alternatively, you can just symlink `clang` to `llvm/tools/clang`,
etc.

Another option is to checkout repositories based on the commit timestamp::

  git checkout `git rev-list -n 1 --before="2009-07-27 13:37" master`

.. _workflow-monocheckout-multicommit:

Monorepo Variant
^^^^^^^^^^^^^^^^

The repository contains natively the source for every sub-projects at the right
revision, which makes this straightforward::

  git clone https://github.com/llvm/llvm-projects.git llvm-projects
  cd llvm-projects
  git checkout $REVISION

As before, at this point clang, llvm, and libcxx are stored in directories
alongside each other.

.. _workflow-cross-repo-commit:

Commit an API Change in LLVM and Update the Sub-projects
--------------------------------------------------------

Today this is possible, even though not common (at least not documented) for
subversion users and for git-svn users. For example, few Git users try to update
LLD or Clang in the same commit as they change an LLVM API.

The multirepo variant does not address this: one would have to commit and push
separately in every individual repository. It would be possible to establish a
protocol whereby users add a special token to their commit messages that causes
the umbrella repo's updater bot to group all of them into a single revision.

The monorepo variant handles this natively.

Branching/Stashing/Updating for Local Development or Experiments
----------------------------------------------------------------

Currently
^^^^^^^^^

SVN does not allow this use case, but developers that are currently using
git-svn can do it. Let's look in practice what it means when dealing with
multiple sub-projects.

To update the repository to tip of trunk::

  git pull
  cd tools/clang
  git pull
  cd ../../projects/libcxx
  git pull

To create a new branch::

  git checkout -b MyBranch
  cd tools/clang
  git checkout -b MyBranch
  cd ../../projects/libcxx
  git checkout -b MyBranch

To switch branches::

  git checkout AnotherBranch
  cd tools/clang
  git checkout AnotherBranch
  cd ../../projects/libcxx
  git checkout AnotherBranch

.. _workflow-multi-branching:

Multirepo Variant
^^^^^^^^^^^^^^^^^

The multirepo works the same as the current Git workflow: every command needs
to be applied to each of the individual repositories.
However, the umbrella repository makes this easy using `git submodule foreach`
to replicate a command on all the individual repositories (or submodules
in this case):

To create a new branch::

  git submodule foreach git checkout -b MyBranch

To switch branches::

  git submodule foreach git checkout AnotherBranch

.. _workflow-mono-branching:

Monorepo Variant
^^^^^^^^^^^^^^^^

Regular Git commands are sufficient, because everything is in a single
repository:

To update the repository to tip of trunk::

  git pull

To create a new branch::

  git checkout -b MyBranch

To switch branches::

  git checkout AnotherBranch

Bisecting
---------

Assuming a developer is looking for a bug in clang (or lld, or lldb, ...).

Currently
^^^^^^^^^

SVN does not have builtin bisection support, but the single revision across
sub-projects makes it possible to script around.

Using the existing Git read-only view of the repositories, it is possible to use
the native Git bisection script over the llvm repository, and use some scripting
to synchronize the clang repository to match the llvm revision.

.. _workflow-multi-bisecting:

Multirepo Variant
^^^^^^^^^^^^^^^^^

With the multi-repositories variant, the cross-repository synchronization is
achieved using the umbrella repository. This repository contains only
submodules for the other sub-projects. The native Git bisection can be used on
the umbrella repository directly. A subtlety is that the bisect script itself
needs to make sure the submodules are updated accordingly.

For example, to find which commit introduces a regression where clang-3.9
crashes but not clang-3.8 passes, one should be able to simply do::

  git bisect start release_39 release_38
  git bisect run ./bisect_script.sh

With the `bisect_script.sh` script being::

  #!/bin/sh
  cd $UMBRELLA_DIRECTORY
  git submodule update llvm clang libcxx #....
  cd $BUILD_DIR

  ninja clang || exit 125   # an exit code of 125 asks "git bisect"
                            # to "skip" the current commit

  ./bin/clang some_crash_test.cpp

When the `git bisect run` command returns, the umbrella repository is set to
the state where the regression is introduced. The commit diff in the umbrella
indicate which submodule was updated, and the last commit in this sub-projects
is the one that the bisect found.

.. _workflow-mono-bisecting:

Monorepo Variant
^^^^^^^^^^^^^^^^

Bisecting on the monorepo is straightforward, and very similar to the above,
except that the bisection script does not need to include the
`git submodule update` step.

The same example, finding which commit introduces a regression where clang-3.9
crashes but not clang-3.8 passes, will look like::

  git bisect start release_39 release_38
  git bisect run ./bisect_script.sh

With the `bisect_script.sh` script being::

  #!/bin/sh
  cd $BUILD_DIR

  ninja clang || exit 125   # an exit code of 125 asks "git bisect"
                            # to "skip" the current commit

  ./bin/clang some_crash_test.cpp

Also, since the monorepo handles commits update across multiple projects, you're
less like to encounter a build failure where a commit change an API in LLVM and
another later one "fixes" the build in clang.

Moving Local Branches to the Monorepo
=====================================

Suppose you have been developing against the existing LLVM git
mirrors.  You have one or more git branches that you want to migrate
to the "final monorepo".

The simplest way to migrate such branches is with the
``migrate-downstream-fork.py`` tool at
https://github.com/jyknight/llvm-git-migration.

Basic migration
---------------

Basic instructions for ``migrate-downstream-fork.py`` are in the
Python script and are expanded on below to a more general recipe::

  # Make a repository which will become your final local mirror of the
  # monorepo.
  mkdir my-monorepo
  git -C my-monorepo init

  # Add a remote to the monorepo.
  git -C my-monorepo remote add upstream/monorepo https://github.com/llvm/llvm-project.git

  # Add remotes for each git mirror you use, from upstream as well as
  # your local mirror.  All projects are listed here but you need only
  # import those for which you have local branches.
  my_projects=( clang
                clang-tools-extra
                compiler-rt
                debuginfo-tests
                libcxx
                libcxxabi
                libunwind
                lld
                lldb
                llvm
                openmp
                polly )
  for p in ${my_projects[@]}; do
    git -C my-monorepo remote add upstream/split/${p} https://github.com/llvm-mirror/${p}.git
    git -C my-monorepo remote add local/split/${p} https://my.local.mirror.org/${p}.git
  done

  # Pull in all the commits.
  git -C my-monorepo fetch --all

  # Run migrate-downstream-fork to rewrite local branches on top of
  # the upstream monorepo.
  (
     cd my-monorepo
     migrate-downstream-fork.py \
       refs/remotes/local \
       refs/tags \
       --new-repo-prefix=refs/remotes/upstream/monorepo \
       --old-repo-prefix=refs/remotes/upstream/split \
       --source-kind=split \
       --revmap-out=monorepo-map.txt
  )

  # Octopus-merge the resulting local split histories to unify them.

  # Assumes local work on local split mirrors is on master (and
  # upstream is presumably represented by some other branch like
  # upstream/master).
  my_local_branch="master"

  git -C my-monorepo branch --no-track local/octopus/master \
    $(git -C my-monorepo merge-base refs/remotes/upstream/monorepo/master \
                                    refs/remotes/local/split/llvm/${my_local_branch})
  git -C my-monorepo checkout local/octopus/${my_local_branch}

  subproject_branches=()
  for p in ${my_projects[@]}; do
    subproject_branch=${p}/local/monorepo/${my_local_branch}
    git -C my-monorepo branch ${subproject_branch} \
      refs/remotes/local/split/${p}/${my_local_branch}
    if [[ "${p}" != "llvm" ]]; then
      subproject_branches+=( ${subproject_branch} )
    fi
  done

  git -C my-monorepo merge ${subproject_branches[@]}

  for p in ${my_projects[@]}; do
    subproject_branch=${p}/local/monorepo/${my_local_branch}
    git -C my-monorepo branch -d ${subproject_branch}
  done

  # Create local branches for upstream monorepo branches.
  for ref in $(git -C my-monorepo for-each-ref --format="%(refname)" \
                   refs/remotes/upstream/monorepo); do
    upstream_branch=${ref#refs/remotes/upstream/monorepo/}
    git -C my-monorepo branch upstream/${upstream_branch} ${ref}
  done

The above gets you to a state like the following::

  U1 - U2 - U3 <- upstream/master
    \   \    \
     \   \    - Llld1 - Llld2 -
      \   \                    \
       \   - Lclang1 - Lclang2-- Lmerge <- local/octopus/master
        \                      /
         - Lllvm1 - Lllvm2-----

Each branched component has its branch rewritten on top of the
monorepo and all components are unified by a giant octopus merge.

If additional active local branches need to be preserved, the above
operations following the assignment to ``my_local_branch`` should be
done for each branch.  Ref paths will need to be updated to map the
local branch to the corresponding upstream branch.  If local branches
have no corresponding upstream branch, then the creation of
``local/octopus/<local branch>`` need not use ``git-merge-base`` to
pinpont its root commit; it may simply be branched from the
appropriate component branch (say, ``llvm/local_release_X``).

Zipping local history
---------------------

The octopus merge is suboptimal for many cases, because walking back
through the history of one component leaves the other components fixed
at a history that likely makes things unbuildable.

Some downstream users track the order commits were made to subprojects
with some kind of "umbrella" project that imports the project git
mirrors as submodules, similar to the multirepo umbrella proposed
above.  Such an umbrella repository looks something like this::

   UM1 ---- UM2 -- UM3 -- UM4 ---- UM5 ---- UM6 ---- UM7 ---- UM8 <- master
   |        |             |        |        |        |        |
  Lllvm1   Llld1         Lclang1  Lclang2  Lllvm2   Llld2     Lmyproj1

The vertical bars represent submodule updates to a particular local
commit in the project mirror.  ``UM3`` in this case is a commit of
some local umbrella repository state that is not a submodule update,
perhaps a ``README`` or project build script update.  Commit ``UM8``
updates a submodule of local project ``myproj``.

The tool ``zip-downstream-fork.py`` at
https://github.com/greened/llvm-git-migration/tree/zip can be used to
convert the umbrella history into a monorepo-based history with
commits in the order implied by submodule updates::

  U1 - U2 - U3 <- upstream/master
   \    \    \
    \    -----\---------------                                    local/zip--.
     \         \              \                                               |
    - Lllvm1 - Llld1 - UM3 -  Lclang1 - Lclang2 - Lllvm2 - Llld2 - Lmyproj1 <-'


The ``U*`` commits represent upstream commits to the monorepo master
branch.  Each submodule update in the local ``UM*`` commits brought in
a subproject tree at some local commit.  The trees in the ``L*1``
commits represent merges from upstream.  These result in edges from
the ``U*`` commits to their corresponding rewritten ``L*1`` commits.
The ``L*2`` commits did not do any merges from upstream.

Note that the merge from ``U2`` to ``Lclang1`` appears redundant, but
if, say, ``U3`` changed some files in upstream clang, the ``Lclang1``
commit appearing after the ``Llld1`` commit would actually represent a
clang tree *earlier* in the upstream clang history.  We want the
``local/zip`` branch to accurately represent the state of our umbrella
history and so the edge ``U2 -> Lclang1`` is a visual reminder of what
clang's tree actually looks like in ``Lclang1``.

Even so, the edge ``U3 -> Llld1`` could be problematic for future
merges from upstream.  git will think that we've already merged from
``U3``, and we have, except for the state of the clang tree.  One
possible migitation strategy is to manually diff clang between ``U2``
and ``U3`` and apply those updates to ``local/zip``.  Another,
possibly simpler strategy is to freeze local work on downstream
branches and merge all submodules from the latest upstream before
running ``zip-downstream-fork.py``.  If downstream merged each project
from upstream in lockstep without any intervening local commits, then
things should be fine without any special action.  We anticipate this
to be the common case.

The tree for ``Lclang1`` outside of clang will represent the state of
things at ``U3`` since all of the upstream projects not participating
in the umbrella history should be in a state respecting the commit
``U3``.  The trees for llvm and lld should correctly represent commits
``Lllvm1`` and ``Llld1``, respectively.

Commit ``UM3`` changed files not related to submodules and we need
somewhere to put them.  It is not safe in general to put them in the
monorepo root directory because they may conflict with files in the
monorepo.  Let's assume we want them in a directory ``local`` in the
monorepo.

**Example 1: Umbrella looks like the monorepo**

For this example, we'll assume that each subproject appears in its own
top-level directory in the umbrella, just as they do in the monorepo .
Let's also assume that we want the files in directory ``myproj`` to
appear in ``local/myproj``.

Given the above run of ``migrate-downstream-fork.py``, a recipe to
create the zipped history is below::

  # Import any non-LLVM repositories the umbrella references.
  git -C my-monorepo remote add localrepo \
                                https://my.local.mirror.org/localrepo.git
  git fetch localrepo

  subprojects=( clang clang-tools-extra compiler-rt debuginfo-tests libclc
                libcxx libcxxabi libunwind lld lldb llgo llvm openmp
                parallel-libs polly pstl )

  # Import histories for upstream split projects (this was probably
  # already done for the ``migrate-downstream-fork.py`` run).
  for project in ${subprojects[@]}; do
    git remote add upstream/split/${project} \
                   https://github.com/llvm-mirror/${subproject}.git
    git fetch umbrella/split/${project}
  done

  # Import histories for downstream split projects (this was probably
  # already done for the ``migrate-downstream-fork.py`` run).
  for project in ${subprojects[@]}; do
    git remote add local/split/${project} \
                   https://my.local.mirror.org/${subproject}.git
    git fetch local/split/${project}
  done

  # Import umbrella history.
  git -C my-monorepo remote add umbrella \
                                https://my.local.mirror.org/umbrella.git
  git fetch umbrella

  # Put myproj in local/myproj
  echo "myproj local/myproj" > my-monorepo/submodule-map.txt

  # Rewrite history
  (
    cd my-monorepo
    zip-downstream-fork.py \
      refs/remotes/umbrella \
      --new-repo-prefix=refs/remotes/upstream/monorepo \
      --old-repo-prefix=refs/remotes/upstream/split \
      --revmap-in=monorepo-map.txt \
      --revmap-out=zip-map.txt \
      --subdir=local \
      --submodule-map=submodule-map.txt \
      --update-tags
   )

   # Create the zip branch (assuming umbrella master is wanted).
   git -C my-monorepo branch --no-track local/zip/master refs/remotes/umbrella/master

Note that if the umbrella has submodules to non-LLVM repositories,
``zip-downstream-fork.py`` needs to know about them to be able to
rewrite commits.  That is why the first step above is to fetch commits
from such repositories.

With ``--update-tags`` the tool will migrate annotated tags pointing
to submodule commits that were inlined into the zipped history.  If
the umbrella pulled in an upstream commit that happened to have a tag
pointing to it, that tag will be migrated, which is almost certainly
not what is wanted.  The tag can always be moved back to its original
commit after rewriting, or the ``--update-tags`` option may be
discarded and any local tags would then be migrated manually.

**Example 2: Nested sources layout**

The tool handles nested submodules (e.g. llvm is a submodule in
umbrella and clang is a submodule in llvm).  The file
``submodule-map.txt`` is a list of pairs, one per line.  The first
pair item describes the path to a submodule in the umbrella
repository.  The second pair item secribes the path where trees for
that submodule should be written in the zipped history.  

Let's say your umbrella repository is actually the llvm repository and
it has submodules in the "nested sources" layout (clang in
tools/clang, etc.).  Let's also say ``projects/myproj`` is a submodule
pointing to some downstream repository.  The submodule map file should
look like this (we still want myproj mapped the same way as
previously)::

  tools/clang clang
  tools/clang/tools/extra clang-tools-extra
  projects/compiler-rt compiler-rt
  projects/debuginfo-tests debuginfo-tests
  projects/libclc libclc
  projects/libcxx libcxx
  projects/libcxxabi libcxxabi
  projects/libunwind libunwind
  tools/lld lld
  tools/lldb lldb
  projects/openmp openmp
  tools/polly polly
  projects/myproj local/myproj

If a submodule path does not appear in the map, the tools assumes it
should be placed in the same place in the monorepo.  That means if you
use the "nested sources" layout in your umrella, you *must* provide
map entries for all of the projects in your umbrella (except llvm).
Otherwise trees from submodule updates will appear underneath llvm in
the zippped history.

Because llvm is itself the umbrella, we use --subdir to write its
content into ``llvm`` in the zippped history::

  # Import any non-LLVM repositories the umbrella references.
  git -C my-monorepo remote add localrepo \
                                https://my.local.mirror.org/localrepo.git
  git fetch localrepo

  subprojects=( clang clang-tools-extra compiler-rt debuginfo-tests libclc
                libcxx libcxxabi libunwind lld lldb llgo llvm openmp
                parallel-libs polly pstl )

  # Import histories for upstream split projects (this was probably
  # already done for the ``migrate-downstream-fork.py`` run).
  for project in ${subprojects[@]}; do
    git remote add upstream/split/${project} \
                   https://github.com/llvm-mirror/${subproject}.git
    git fetch umbrella/split/${project}
  done

  # Import histories for downstream split projects (this was probably
  # already done for the ``migrate-downstream-fork.py`` run).
  for project in ${subprojects[@]}; do
    git remote add local/split/${project} \
                   https://my.local.mirror.org/${subproject}.git
    git fetch local/split/${project}
  done

  # Import umbrella history.  We want this under a different refspec
  # so zip-downstream-fork.py knows what it is.
  git -C my-monorepo remote add umbrella \
                                 https://my.local.mirror.org/llvm.git
  git fetch umbrella

  # Create the submodule map.
  echo "tools/clang clang" > my-monorepo/submodule-map.txt
  echo "tools/clang/tools/extra clang-tools-extra" >> my-monorepo/submodule-map.txt
  echo "projects/compiler-rt compiler-rt" >> my-monorepo/submodule-map.txt
  echo "projects/debuginfo-tests debuginfo-tests" >> my-monorepo/submodule-map.txt
  echo "projects/libclc libclc" >> my-monorepo/submodule-map.txt
  echo "projects/libcxx libcxx" >> my-monorepo/submodule-map.txt
  echo "projects/libcxxabi libcxxabi" >> my-monorepo/submodule-map.txt
  echo "projects/libunwind libunwind" >> my-monorepo/submodule-map.txt
  echo "tools/lld lld" >> my-monorepo/submodule-map.txt
  echo "tools/lldb lldb" >> my-monorepo/submodule-map.txt
  echo "projects/openmp openmp" >> my-monorepo/submodule-map.txt
  echo "tools/polly polly" >> my-monorepo/submodule-map.txt
  echo "projects/myproj local/myproj" >> my-monorepo/submodule-map.txt

  # Rewrite history
  (
    cd my-monorepo
    zip-downstream-fork.py \
      refs/remotes/umbrella \
      --new-repo-prefix=refs/remotes/upstream/monorepo \
      --old-repo-prefix=refs/remotes/upstream/split \
      --revmap-in=monorepo-map.txt \
      --revmap-out=zip-map.txt \
      --subdir=llvm \
      --submodule-map=submodule-map.txt \
      --update-tags
   )

   # Create the zip branch (assuming umbrella master is wanted).
   git -C my-monorepo branch --no-track local/zip/master refs/remotes/umbrella/master


Comments at the top of ``zip-downstream-fork.py`` describe in more
detail how the tool works and various implications of its operation.

Importing local repositories
----------------------------

You may have additional repositories that integrate with the LLVM
ecosystem, essentially extending it with new tools.  If such
repositories are tightly coupled with LLVM, it may make sense to
import them into your local mirror of the monorepo.

If such repositores participated in the umbrella repository used
during the zipping process above, they will automatically be added to
the monorepo.  For downstream repositories that don't participate in
an umbrella setup, the ``import-downstream-repo.py`` tool at
https://github.com/greened/llvm-git-migration/tree/import can help with
getting them into the monorepo.  A recipe follows::

  # Import downstream repo history into the monorepo.
  git -C my-monorepo remote add myrepo https://my.local.mirror.org/myrepo.git
  git fetch myrepo

  my_local_tags=( refs/tags/release
                  refs/tags/hotfix )

  (
    cd my-monorepo
    import-downstream-repo.py \
      refs/remotes/myrepo \
      ${my_local_tags[@]} \
      --new-repo-prefix=refs/remotes/upstream/monorepo \
      --subdir=myrepo \
      --tag-prefix="myrepo-"
   )

   # Preserve release braches.
   for ref in $(git -C my-monorepo for-each-ref --format="%(refname)" \
                  refs/remotes/myrepo/release); do
     branch=${ref#refs/remotes/myrepo/}
     git -C my-monorepo branch --no-track myrepo/${branch} ${ref}
   done

   # Preserve master.
   git -C my-monorepo branch --no-track myrepo/master refs/remotes/myrepo/master

   # Merge master.
   git -C my-monorepo checkout local/zip/master  # Or local/octopus/master
   git -C my-monorepo merge myrepo/master

You may want to merge other corresponding branches, for example
``myrepo`` release branches if they were in lockstep with LLVM project
releases.

``--tag-prefix`` tells ``import-downstream-repo.py`` to rename
annotated tags with the given prefix.  Due to limitations with
``fast_filter_branch.py``, unannotated tags cannot be renamed
(``fast_filter_branch.py`` considers them branches, not tags).  Since
the upstream monorepo had its tags rewritten with an "llvmorg-"
prefix, name conflicts should not be an issue.  ``--tag-prefix`` can
be used to more clearly indicate which tags correspond to various
imported repositories.

Given this repository history::

  R1 - R2 - R3 <- master
       ^
       |
    release/1

The above recipe results in a history like this::

  U1 - U2 - U3 <- upstream/master
   \    \    \
    \    -----\---------------                                         local/zip--.
     \         \              \                                                    |
    - Lllvm1 - Llld1 - UM3 -  Lclang1 - Lclang2 - Lllvm2 - Llld2 - Lmyproj1 - M1 <-'
                                                                             /
                                                                 R1 - R2 - R3  <-.
                                                                      ^           |
                                                                      |           |
                                                               myrepo-release/1   |
                                                                                  |
                                                                   myrepo/master--'

Commits ``R1``, ``R2`` and ``R3`` have trees that *only* contain blobs
from ``myrepo``.  If you require commits from ``myrepo`` to be
interleaved with commits on local project branches (for example,
interleaved with ``llvm1``, ``llvm2``, etc. above) and myrepo doesn't
appear in an umbrella repository, a new tool will need to be
developed.  Creating such a tool would involve:

1. Modifying ``fast_filter_branch.py`` to optionally take a
   revlist directly rather than generating it itself

2. Creating a tool to generate an interleaved ordering of local
   commits based on some criteria (``zip-downstream-fork.py`` uses the
   umbrella history as its criterion)

3. Generating such an ordering and feeding it to
   ``fast_filter_branch.py`` as a revlist

Some care will also likely need to be taken to handle merge commits,
to ensure the parents of such commits migrate correctly.

Scrubbing the Local Monorepo
----------------------------

Once all of the migrating, zipping and importing is done, it's time to
clean up.  The python tools use ``git-fast-import`` which leaves a lot
of cruft around and we want to shrink our new monorepo mirror as much
as possible.  Here is one way to do it::

  git -C my-monorepo checkout master

  # Delete branches we no longer need.  Do this for any other branches
  # you merged above.
  git -C my-monorepo branch -D local/zip/master || true
  git -C my-monorepo branch -D local/octopus/master || true

  # Remove remotes.
  git -C my-monorepo remote remove upstream/monorepo

  for p in ${my_projects[@]}; do
    git -C my-monorepo remote remove upstream/split/${p}
    git -C my-monorepo remote remove local/split/${p}
  done

  git -C my-monorepo remote remove localrepo
  git -C my-monorepo remote remove umbrella
  git -C my-monorepo remote remove myrepo

  # Add anything else here you don't need.  refs/tags/release is
  # listed below assuming tags have been rewritten with a local prefix.
  # If not, remove it from this list.
  refs_to_clean=(
    refs/original
    refs/remotes
    refs/tags/backups
    refs/tags/release
  )

  git -C my-monorepo for-each-ref --format="%(refname)" ${refs_to_clean[@]} |
    xargs -n1 --no-run-if-empty git -C my-monorepo update-ref -d

  git -C my-monorepo reflog expire --all --expire=now

  # fast_filter_branch.py might have gc running in the background.
  while ! git -C my-monorepo \
    -c gc.reflogExpire=0 \
    -c gc.reflogExpireUnreachable=0 \
    -c gc.rerereresolved=0 \
    -c gc.rerereunresolved=0 \
    -c gc.pruneExpire=now \
    gc --prune=now; do
    continue
  done

  # Takes a LOOOONG time!
  git -C my-monorepo repack -A -d -f --depth=250 --window=250

  git -C my-monorepo prune-packed
  git -C my-monorepo prune

You should now have a trim monorepo.  Upload it to your git server and
happy hacking!

References
==========

.. [LattnerRevNum] Chris Lattner, http://lists.llvm.org/pipermail/llvm-dev/2011-July/041739.html
.. [TrickRevNum] Andrew Trick, http://lists.llvm.org/pipermail/llvm-dev/2011-July/041721.html
.. [JSonnRevNum] Joerg Sonnenberg, http://lists.llvm.org/pipermail/llvm-dev/2011-July/041688.html
.. [MatthewsRevNum] Chris Matthews, http://lists.llvm.org/pipermail/cfe-dev/2016-July/049886.html
.. [submodules] Git submodules, https://git-scm.com/book/en/v2/Git-Tools-Submodules)
.. [statuschecks] GitHub status-checks, https://help.github.com/articles/about-required-status-checks/
.. [LebarCHERI] Port *CHERI* to a single repository rewriting history, http://lists.llvm.org/pipermail/llvm-dev/2016-July/102787.html
.. [AminiCHERI] Port *CHERI* to a single repository preserving history, http://lists.llvm.org/pipermail/llvm-dev/2016-July/102804.html
