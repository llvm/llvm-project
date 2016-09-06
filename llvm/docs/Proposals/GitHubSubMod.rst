===============================================
Moving LLVM Projects to GitHub with Sub-Modules
===============================================

Introduction
============

This is a proposal to move our current revision control system from our own
hosted Subversion to GitHub. Below are the financial and technical arguments as
to why we need such a move and how will people (and validation infrastructure)
continue to work with a Git-based LLVM.

There will be a survey pointing at this document when we'll know the community's
reaction and, if we collectively decide to move, the time-frames. Be sure to make
your views count.

Essentially, the proposal is divided in the following parts:

* Outline of the reasons to move to Git and GitHub
* Description on what the work flow will look like (compared to SVN)
* Remaining issues and potential problems
* The proposed migration plan

Why Git, and Why GitHub?
========================

Why move at all?
----------------

The strongest reason for the move, and why this discussion started in the first
place, is that we currently host our own Subversion server and Git mirror in a
voluntary basis. The LLVM Foundation sponsors the server and provides limited
support, but there is only so much it can do.

The volunteers are not Sysadmins themselves, but compiler engineers that happen
to know a thing or two about hosting servers. We also don't have 24/7 support,
and we sometimes wake up to see that continuous integration is broken because
the SVN server is either down or unresponsive.

With time and money, the foundation and volunteers could improve our services,
implement more functionality and provide around the clock support, so that we
can have a first class infrastructure with which to work. But the cost is not
small, both in money and time invested.

On the other hand, there are multiple services out there (GitHub, GitLab,
BitBucket among others) that offer that same service (24/7 stability, disk space,
Git server, code browsing, forking facilities, etc) for the very affordable price
of *free*.

Why Git?
--------

Most new coders nowadays start with Git. A lot of them have never used SVN, CVS
or anything else. Websites like GitHub have changed the landscape of open source
contributions, reducing the cost of first contribution and fostering
collaboration.

Git is also the version control most LLVM developers use. Despite the sources
being stored in an SVN server, most people develop using the Git-SVN integration,
and that shows that Git is not only more powerful than SVN, but people have
resorted to using a bridge because its features are now indispensable to their
internal and external workflows.

In essence, Git allows you to:

* Commit, squash, merge, fork locally without any penalty to the server
* Add as many branches as necessary to allow for multiple threads of development
* Collaborate with peers directly, even without access to the Internet
* Have multiple trees without multiplying disk space.

In addition, because Git seems to be replacing every project's version control
system, there are many more tools that can use Git's enhanced feature set, so
new tooling is much more likely to support Git first (if not only), than any
other version control system.

Why GitHub?
-----------

GitHub, like GitLab and BitBucket, provide free code hosting for open source
projects. Essentially, they will completely replace *all* the infrastructure that
we have today that serves code repository, mirroring, user control, etc.

They also have a dedicated team to monitor, migrate, improve and distribute the
contents of the repositories depending on region and load. A level of quality
that we'd never have without spending money that would be better spent elsewhere,
for example development meetings, sponsoring disadvantaged people to work on
compilers and foster diversity and equality in our community.

GitHub has the added benefit that we already have a presence there. Many
developers use it already, and the mirror from our current repository is already
set up.

Furthermore, GitHub has an *SVN view* (https://github.com/blog/626-announcing-svn-support)
where people that still have/want to use SVN infrastructure and tooling can
slowly migrate or even stay working as if it was an SVN repository (including
read-write access).

So, any of the three solutions solve the cost and maintenance problem, but GitHub
has two additional features that would be beneficial to the migration plan as
well as the community already settled there.


What will the new workflow look like
====================================

In order to move version control, we need to make sure that we get all the
benefits with the least amount of problems. That's why the migration plan will
be slow, one step at a time, and we'll try to make it look as close as possible
to the current style without impacting the new features we want.

Each LLVM project will continue to be hosted as separate GitHub repository
under a single GitHub organisation. Users can continue to choose to use either
SVN or Git to access the repositories to suit their current workflow.

In addition, we'll create a repository that will mimic our current *linear
history* repository. The most accepted proposal, then, was to have an umbrella
project that will contain *sub-modules* (https://git-scm.com/book/en/v2/Git-Tools-Submodules)
of all the LLVM projects and nothing else.

This repository can be checked out on its own, in order to have *all* LLVM
projects in a single check-out, as many people have suggested, but it can also
only hold the references to the other projects, and be used for the sole purpose
of understanding the *sequence* in which commits were added by using the
``git rev-list --count hash`` or ``git describe hash`` commands.

One example of such a repository is Takumi's llvm-project-submodule
(https://github.com/chapuni/llvm-project-submodule), which when checked out,
will have the references to all sub-modules but not check them out, so one will
need to *init* the module manually. This will allow the *exact* same behaviour
as checking out individual SVN repositories, as it will keep the correct linear
history.

There is no need to additional tags, flags and properties, or external
services controlling the history, since both SVN and *git rev-list* can already
do that on their own.

We will need additional server hooks to avoid non-fast-forwards commits (ex.
merges, forced pushes, etc) in order to keep the linearity of the history.

The three types hooks to be implemented are:

* Status Checks: By placing status checks on a protected branch, we can guarantee
  that the history is kept linear and sane at all times, on all repositories.
  See: https://help.github.com/articles/about-required-status-checks/
* Umbrella updates: By using GitHub web hooks, we can update a small web-service
  inside LLVM's own infrastructure to update the umbrella project remotely. The
  maintenance of this service will be lower than the current SVN maintenance and
  the scope of its failures will be less severe.
  See: https://developer.github.com/webhooks/
* Commits email update: By adding an email web hook, we can make every push show
  in the lists, allowing us to retain history and do post-commit reviews.
  See: https://help.github.com/articles/managing-notifications-for-pushes-to-a-repository/

Access will be transferred one-to-one to GitHub accounts for everyone that already
has commit access to our current repository. Those who don't have accounts will
have to create one in order to continue contributing to the project. In the
future, people only need to provide their GitHub accounts to be granted access.

In a nutshell:

* The projects' repositories will remain identical, with a new address (GitHub).
* They'll continue to have SVN access (Read-Write), but will also gain Git RW access.
* The linear history can still be accessed in the (RO) submodule meta project.
* Individual projects' history will be local (ie. not interlaced with the other
  projects, as the current SVN repos are), and we need the umbrella project
  (using submodules) to have the same view as we had in SVN.

Additionally, each repository will have the following server hooks:

* Pre-commit hooks to stop people from applying non-fast-forward merges
* Webhook to update the umbrella project (via buildbot or web services)
* Email hook to each commits list (llvm-commit, cfe-commit, etc)

Essentially, we're adding Git RW access in addition to the already existing
structure, with all the additional benefits of it being in GitHub.

Example of a working version:

* Repository: https://github.com/llvm-beanz/llvm-submodules
* Update bot: http://beanz-bot.com:8180/jenkins/job/submodule-update/

What will *not* be changed
--------------------------

This is a change of version control system, not the whole infrastructure. There
are plans to replace our current tools (review, bugs, documents), but they're
all orthogonal to this proposal.

We'll also be keeping the buildbots (and migrating them to use Git) as well as
LNT, and any other system that currently provides value upstream.

Any discussion regarding those tools are out of scope in this proposal.

Remaining questions and problems
================================

1. How much the SVN view emulates and how much it'll break tools/CI?

For this one, we'll need people that will have problems in that area to tell
us what's wrong and how to help them fix it.

We also recommend people and companies to migrate to Git, for its many other
additional benefits.

2. Which tools will need changing?

LNT may break, since it relies on SVN's history. We can continue to
use LNT with the SVN-View, but it would be best to move it to Git once and for
all.

The LLVMLab bisect tool will also be affected and will need adjusting. As with
LNT, it should be fine to use GitHub's SVN view, but changing it to work on Git
will be required in the long term.

Phabricator will also need to change its configuration to point at the GitHub
repositories, but since it already works with Git, this will be a trivial change.

Migration Plan
==============

If we decide to move, we'll have to set a date for the process to begin.

As usual, we should be announcing big changes in one release to happen in the
next one. But since this won't impact external users (if they rely on our source
release tarballs), we don't necessarily have to.

We will have to make sure all the *problems* reported are solved before the
final push. But we can start all non-binding processes (like mirroring to GitHub
and testing the SVN interface in it) before any hard decision.

Here's a proposed plan:

STEP #1 : Pre Move

0. Update docs to mention the move, so people are aware the it's going on.
1. Register an official GitHub project with the LLVM foundation.
2. Setup another (read-only) mirror of llvm.org/git at this GitHub project,
   adding all necessary hooks to avoid broken history (merge, dates, pushes), as
   well as a webhook to update the umbrella project (see below).
3. Make sure we have an llvm-project (with submodules) setup in the official
   account, with all necessary hooks (history, update, merges).
4. Make sure bisecting with llvm-project works.
5. Make sure no one has any other blocker.

STEP #2 : Git Move

6. Update the buildbots to pick up updates and commits from the official git
   repository.
7. Update Phabricator to pick up commits from the official git repository.
8. Tell people living downstream to pick up commits from the official git
   repository.
9. Give things time to settle. We could play some games like disabling the SVN
   repository for a few hours on purpose so that people can test that their
   infrastructure has really become independent of the SVN repository.

Until this point nothing has changed for developers, it will just
boil down to a lot of work for buildbot and other infrastructure
owners.

Once all dependencies are cleared, and all problems have been solved:

STEP #3: Write Access Move

10. Collect peoples GitHub account information, adding them to the project.
11. Switch SVN repository to read-only and allow pushes to the GitHub repository.
12. Mirror Git to SVN.

STEP #4 : Post Move

13. Archive the SVN repository, if GitHub's SVN is good enough.
14. Review and update *all* LLVM documentation.
15. Review website links pointing to viewvc/klaus/phab etc. to point to GitHub
    instead.
