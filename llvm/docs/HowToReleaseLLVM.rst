=================================
How To Release LLVM To The Public
=================================

Introduction
============

This document contains information about successfully releasing LLVM ---
including sub-projects: e.g., ``clang`` and ``compiler-rt`` --- to the public.
It is the Release Manager's responsibility to ensure that a high quality build
of LLVM is released.

If you're looking for the document on how to test the release candidates and
create the binary packages, please refer to the :doc:`ReleaseProcess` instead.

.. _timeline:

Release Timeline
================

LLVM is released on a time based schedule --- with major releases roughly
every 6 months.  In between major releases there may be dot releases.
The release manager will determine if and when to make a dot release based
on feedback from the community.  Typically, dot releases should be made if
there are large number of bug-fixes in the stable branch or a critical bug
has been discovered that affects a large number of users.

Unless otherwise stated, dot releases will follow the same procedure as
major releases.

The release process is roughly as follows:

* Set code freeze and branch creation date for 6 months after last code freeze
  date.  Announce release schedule to the LLVM community and update the website.

* Create release branch and begin release process.

* Send out release candidate sources for first round of testing.  Testing lasts
  7-10 days.  During the first round of testing, any regressions found should be
  fixed.  Patches are merged from mainline into the release branch.  Also, all
  features need to be completed during this time.  Any features not completed at
  the end of the first round of testing will be removed or disabled for the
  release.

* Generate and send out the second release candidate sources.  Only *critical*
  bugs found during this testing phase will be fixed.  Any bugs introduced by
  merged patches will be fixed.  If so a third round of testing is needed.

* The release notes are updated.

* Finally, release!

The release process will be accelerated for dot releases.  If the first round
of testing finds no critical bugs and no regressions since the last major release,
then additional rounds of testing will not be required.

Release Process
===============

.. contents::
   :local:

Release Administrative Tasks
----------------------------

This section describes a few administrative tasks that need to be done for the
release process to begin.  Specifically, it involves:

* Creating the release branch,

* Setting version numbers, and

* Tagging release candidates for the release team to begin testing.

Create Release Branch
^^^^^^^^^^^^^^^^^^^^^

Branch the Subversion trunk using the following procedure:

#. Remind developers that the release branching is imminent and to refrain from
   committing patches that might break the build.  E.g., new features, large
   patches for works in progress, an overhaul of the type system, an exciting
   new TableGen feature, etc.

#. Verify that the current Subversion trunk is in decent shape by
   examining nightly tester and buildbot results.

#. Create the release branch for ``llvm``, ``clang``, and other sub-projects,
   from the last known good revision.  The branch's name is
   ``release_XY``, where ``X`` is the major and ``Y`` the minor release
   numbers.  Use ``utils/release/tag.sh`` to tag the release.

#. Advise developers that they may now check their patches into the Subversion
   tree again.

#. The Release Manager should switch to the release branch, because all changes
   to the release will now be done in the branch.  The easiest way to do this is
   to grab a working copy using the following commands:

   ::

     $ svn co https://llvm.org/svn/llvm-project/llvm/branches/release_XY llvm-X.Y

     $ svn co https://llvm.org/svn/llvm-project/cfe/branches/release_XY clang-X.Y

     $ svn co https://llvm.org/svn/llvm-project/test-suite/branches/release_XY test-suite-X.Y

Update LLVM Version
^^^^^^^^^^^^^^^^^^^

After creating the LLVM release branch, update the release branches'
``autoconf`` and ``configure.ac`` versions from '``X.Ysvn``' to '``X.Y``'.
Update it on mainline as well to be the next version ('``X.Y+1svn``').
Regenerate the configure scripts for both ``llvm`` and the ``test-suite``.

In addition, the version numbers of all the Bugzilla components must be updated
for the next release.

Tagging the LLVM Release Candidates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tag release candidates using the tag.sh script in utils/release.

::

  $ ./tag.sh -release X.Y.Z -rc $RC

The Release Manager may supply pre-packaged source tarballs for users.  This can
be done with the export.sh script in utils/release.

::

  $ ./export.sh -release X.Y.Z -rc $RC

This will generate source tarballs for each LLVM project being validated, which
can be uploaded to the website for further testing.

Build Clang Binary Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Creating the ``clang`` binary distribution requires following the instructions
:doc:`here <ReleaseProcess>`.

That process will perform both Release+Asserts and Release builds but only
pack the Release build for upload. You should use the Release+Asserts sysroot,
normally under ``final/Phase3/Release+Asserts/llvmCore-3.8.1-RCn.install/``,
for test-suite and run-time benchmarks, to make sure nothing serious has 
passed through the net. For compile-time benchmarks, use the Release version.

The minimum required version of the tools you'll need are :doc:`here <GettingStarted>`

Release Qualification Criteria
------------------------------

A release is qualified when it has no regressions from the previous release (or
baseline).  Regressions are related to correctness first and performance second.
(We may tolerate some minor performance regressions if they are deemed
necessary for the general quality of the compiler.)

More specifically, Clang/LLVM is qualified when it has a clean test with all
supported sub-projects included (``make check-all``), per target, and it has no
regressions with the ``test-suite`` in relation to the previous release.

Regressions are new failures in the set of tests that are used to qualify
each product and only include things on the list.  Every release will have
some bugs in it.  It is the reality of developing a complex piece of
software.  We need a very concrete and definitive release criteria that
ensures we have monotonically improving quality on some metric.  The metric we
use is described below.  This doesn't mean that we don't care about other
criteria, but these are the criteria which we found to be most important and
which must be satisfied before a release can go out.

Official Testing
----------------

A few developers in the community have dedicated time to validate the release
candidates and volunteered to be the official release testers for each
architecture.

These will be the ones testing, generating and uploading the official binaries
to the server, and will be the minimum tests *necessary* for the release to
proceed.

This will obviously not cover all OSs and distributions, so additional community
validation is important. However, if community input is not reached before the
release is out, all bugs reported will have to go on the next stable release.

The official release managers are:

* Major releases (X.0): Hans Wennborg
* Stable releases (X.n): Tom Stellard

The official release testers are volunteered from the community and have
consistently validated and released binaries for their targets/OSs. To contact
them, you should email the ``release-testers@lists.llvm.org`` mailing list.

The official testers list is in the file ``RELEASE_TESTERS.TXT``, in the ``LLVM``
repository.

Community Testing
-----------------

Once all testing has been completed and appropriate bugs filed, the release
candidate tarballs are put on the website and the LLVM community is notified.

We ask that all LLVM developers test the release in any the following ways:

#. Download ``llvm-X.Y``, ``llvm-test-X.Y``, and the appropriate ``clang``
   binary.  Build LLVM.  Run ``make check`` and the full LLVM test suite (``make
   TEST=nightly report``).

#. Download ``llvm-X.Y``, ``llvm-test-X.Y``, and the ``clang`` sources.  Compile
   everything.  Run ``make check`` and the full LLVM test suite (``make
   TEST=nightly report``).

#. Download ``llvm-X.Y``, ``llvm-test-X.Y``, and the appropriate ``clang``
   binary. Build whole programs with it (ex. Chromium, Firefox, Apache) for
   your platform.

#. Download ``llvm-X.Y``, ``llvm-test-X.Y``, and the appropriate ``clang``
   binary. Build *your* programs with it and check for conformance and
   performance regressions.

#. Run the :doc:`release process <ReleaseProcess>`, if your platform is
   *different* than that which is officially supported, and report back errors
   only if they were not reported by the official release tester for that
   architecture.

We also ask that the OS distribution release managers test their packages with
the first candidate of every release, and report any *new* errors in Bugzilla.
If the bug can be reproduced with an unpatched upstream version of the release
candidate (as opposed to the distribution's own build), the priority should be
release blocker.

During the first round of testing, all regressions must be fixed before the
second release candidate is tagged.

In the subsequent stages, the testing is only to ensure that bug
fixes previously merged in have not created new major problems. *This is not
the time to solve additional and unrelated bugs!* If no patches are merged in,
the release is determined to be ready and the release manager may move onto the
next stage.

Reporting Regressions
---------------------

Every regression that is found during the tests (as per the criteria above),
should be filled in a bug in Bugzilla with the priority *release blocker* and
blocking a specific release.

To help manage all the bugs reported and which ones are blockers or not, a new
"[meta]" bug should be created and all regressions *blocking* that Meta. Once
all blockers are done, the Meta can be closed.

If a bug can't be reproduced, or stops being a blocker, it should be removed
from the Meta and its priority decreased to *normal*. Debugging can continue,
but on trunk.

Merge Requests
--------------

You can use any of the following methods to request that a revision from trunk
be merged into a release branch:

#. Use the ``utils/release/merge-request.sh`` script which will automatically
   file a bug_ requesting that the patch be merged. e.g. To request revision
   12345 be merged into the branch for the 5.0.1 release:
   ``llvm.src/utils/release/merge-request.sh -stable-version 5.0 -r 12345 -user bugzilla@example.com``

#. Manually file a bug_ with the subject: "Merge r12345 into the X.Y branch",
   enter the commit(s) that you want merged in the "Fixed by Commit(s)" and mark
   it as a blocker of the current release bug.  Release bugs are given aliases
   in the form of release-x.y.z, so to mark a bug as a blocker for the 5.0.1
   release, just enter release-5.0.1 in the "Blocks" field.

#. Reply to the commit email on llvm-commits for the revision to merge and cc
   the release manager.

.. _bug: https://bugs.llvm.org/

Release Patch Rules
-------------------

Below are the rules regarding patching the release branch:

#. Patches applied to the release branch may only be applied by the release
   manager, the official release testers or the code owners with approval from
   the release manager.

#. During the first round of testing, patches that fix regressions or that are
   small and relatively risk free (verified by the appropriate code owner) are
   applied to the branch.  Code owners are asked to be very conservative in
   approving patches for the branch.  We reserve the right to reject any patch
   that does not fix a regression as previously defined.

#. During the remaining rounds of testing, only patches that fix critical
   regressions may be applied.

#. For dot releases all patches must maintain both API and ABI compatibility with
   the previous major release.  Only bug-fixes will be accepted.

Merging Patches
^^^^^^^^^^^^^^^

The ``utils/release/merge.sh`` script can be used to merge individual revisions
into any one of the llvm projects. To merge revision ``$N`` into project
``$PROJ``, do:

#. ``svn co https://llvm.org/svn/llvm-project/$PROJ/branches/release_XX
   $PROJ.src``

#. ``$PROJ.src/utils/release/merge.sh --proj $PROJ --rev $N``

#. Run regression tests.

#. ``cd $PROJ.src``. Run the ``svn commit`` command printed out by ``merge.sh``
   in step 2.

Release Final Tasks
-------------------

The final stages of the release process involves tagging the "final" release
branch, updating documentation that refers to the release, and updating the
demo page.

Update Documentation
^^^^^^^^^^^^^^^^^^^^

Review the documentation and ensure that it is up to date.  The "Release Notes"
must be updated to reflect new features, bug fixes, new known issues, and
changes in the list of supported platforms.  The "Getting Started Guide" should
be updated to reflect the new release version number tag available from
Subversion and changes in basic system requirements.  Merge both changes from
mainline into the release branch.

.. _tag:

Tag the LLVM Final Release
^^^^^^^^^^^^^^^^^^^^^^^^^^

Tag the final release sources using the tag.sh script in utils/release.

::

  $ ./tag.sh -release X.Y.Z -final

Update the LLVM Demo Page
-------------------------

The LLVM demo page must be updated to use the new release.  This consists of
using the new ``clang`` binary and building LLVM.

Update the LLVM Website
^^^^^^^^^^^^^^^^^^^^^^^

The website must be updated before the release announcement is sent out.  Here
is what to do:

#. Check out the ``www`` module from Subversion.

#. Create a new sub-directory ``X.Y`` in the releases directory.

#. Commit the ``llvm``, ``test-suite``, ``clang`` source and binaries in this
   new directory.

#. Copy and commit the ``llvm/docs`` and ``LICENSE.txt`` files into this new
   directory.  The docs should be built with ``BUILD_FOR_WEBSITE=1``.

#. Commit the ``index.html`` to the ``release/X.Y`` directory to redirect (use
   from previous release).

#. Update the ``releases/download.html`` file with the new release.

#. Update the ``releases/index.html`` with the new release and link to release
   documentation.

#. Finally, update the main page (``index.html`` and sidebar) to point to the
   new release and release announcement.  Make sure this all gets committed back
   into Subversion.

Announce the Release
^^^^^^^^^^^^^^^^^^^^

Send an email to the list announcing the release, pointing people to all the
relevant documentation, download pages and bugs fixed.

