=================================
How To Release LLVM To The Public
=================================

.. contents::
   :local:
   :depth: 1

Introduction
============

This document contains information about successfully releasing LLVM ---
including subprojects: e.g., ``clang`` and ``dragonegg`` --- to the public.  It
is the Release Manager's responsibility to ensure that a high quality build of
LLVM is released.

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

* Generate and send out the second release candidate sources.  Only *critial*
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

#. Create the release branch for ``llvm``, ``clang``, the ``test-suite``, and
   ``dragonegg`` from the last known good revision.  The branch's name is
   ``release_XY``, where ``X`` is the major and ``Y`` the minor release
   numbers.  The branches should be created using the following commands:

   ::

     $ svn copy https://llvm.org/svn/llvm-project/llvm/trunk \
                https://llvm.org/svn/llvm-project/llvm/branches/release_XY

     $ svn copy https://llvm.org/svn/llvm-project/cfe/trunk \
                https://llvm.org/svn/llvm-project/cfe/branches/release_XY

     $ svn copy https://llvm.org/svn/llvm-project/dragonegg/trunk \
                https://llvm.org/svn/llvm-project/dragonegg/branches/release_XY

     $ svn copy https://llvm.org/svn/llvm-project/test-suite/trunk \
                https://llvm.org/svn/llvm-project/test-suite/branches/release_XY

#. Advise developers that they may now check their patches into the Subversion
   tree again.

#. The Release Manager should switch to the release branch, because all changes
   to the release will now be done in the branch.  The easiest way to do this is
   to grab a working copy using the following commands:

   ::

     $ svn co https://llvm.org/svn/llvm-project/llvm/branches/release_XY llvm-X.Y

     $ svn co https://llvm.org/svn/llvm-project/cfe/branches/release_XY clang-X.Y

     $ svn co https://llvm.org/svn/llvm-project/dragonegg/branches/release_XY dragonegg-X.Y

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

Building the Release
--------------------

The builds of ``llvm``, ``clang``, and ``dragonegg`` *must* be free of
errors and warnings in Debug, Release+Asserts, and Release builds.  If all
builds are clean, then the release passes Build Qualification.

The ``make`` options for building the different modes:

+-----------------+---------------------------------------------+
| Mode            | Options                                     |
+=================+=============================================+
| Debug           | ``ENABLE_OPTIMIZED=0``                      |
+-----------------+---------------------------------------------+
| Release+Asserts | ``ENABLE_OPTIMIZED=1``                      |
+-----------------+---------------------------------------------+
| Release         | ``ENABLE_OPTIMIZED=1 DISABLE_ASSERTIONS=1`` |
+-----------------+---------------------------------------------+

Build LLVM
^^^^^^^^^^

Build ``Debug``, ``Release+Asserts``, and ``Release`` versions
of ``llvm`` on all supported platforms.  Directions to build ``llvm``
are :doc:`here <GettingStarted>`.

Build Clang Binary Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Creating the ``clang`` binary distribution (Debug/Release+Asserts/Release)
requires performing the following steps for each supported platform:

#. Build clang according to the directions `here
   <http://clang.llvm.org/get_started.html>`__.

#. Build both a Debug and Release version of clang.  The binary will be the
   Release build.

#. Package ``clang`` (details to follow).

Target Specific Build Details
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The table below specifies which compilers are used for each Arch/OS combination
when qualifying the build of ``llvm``, ``clang``, and ``dragonegg``.

+--------------+---------------+----------------------+
| Architecture | OS            | compiler             |
+==============+===============+======================+
| x86-32       | Mac OS 10.5   | gcc 4.0.1            |
+--------------+---------------+----------------------+
| x86-32       | Linux         | gcc 4.2.X, gcc 4.3.X |
+--------------+---------------+----------------------+
| x86-32       | FreeBSD       | gcc 4.2.X            |
+--------------+---------------+----------------------+
| x86-32       | mingw         | gcc 3.4.5            |
+--------------+---------------+----------------------+
| x86-64       | Mac OS 10.5   | gcc 4.0.1            |
+--------------+---------------+----------------------+
| x86-64       | Linux         | gcc 4.2.X, gcc 4.3.X |
+--------------+---------------+----------------------+
| x86-64       | FreeBSD       | gcc 4.2.X            |
+--------------+---------------+----------------------+
| ARMv7        | Linux         | gcc 4.6.X, gcc 4.7.X |
+--------------+---------------+----------------------+

Release Qualification Criteria
------------------------------

A release is qualified when it has no regressions from the previous release (or
baseline).  Regressions are related to correctness first and performance second.
(We may tolerate some minor performance regressions if they are deemed
necessary for the general quality of the compiler.)

**Regressions are new failures in the set of tests that are used to qualify
each product and only include things on the list.  Every release will have
some bugs in it.  It is the reality of developing a complex piece of
software.  We need a very concrete and definitive release criteria that
ensures we have monotonically improving quality on some metric.  The metric we
use is described below.  This doesn't mean that we don't care about other
criteria, but these are the criteria which we found to be most important and
which must be satisfied before a release can go out.**

Qualify LLVM
^^^^^^^^^^^^

LLVM is qualified when it has a clean test run without a front-end.  And it has
no regressions when using either ``clang`` or ``dragonegg`` with the
``test-suite`` from the previous release.

Qualify Clang
^^^^^^^^^^^^^

``Clang`` is qualified when front-end specific tests in the ``llvm`` regression
test suite all pass, clang's own test suite passes cleanly, and there are no
regressions in the ``test-suite``.

Specific Target Qualification Details
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+--------------+-------------+----------------+-----------------------------+
| Architecture | OS          | clang baseline | tests                       |
+==============+=============+================+=============================+
| x86-32       | Linux       | last release   | llvm regression tests,      |
|              |             |                | clang regression tests,     |
|              |             |                | test-suite (including spec) |
+--------------+-------------+----------------+-----------------------------+
| x86-32       | FreeBSD     | last release   | llvm regression tests,      |
|              |             |                | clang regression tests,     |
|              |             |                | test-suite                  |
+--------------+-------------+----------------+-----------------------------+
| x86-32       | mingw       | none           | QT                          |
+--------------+-------------+----------------+-----------------------------+
| x86-64       | Mac OS 10.X | last release   | llvm regression tests,      |
|              |             |                | clang regression tests,     |
|              |             |                | test-suite (including spec) |
+--------------+-------------+----------------+-----------------------------+
| x86-64       | Linux       | last release   | llvm regression tests,      |
|              |             |                | clang regression tests,     |
|              |             |                | test-suite (including spec) |
+--------------+-------------+----------------+-----------------------------+
| x86-64       | FreeBSD     | last release   | llvm regression tests,      |
|              |             |                | clang regression tests,     |
|              |             |                | test-suite                  |
+--------------+-------------+----------------+-----------------------------+
| ARMv7A       | Linux       | last release   | llvm regression tests,      |
|              |             |                | clang regression tests,     |
|              |             |                | test-suite                  |
+--------------+-------------+----------------+-----------------------------+

Community Testing
-----------------

Once all testing has been completed and appropriate bugs filed, the release
candidate tarballs are put on the website and the LLVM community is notified.
Ask that all LLVM developers test the release in 2 ways:

#. Download ``llvm-X.Y``, ``llvm-test-X.Y``, and the appropriate ``clang``
   binary.  Build LLVM.  Run ``make check`` and the full LLVM test suite (``make
   TEST=nightly report``).

#. Download ``llvm-X.Y``, ``llvm-test-X.Y``, and the ``clang`` sources.  Compile
   everything.  Run ``make check`` and the full LLVM test suite (``make
   TEST=nightly report``).

Ask LLVM developers to submit the test suite report and ``make check`` results
to the list.  Verify that there are no regressions from the previous release.
The results are not used to qualify a release, but to spot other potential
problems.  For unsupported targets, verify that ``make check`` is at least
clean.

During the first round of testing, all regressions must be fixed before the
second release candidate is tagged.

If this is the second round of testing, the testing is only to ensure that bug
fixes previously merged in have not created new major problems. *This is not
the time to solve additional and unrelated bugs!* If no patches are merged in,
the release is determined to be ready and the release manager may move onto the
next stage.

Release Patch Rules
-------------------

Below are the rules regarding patching the release branch:

#. Patches applied to the release branch may only be applied by the release
   manager.

#. During the first round of testing, patches that fix regressions or that are
   small and relatively risk free (verified by the appropriate code owner) are
   applied to the branch.  Code owners are asked to be very conservative in
   approving patches for the branch.  We reserve the right to reject any patch
   that does not fix a regression as previously defined.

#. During the remaining rounds of testing, only patches that fix critical
   regressions may be applied.

#. For dot releases all patches must mantain both API and ABI compatibility with
   the previous major release.  Only bugfixes will be accepted.

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

#. Create a new subdirectory ``X.Y`` in the releases directory.

#. Commit the ``llvm``, ``test-suite``, ``clang`` source, ``clang binaries``,
   ``dragonegg`` source, and ``dragonegg`` binaries in this new directory.

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

Have Chris send out the release announcement when everything is finished.

