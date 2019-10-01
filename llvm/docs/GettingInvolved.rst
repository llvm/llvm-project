Getting Involved
================

LLVM welcomes contributions of all kinds. To get started, please review the following topics:

.. contents::
   :local:

.. toctree::
   :hidden:

   Contributing
   DeveloperPolicy
   SphinxQuickstartTemplate
   Phabricator
   HowToSubmitABug
   BugLifeCycle
   CodingStandards

:doc:`Contributing`
   An overview on how to contribute to LLVM.

:doc:`DeveloperPolicy`
   The LLVM project's policy towards developers and their contributions.

:doc:`SphinxQuickstartTemplate`
  A template + tutorial for writing new Sphinx documentation. It is meant
  to be read in source form.

:doc:`Phabricator`
   Describes how to use the Phabricator code review tool hosted on
   http://reviews.llvm.org/ and its command line interface, Arcanist.

:doc:`HowToSubmitABug`
   Instructions for properly submitting information about any bugs you run into
   in the LLVM system.

:doc:`BugLifeCycle`
   Describes how bugs are reported, triaged and closed.

:doc:`CodingStandards`
  Details the LLVM coding standards and provides useful information on writing
  efficient C++ code.

.. _development-process:

Development Process
-------------------

Information about LLVM's development process.

.. toctree::
   :hidden:

   Projects
   LLVMBuild
   HowToReleaseLLVM
   Packaging
   ReleaseProcess
   HowToAddABuilder
   ReleaseNotes

:doc:`Projects`
  How-to guide and templates for new projects that *use* the LLVM
  infrastructure.  The templates (directory organization, Makefiles, and test
  tree) allow the project code to be located outside (or inside) the ``llvm/``
  tree, while using LLVM header files and libraries.

:doc:`LLVMBuild`
  Describes the LLVMBuild organization and files used by LLVM to specify
  component descriptions.

:doc:`HowToReleaseLLVM`
  This is a guide to preparing LLVM releases. Most developers can ignore it.

:doc:`ReleaseProcess`
  This is a guide to validate a new release, during the release process. Most developers can ignore it.

:doc:`HowToAddABuilder`
   Instructions for adding new builder to LLVM buildbot master.

:doc:`Packaging`
   Advice on packaging LLVM into a distribution.

:doc:`Release notes for the current release <ReleaseNotes>`
   This describes new features, known bugs, and other limitations.

.. _mailing-lists:

Mailing Lists
-------------

If you can't find what you need in these docs, try consulting the mailing
lists.

`Developer's List (llvm-dev)`__
  This list is for people who want to be included in technical discussions of
  LLVM. People post to this list when they have questions about writing code
  for or using the LLVM tools. It is relatively low volume.

  .. __: http://lists.llvm.org/mailman/listinfo/llvm-dev

`Commits Archive (llvm-commits)`__
  This list contains all commit messages that are made when LLVM developers
  commit code changes to the repository. It also serves as a forum for
  patch review (i.e. send patches here). It is useful for those who want to
  stay on the bleeding edge of LLVM development. This list is very high
  volume.

  .. __: http://lists.llvm.org/pipermail/llvm-commits/

`Bugs & Patches Archive (llvm-bugs)`__
  This list gets emailed every time a bug is opened and closed. It is
  higher volume than the LLVM-dev list.

  .. __: http://lists.llvm.org/pipermail/llvm-bugs/

`Test Results Archive (llvm-testresults)`__
  A message is automatically sent to this list by every active nightly tester
  when it completes.  As such, this list gets email several times each day,
  making it a high volume list.

  .. __: http://lists.llvm.org/pipermail/llvm-testresults/

`LLVM Announcements List (llvm-announce)`__
  This is a low volume list that provides important announcements regarding
  LLVM.  It gets email about once a month.

  .. __: http://lists.llvm.org/mailman/listinfo/llvm-announce

IRC
---

Users and developers of the LLVM project (including subprojects such as Clang)
can be found in #llvm on `irc.oftc.net <irc://irc.oftc.net/llvm>`_.

This channel has several bots.

* Buildbot reporters

  * llvmbb - Bot for the main LLVM buildbot master.
    http://lab.llvm.org:8011/console
  * smooshlab - Apple's internal buildbot master.

* robot - Bugzilla linker. %bug <number>

* clang-bot - A `geordi <http://www.eelis.net/geordi/>`_ instance running
  near-trunk clang instead of gcc.

.. _meetups-social-events:

Meetups and social events
-------------------------

.. toctree::
   :hidden:

   MeetupGuidelines

Besides developer `meetings and conferences <https://llvm.org/devmtg/>`_,
there are several user groups called
`LLVM Socials <https://www.meetup.com/pro/llvm/>`_. We greatly encourage you to
join one in your city. Or start a new one if there is none:

:doc:`MeetupGuidelines`

.. _community-proposals:

Community wide proposals
------------------------

Proposals for massive changes in how the community behaves and how the work flow
can be better.

.. toctree::
   :hidden:

   CodeOfConduct
   Proposals/GitHubMove
   BugpointRedesign
   Proposals/LLVMLibC
   Proposals/TestSuite
   Proposals/VariableNames
   Proposals/VectorizationPlan

:doc:`CodeOfConduct`
   Proposal to adopt a code of conduct on the LLVM social spaces (lists, events,
   IRC, etc).

:doc:`Proposals/GitHubMove`
   Proposal to move from SVN/Git to GitHub.

:doc:`BugpointRedesign`
   Design doc for a redesign of the Bugpoint tool.

:doc:`Proposals/LLVMLibC`
   Proposal to add a libc implementation under the LLVM project.

:doc:`Proposals/TestSuite`
   Proposals for additional benchmarks/programs for llvm's test-suite.

:doc:`Proposals/VariableNames`
   Proposal to change the variable names coding standard.

:doc:`Proposals/VectorizationPlan`
   Proposal to model the process and upgrade the infrastructure of LLVM's Loop Vectorizer.