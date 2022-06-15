Getting Involved
================

LLVM welcomes contributions of all kinds. To get started, please review the following topics:

.. contents::
   :local:

.. toctree::
   :hidden:

   Contributing
   DeveloperPolicy
   CodeReview
   SupportPolicy
   SphinxQuickstartTemplate
   Phabricator
   HowToSubmitABug
   BugLifeCycle
   CodingStandards
   GitHub
   GitBisecting
   GitRepositoryPolicy

:doc:`Contributing`
   An overview on how to contribute to LLVM.

:doc:`DeveloperPolicy`
   The LLVM project's policy towards developers and their contributions.

:doc:`CodeReview`
   The LLVM project's code-review process.

:doc:`SupportPolicy`
   The LLVM support policy for core and non-core components.

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

:doc:`GitHub`
  Describes how to use the llvm-project repository on GitHub.

:doc:`GitBisecting`
  Describes how to use ``git bisect`` on LLVM's repository.

:doc:`GitRepositoryPolicy`
   Collection of policies around the git repositories.

.. _development-process:

Development Process
-------------------

Information about LLVM's development process.

.. toctree::
   :hidden:

   Projects
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

.. _lists-forums:

Forums & Mailing Lists
----------------------

If you can't find what you need in these docs, try consulting the
Discourse forums. There are also commit mailing lists for all commits to the LLVM Project.
The :doc:`CodeOfConduct` applies to all these forums and mailing lists.

`LLVM Discourse`__
  The forums for all things LLVM and related sub-projects. There are categories and subcategories for a wide variety of areas within LLVM. You can also view tags or search for a specific topic. 

  .. __: https://discourse.llvm.org/

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

`LLVM Announcements`__
  If you just want project wide announcements such as releases, developers meetings, or blog posts, then you should check out the Announcement category on LLVM Discourse. 

  .. __: https://discourse.llvm.org/c/announce/46 

.. _online-sync-ups:

Online Sync-Ups
---------------

A number of regular calls are organized on specific topics. It should be
expected that the range of topics will change over time. At the time of
writing, the following sync-ups are organized.
The :doc:`CodeOfConduct` applies to all online sync-ups.

If you'd like to organize a new sync-up, please add the info in the table
below. Please also create a calendar event for it and invite calendar@llvm.org
to the event, so that it'll show up on the :ref:`llvm-community-calendar`. 

.. list-table:: LLVM regular sync-up calls
   :widths: 25 25 25 25
   :header-rows: 1

   * - Topic
     - Frequency
     - Calendar link
     - Minutes/docs link
   * - Loop Optimization Working Group
     - Every 2 weeks on Wednesday
     - `ics <./_static/LoopOptWG_invite.ics>`__
     - `Minutes/docs <https://docs.google.com/document/d/1sdzoyB11s0ccTZ3fobqctDpgJmRoFcz0sviKxqczs4g/edit>`__
   * - RISC-V
     - Every 2 weeks on Thursday
     - `ics <https://calendar.google.com/calendar/ical/lowrisc.org_0n5pkesfjcnp0bh5hps1p0bd80%40group.calendar.google.com/public/basic.ics>`__
       `gcal <https://calendar.google.com/calendar/b/1?cid=bG93cmlzYy5vcmdfMG41cGtlc2ZqY25wMGJoNWhwczFwMGJkODBAZ3JvdXAuY2FsZW5kYXIuZ29vZ2xlLmNvbQ>`__
     -
   * - Scalable Vectors and Arm SVE
     - Monthly, every 3rd Tuesday
     - `ics <https://calendar.google.com/calendar/ical/bjms39pe6k6bo5egtsp7don414%40group.calendar.google.com/public/basic.ics>`__
       `gcal <https://calendar.google.com/calendar/u/0/embed?src=bjms39pe6k6bo5egtsp7don414@group.calendar.google.com>`__
     - `Minutes/docs <https://docs.google.com/document/d/1UPH2Hzou5RgGT8XfO39OmVXKEibWPfdYLELSaHr3xzo/edit>`__
   * - ML Guided Compiler Optimizations
     - Monthly
     -
     - `Minutes/docs <https://docs.google.com/document/d/1JecbplF09l3swTjze-UVeLh4L48svJxGVy4mz_e9Rhs/edit?usp=gmail#heading=h.ts9cmcjbir1j>`__
   * - `LLVM security group <https://llvm.org/docs/Security.html>`__
     - Monthly, every 3rd Tuesday
     - `ics <https://calendar.google.com/calendar/ical/eoh3m9k1l6vqbd1fkp94fv5q74%40group.calendar.google.com/public/basic.ics>`__
       `gcal <https://calendar.google.com/calendar/embed?src=eoh3m9k1l6vqbd1fkp94fv5q74%40group.calendar.google.com>`__
     - `Minutes/docs <https://discourse.llvm.org/t/llvm-security-group-public-sync-ups/62735>`__
   * - `CIRCT <https://github.com/llvm/circt>`__
     - Weekly, on Wednesday
     -
     - `Minutes/docs <https://docs.google.com/document/d/1fOSRdyZR2w75D87yU2Ma9h2-_lEPL4NxvhJGJd-s5pk/edit#heading=h.mulvhjtr8dk9>`__
   * - `MLIR <https://mlir.llvm.org>`__ design meetings
     - Weekly, on Thursdays
     -
     - `Minutes/docs <https://docs.google.com/document/d/1y_9f1AbfgcoVdJh4_aM6-BaSHvrHl8zuA5G4jv_94K8/edit#heading=h.cite1kolful9>`__
   * - flang
     - Multiple meeting series, `documented here <https://github.com/llvm/llvm-project/blob/main/flang/docs/GettingInvolved.md#calls>`__
     -
     -
   * - OpenMP
     - Multiple meeting series, `documented here <https://openmp.llvm.org/docs/SupportAndFAQ.html>`__
     -
     -
   * - LLVM Alias Analysis
     - Every 4 weeks on Tuesdays
     - `ics <http://lists.llvm.org/pipermail/llvm-dev/attachments/20201103/a3499a67/attachment-0001.ics>`__
     - `Minutes/docs <https://docs.google.com/document/d/17U-WvX8qyKc3S36YUKr3xfF-GHunWyYowXbxEdpHscw>`__
   * - Windows/COFF related developments
     - Every 2 months on Thursday
     -
     - `Minutes/docs <https://docs.google.com/document/d/1A-W0Sas_oHWTEl_x_djZYoRtzAdTONMW_6l1BH9G6Bo/edit?usp=sharing>`__
   * - Vector Predication
     - Every 2 weeks on Tuesdays, 3pm UTC
     -
     - `Minutes/docs <https://docs.google.com/document/d/1q26ToudQjnqN5x31zk8zgq_s0lem1-BF8pQmciLa4k8/edit?usp=sharing>`__
   * - LLVM Pointer Authentication
     - Every month on Mondays
     - `ics <https://calendar.google.com/calendar/ical/fr1qtmrmt2s9odufjvurkb6j70%40group.calendar.google.com/public/basic.ics>`__
     - `Minutes/docs <https://discourse.llvm.org/t/llvm-pointer-authentication-sync-ups/62661>`__
   * - MemorySSA in LLVM
     - Every 8 weeks on Mondays
     - `ics <https://calendar.google.com/calendar/ical/c_1mincouiltpa24ac14of14lhi4%40group.calendar.google.com/public/basic.ics>`__
       `gcal <https://calendar.google.com/calendar/embed?src=c_1mincouiltpa24ac14of14lhi4%40group.calendar.google.com>`__
     - `Minutes/docs <https://docs.google.com/document/d/1-uEEZfmRdPThZlctOq9eXlmUaSSAAi8oKxhrPY_lpjk/edit#>`__
   * - LLVM Embedded Toolchains
     - Every 4 weeks on Thursdays
     - `ics <https://drive.google.com/file/d/1uNa-PFYkhAfT83kR2Nc4Fi706TAQFBEL/view?usp=sharing>`__
       `gcal <https://calendar.google.com/calendar/u/0?cid=ZDQyc3ZlajJmbjIzNG1jaTUybjFsdjA2dWNAZ3JvdXAuY2FsZW5kYXIuZ29vZ2xlLmNvbQ>`__
     - `Minutes/docs <https://docs.google.com/document/d/1GahxppHJ7o1O_fn1Mbidu1DHEg7V2aOr92LXCtNV1_o/edit?usp=sharing>`__
   * - Clang C and C++ Language Working Group
     - 1st and 3rd Wednesday of the month
     - `gcal <https://calendar.google.com/calendar/u/0?cid=cW1lZGg0ZXNpMnIyZDN2aTVydGVrdWF1YzRAZ3JvdXAuY2FsZW5kYXIuZ29vZ2xlLmNvbQ>`__
     - `Minutes/docs <https://docs.google.com/document/d/1x5-RbOC6-jnI_NcJ9Dp4pSmGhhNe7lUevuWUIB46TeM/edit?usp=sharing>`__

.. _office-hours:

Office hours
------------

A number of experienced LLVM contributors make themselves available for a chat
on a regular schedule, to anyone who is looking for some guidance. Please find
the list of who is available when, through which medium, and what their area of
expertise is. Don't be too shy to dial in!

The :doc:`CodeOfConduct` applies to all office hours.

Of course, people take time off from time to time, so if you dial in and you
don't find anyone present, chances are they happen to be off that day.

.. list-table:: LLVM office hours
  :widths: 15 40 15 15 15
  :header-rows: 1

  * - Name
    - In-scope topics
    - When?
    - Where?
    - Languages
  * - Kristof Beyls
    - General questions on how to contribute to LLVM; organizing meetups;
      submitting talks; and other general LLVM-related topics. Arm/AArch64
      codegen. LLVM security group. LLVM Office Hours.
    - Every 2nd and 4th Wednesday of the month at 9.30am CET, for 30 minutes.
      `ics <https://calendar.google.com/calendar/ical/co0h4ndpvtfe64opn7eraiq3ac%40group.calendar.google.com/public/basic.ics>`__
    - `Jitsi <https://meet.jit.si/KristofBeylsLLVMOfficeHour>`__
    - English, Flemish, Dutch
  * - Alina Sbirlea
    - General questions on how to contribute to LLVM; women in compilers;
      MemorySSA, BatchAA, various loop passes, new pass manager.
    - Monthly, 2nd Tuesdays, 10.00am PT/7:00pm CET, for 30 minutes.
      `ics <https://calendar.google.com/calendar/ical/c_pm6e7160iq7n5fcm1s6m3rjhh4%40group.calendar.google.com/public/basic.ics>`__
      `gcal <https://calendar.google.com/calendar/embed?src=c_pm6e7160iq7n5fcm1s6m3rjhh4%40group.calendar.google.com>`__
    - `GoogleMeet <https://meet.google.com/hhk-xpdj-gvx>`__
    - English, Romanian
  * - Aaron Ballman
    - Clang internals; frontend attributes; clang-tidy; clang-query; AST matchers
    - Monthly, 2nd Monday of the month at 10:00am Eastern, for 30 minutes.
      `ics <https://calendar.google.com/calendar/ical/npgke5dug0uliud0qapptmps58%40group.calendar.google.com/public/basic.ics>`__
      `gcal <https://calendar.google.com/calendar/embed?src=npgke5dug0uliud0qapptmps58%40group.calendar.google.com>`__
    - `GoogleMeet <https://meet.google.com/xok-iqne-gmi>`__
    - English, Norwegian (not fluently)
  * - Johannes Doerfert (he/him)
    - OpenMP, LLVM-IR, interprocedural optimizations, Attributor, workshops, research, ...
    - Every 2 weeks, Wednesdays 10:30am (Chicago Time), for 1 hour.
      `ics <https://outlook.office365.com/owa/calendar/5c93bae7d8fc48f5bd150f9efda8806f@anl.gov/4929bd0f955b4f9b82b7bc995cb3b1bf665030567582422766/S-1-8-2190330107-3188380388-3000953923-1819937831/reachcalendar.ics>`__
    - `MS Teams <https://teams.microsoft.com/l/meetup-join/19%3ameeting_MTM2ZDNkNGUtZjg3My00ZDFjLTk5NjctYTIwMmI3MWM4MjFj%40thread.v2/0?context=%7b%22Tid%22%3a%220cfca185-25f7-49e3-8ae7-704d5326e285%22%2c%22Oid%22%3a%220cacfb51-1523-4455-a443-adca936d1af0%22%7d>`__
    - English, German
  * - Tobias Grosser
    - General questions on how to contribute to LLVM/MLIR, Polly, Loop Optimization, FPL, Research in LLVM, PhD in CS, Summer of Code.
    - Monthly, last Monday of the month at 18:00 London time (typically 9am PT), for 30 minutes.
    - `Video Call <https://meet.grosser.science/LLVMOfficeHours>`__
    - English, German, Spanish, French
  * - Anastasia Stulova
    - Clang internals for C/C++ language extensions and dialects, OpenCL, GPU, SPIR-V, how to contribute, women in compilers. 
    - Monthly, 2nd Thursday of the month at 17:00 BST - London time (9:00am PT except for 2 weeks in spring), 1 hour slot.
    - `GoogleMeet <https://meet.google.com/kdy-fdbv-nuk>`__
    - English, Russian, German (not fluently)


Guidance for office hours hosts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* If you're interested in becoming an office hours host, please add your
  information to the list above. Please create a calendar event for it and
  invite calendar@llvm.org to the event so that it'll show up on the 
  :ref:`llvm-community-calendar`.
* When starting an office hours session, consider typing something like "*Hi,
  I'm available for chats in the next half hour at* video chat URL. *I'm
  looking forward to having conversations on the video chat or here.*" on the
  LLVM chat channels that you are already on. These could include:

    * the `#office-hours Discord channel
      <https://discord.com/channels/636084430946959380/976196303681896538>`__.
    * :ref:`IRC`

  Doing this can help:
    * overcome potential anxiety to call in for a first time,
    * people who prefer to first exchange a few messages through text chat
      before dialing in, and
    * remind the wider community that office hours do exist.
* If you decide to no longer host office hours, please do remove your entry
  from the list above.


.. _IRC:

IRC
---

Users and developers of the LLVM project (including subprojects such as Clang)
can be found in #llvm on `irc.oftc.net <irc://irc.oftc.net/llvm>`_.

This channel has several bots.

* Buildbot reporters

  * llvmbb - Bot for the main LLVM buildbot master.
    http://lab.llvm.org/buildbot/#/console

* robot - Bugzilla linker. %bug <number>

* clang-bot - A `geordi <http://www.eelis.net/geordi/>`_ instance running
  near-trunk clang instead of gcc.

In addition to the traditional IRC there is a
`Discord <https://discord.com/channels/636084430946959380/636725486533345280>`_
chat server available. To sign up, please use this
`invitation link <https://discord.com/invite/xS7Z362>`_.


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

   Proposals/GitHubMove
   BugpointRedesign
   Proposals/TestSuite
   Proposals/VariableNames
   Proposals/VectorizationPlan
   Proposals/VectorPredication

:doc:`Proposals/GitHubMove`
   Proposal to move from SVN/Git to GitHub.

:doc:`BugpointRedesign`
   Design doc for a redesign of the Bugpoint tool.

:doc:`Proposals/TestSuite`
   Proposals for additional benchmarks/programs for llvm's test-suite.

:doc:`Proposals/VariableNames`
   Proposal to change the variable names coding standard.

:doc:`Proposals/VectorizationPlan`
   Proposal to model the process and upgrade the infrastructure of LLVM's Loop Vectorizer.

:doc:`Proposals/VectorPredication`
   Proposal for predicated vector instructions in LLVM.

.. _llvm-community-calendar:

LLVM community calendar
-----------------------

We aim to maintain a public calendar view of all events happening in the LLVM
community such as :ref:`online-sync-ups` and :ref:`office-hours`. The calendar
can be found at
https://calendar.google.com/calendar/u/0/embed?src=calendar@llvm.org and can
also be seen inline below:

.. raw:: html

    <iframe src="https://calendar.google.com/calendar/embed?height=600&wkst=1&bgcolor=%23ffffff&ctz=UTC&showCalendars=0&showDate=1&showNav=1&src=Y2FsZW5kYXJAbGx2bS5vcmc&color=%23039BE5" style="border:solid 1px #777" width="800" height="600" frameborder="0" scrolling="no"></iframe>
