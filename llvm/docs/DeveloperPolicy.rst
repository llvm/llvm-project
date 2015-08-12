=====================
LLVM Developer Policy
=====================

.. contents::
   :local:

Introduction
============

This document contains the LLVM Developer Policy which defines the project's
policy towards developers and their contributions. The intent of this policy is
to eliminate miscommunication, rework, and confusion that might arise from the
distributed nature of LLVM's development.  By stating the policy in clear terms,
we hope each developer can know ahead of time what to expect when making LLVM
contributions.  This policy covers all llvm.org subprojects, including Clang,
LLDB, libc++, etc.

This policy is also designed to accomplish the following objectives:

#. Attract both users and developers to the LLVM project.

#. Make life as simple and easy for contributors as possible.

#. Keep the top of Subversion trees as stable as possible.

#. Establish awareness of the project's :ref:`copyright, license, and patent
   policies <copyright-license-patents>` with contributors to the project.

This policy is aimed at frequent contributors to LLVM. People interested in
contributing one-off patches can do so in an informal way by sending them to the
`llvm-commits mailing list
<http://lists.llvm.org/mailman/listinfo/llvm-commits>`_ and engaging another
developer to see it through the process.

Developer Policies
==================

This section contains policies that pertain to frequent LLVM developers.  We
always welcome `one-off patches`_ from people who do not routinely contribute to
LLVM, but we expect more from frequent contributors to keep the system as
efficient as possible for everyone.  Frequent LLVM contributors are expected to
meet the following requirements in order for LLVM to maintain a high standard of
quality.

Stay Informed
-------------

Developers should stay informed by reading at least the "dev" mailing list for
the projects you are interested in, such as `llvm-dev
<http://lists.llvm.org/mailman/listinfo/llvm-dev>`_ for LLVM, `cfe-dev
<http://lists.llvm.org/mailman/listinfo/cfe-dev>`_ for Clang, or `lldb-dev
<http://lists.llvm.org/mailman/listinfo/lldb-dev>`_ for LLDB.  If you are
doing anything more than just casual work on LLVM, it is suggested that you also
subscribe to the "commits" mailing list for the subproject you're interested in,
such as `llvm-commits
<http://lists.llvm.org/mailman/listinfo/llvm-commits>`_, `cfe-commits
<http://lists.llvm.org/mailman/listinfo/cfe-commits>`_, or `lldb-commits
<http://lists.llvm.org/mailman/listinfo/lldb-commits>`_.  Reading the
"commits" list and paying attention to changes being made by others is a good
way to see what other people are interested in and watching the flow of the
project as a whole.

We recommend that active developers register an email account with `LLVM
Bugzilla <http://llvm.org/bugs/>`_ and preferably subscribe to the `llvm-bugs
<http://lists.llvm.org/mailman/listinfo/llvm-bugs>`_ email list to keep track
of bugs and enhancements occurring in LLVM.  We really appreciate people who are
proactive at catching incoming bugs in their components and dealing with them
promptly.

Please be aware that all public LLVM mailing lists are public and archived, and
that notices of confidentiality or non-disclosure cannot be respected.

.. _patch:
.. _one-off patches:

Making and Submitting a Patch
-----------------------------

When making a patch for review, the goal is to make it as easy for the reviewer
to read it as possible.  As such, we recommend that you:

#. Make your patch against the Subversion trunk, not a branch, and not an old
   version of LLVM.  This makes it easy to apply the patch.  For information on
   how to check out SVN trunk, please see the `Getting Started
   Guide <GettingStarted.html#checkout>`_.

#. Similarly, patches should be submitted soon after they are generated.  Old
   patches may not apply correctly if the underlying code changes between the
   time the patch was created and the time it is applied.

#. Patches should be made with ``svn diff``, or similar. If you use a
   different tool, make sure it uses the ``diff -u`` format and that it
   doesn't contain clutter which makes it hard to read.

#. If you are modifying generated files, such as the top-level ``configure``
   script, please separate out those changes into a separate patch from the rest
   of your changes.

Once your patch is ready, submit it by emailing it to the appropriate project's
commit mailing list (or commit it directly if applicable). Alternatively, some
patches get sent to the project's development list or component of the LLVM bug
tracker, but the commit list is the primary place for reviews and should
generally be preferred.

When sending a patch to a mailing list, it is a good idea to send it as an
*attachment* to the message, not embedded into the text of the message.  This
ensures that your mailer will not mangle the patch when it sends it (e.g. by
making whitespace changes or by wrapping lines).

*For Thunderbird users:* Before submitting a patch, please open *Preferences >
Advanced > General > Config Editor*, find the key
``mail.content_disposition_type``, and set its value to ``1``. Without this
setting, Thunderbird sends your attachment using ``Content-Disposition: inline``
rather than ``Content-Disposition: attachment``. Apple Mail gamely displays such
a file inline, making it difficult to work with for reviewers using that
program.

When submitting patches, please do not add confidentiality or non-disclosure
notices to the patches themselves.  These notices conflict with the `LLVM
License`_ and may result in your contribution being excluded.

.. _code review:

Code Reviews
------------

LLVM has a code review policy. Code review is one way to increase the quality of
software. We generally follow these policies:

#. All developers are required to have significant changes reviewed before they
   are committed to the repository.

#. Code reviews are conducted by email on the relevant project's commit mailing
   list, or alternatively on the project's development list or bug tracker.

#. Code can be reviewed either before it is committed or after.  We expect major
   changes to be reviewed before being committed, but smaller changes (or
   changes where the developer owns the component) can be reviewed after commit.

#. The developer responsible for a code change is also responsible for making
   all necessary review-related changes.

#. Code review can be an iterative process, which continues until the patch is
   ready to be committed. Specifically, once a patch is sent out for review, it
   needs an explicit "looks good" before it is submitted. Do not assume silent
   approval, or request active objections to the patch with a deadline.

Sometimes code reviews will take longer than you would hope for, especially for
larger features. Accepted ways to speed up review times for your patches are:

* Review other people's patches. If you help out, everybody will be more
  willing to do the same for you; goodwill is our currency.
* Ping the patch. If it is urgent, provide reasons why it is important to you to
  get this patch landed and ping it every couple of days. If it is
  not urgent, the common courtesy ping rate is one week. Remember that you're
  asking for valuable time from other professional developers.
* Ask for help on IRC. Developers on IRC will be able to either help you
  directly, or tell you who might be a good reviewer.
* Split your patch into multiple smaller patches that build on each other. The
  smaller your patch, the higher the probability that somebody will take a quick
  look at it.

Developers should participate in code reviews as both reviewers and
reviewees. If someone is kind enough to review your code, you should return the
favor for someone else.  Note that anyone is welcome to review and give feedback
on a patch, but only people with Subversion write access can approve it.

There is a web based code review tool that can optionally be used
for code reviews. See :doc:`Phabricator`.

Code Owners
-----------

The LLVM Project relies on two features of its process to maintain rapid
development in addition to the high quality of its source base: the combination
of code review plus post-commit review for trusted maintainers.  Having both is
a great way for the project to take advantage of the fact that most people do
the right thing most of the time, and only commit patches without pre-commit
review when they are confident they are right.

The trick to this is that the project has to guarantee that all patches that are
committed are reviewed after they go in: you don't want everyone to assume
someone else will review it, allowing the patch to go unreviewed.  To solve this
problem, we have a notion of an 'owner' for a piece of the code.  The sole
responsibility of a code owner is to ensure that a commit to their area of the
code is appropriately reviewed, either by themself or by someone else.  The list
of current code owners can be found in the file
`CODE_OWNERS.TXT <http://llvm.org/viewvc/llvm-project/llvm/trunk/CODE_OWNERS.TXT?view=markup>`_
in the root of the LLVM source tree.

Note that code ownership is completely different than reviewers: anyone can
review a piece of code, and we welcome code review from anyone who is
interested.  Code owners are the "last line of defense" to guarantee that all
patches that are committed are actually reviewed.

Being a code owner is a somewhat unglamorous position, but it is incredibly
important for the ongoing success of the project.  Because people get busy,
interests change, and unexpected things happen, code ownership is purely opt-in,
and anyone can choose to resign their "title" at any time. For now, we do not
have an official policy on how one gets elected to be a code owner.

.. _include a testcase:

Test Cases
----------

Developers are required to create test cases for any bugs fixed and any new
features added.  Some tips for getting your testcase approved:

* All feature and regression test cases are added to the ``llvm/test``
  directory. The appropriate sub-directory should be selected (see the
  :doc:`Testing Guide <TestingGuide>` for details).

* Test cases should be written in :doc:`LLVM assembly language <LangRef>`.

* Test cases, especially for regressions, should be reduced as much as possible,
  by :doc:`bugpoint <Bugpoint>` or manually. It is unacceptable to place an
  entire failing program into ``llvm/test`` as this creates a *time-to-test*
  burden on all developers. Please keep them short.

Note that llvm/test and clang/test are designed for regression and small feature
tests only. More extensive test cases (e.g., entire applications, benchmarks,
etc) should be added to the ``llvm-test`` test suite.  The llvm-test suite is
for coverage (correctness, performance, etc) testing, not feature or regression
testing.

Quality
-------

The minimum quality standards that any change must satisfy before being
committed to the main development branch are:

#. Code must adhere to the `LLVM Coding Standards <CodingStandards.html>`_.

#. Code must compile cleanly (no errors, no warnings) on at least one platform.

#. Bug fixes and new features should `include a testcase`_ so we know if the
   fix/feature ever regresses in the future.

#. Code must pass the ``llvm/test`` test suite.

#. The code must not cause regressions on a reasonable subset of llvm-test,
   where "reasonable" depends on the contributor's judgement and the scope of
   the change (more invasive changes require more testing). A reasonable subset
   might be something like "``llvm-test/MultiSource/Benchmarks``".

Additionally, the committer is responsible for addressing any problems found in
the future that the change is responsible for.  For example:

* The code should compile cleanly on all supported platforms.

* The changes should not cause any correctness regressions in the ``llvm-test``
  suite and must not cause any major performance regressions.

* The change set should not cause performance or correctness regressions for the
  LLVM tools.

* The changes should not cause performance or correctness regressions in code
  compiled by LLVM on all applicable targets.

* You are expected to address any `Bugzilla bugs <http://llvm.org/bugs/>`_ that
  result from your change.

We prefer for this to be handled before submission but understand that it isn't
possible to test all of this for every submission.  Our build bots and nightly
testing infrastructure normally finds these problems.  A good rule of thumb is
to check the nightly testers for regressions the day after your change.  Build
bots will directly email you if a group of commits that included yours caused a
failure.  You are expected to check the build bot messages to see if they are
your fault and, if so, fix the breakage.

Commits that violate these quality standards (e.g. are very broken) may be
reverted. This is necessary when the change blocks other developers from making
progress. The developer is welcome to re-commit the change after the problem has
been fixed.

.. _commit messages:

Commit messages
---------------

Although we don't enforce the format of commit messages, we prefer that
you follow these guidelines to help review, search in logs, email formatting
and so on. These guidelines are very similar to rules used by other open source
projects.

Most importantly, the contents of the message should be carefully written to
convey the rationale of the change (without delving too much in detail). It
also should avoid being vague or overly specific. For example, "bits were not
set right" will leave the reviewer wondering about which bits, and why they
weren't right, while "Correctly set overflow bits in TargetInfo" conveys almost
all there is to the change.

Below are some guidelines about the format of the message itself:

* Separate the commit message into title, body and, if you're not the original
  author, a "Patch by" attribution line (see below).

* The title should be concise. Because all commits are emailed to the list with
  the first line as the subject, long titles are frowned upon.  Short titles
  also look better in `git log`.

* When the changes are restricted to a specific part of the code (e.g. a
  back-end or optimization pass), it is customary to add a tag to the
  beginning of the line in square brackets.  For example, "[SCEV] ..."
  or "[OpenMP] ...". This helps email filters and searches for post-commit
  reviews.

* The body, if it exists, should be separated from the title by an empty line.

* The body should be concise, but explanatory, including a complete
  reasoning.  Unless it is required to understand the change, examples,
  code snippets and gory details should be left to bug comments, web
  review or the mailing list.

* If the patch fixes a bug in bugzilla, please include the PR# in the message.

* `Attribution of Changes`_ should be in a separate line, after the end of
  the body, as simple as "Patch by John Doe.". This is how we officially
  handle attribution, and there are automated processes that rely on this
  format.

* Text formatting and spelling should follow the same rules as documentation
  and in-code comments, ex. capitalization, full stop, etc.

* If the commit is a bug fix on top of another recently committed patch, or a
  revert or reapply of a patch, include the svn revision number of the prior
  related commit. This could be as simple as "Revert rNNNN because it caused
  PR#".

For minor violations of these recommendations, the community normally favors
reminding the contributor of this policy over reverting. Minor corrections and
omissions can be handled by sending a reply to the commits mailing list.

Obtaining Commit Access
-----------------------

We grant commit access to contributors with a track record of submitting high
quality patches.  If you would like commit access, please send an email to
`Chris <mailto:sabre@nondot.org>`_ with the following information:

#. The user name you want to commit with, e.g. "hacker".

#. The full name and email address you want message to llvm-commits to come
   from, e.g. "J. Random Hacker <hacker@yoyodyne.com>".

#. A "password hash" of the password you want to use, e.g. "``2ACR96qjUqsyM``".
   Note that you don't ever tell us what your password is; you just give it to
   us in an encrypted form.  To get this, run "``htpasswd``" (a utility that
   comes with apache) in crypt mode (often enabled with "``-d``"), or find a web
   page that will do it for you.

Once you've been granted commit access, you should be able to check out an LLVM
tree with an SVN URL of "https://username@llvm.org/..." instead of the normal
anonymous URL of "http://llvm.org/...".  The first time you commit you'll have
to type in your password.  Note that you may get a warning from SVN about an
untrusted key; you can ignore this.  To verify that your commit access works,
please do a test commit (e.g. change a comment or add a blank line).  Your first
commit to a repository may require the autogenerated email to be approved by a
mailing list.  This is normal and will be done when the mailing list owner has
time.

If you have recently been granted commit access, these policies apply:

#. You are granted *commit-after-approval* to all parts of LLVM.  To get
   approval, submit a `patch`_ to `llvm-commits
   <http://lists.llvm.org/mailman/listinfo/llvm-commits>`_. When approved,
   you may commit it yourself.

#. You are allowed to commit patches without approval which you think are
   obvious. This is clearly a subjective decision --- we simply expect you to
   use good judgement.  Examples include: fixing build breakage, reverting
   obviously broken patches, documentation/comment changes, any other minor
   changes.

#. You are allowed to commit patches without approval to those portions of LLVM
   that you have contributed or maintain (i.e., have been assigned
   responsibility for), with the proviso that such commits must not break the
   build.  This is a "trust but verify" policy, and commits of this nature are
   reviewed after they are committed.

#. Multiple violations of these policies or a single egregious violation may
   cause commit access to be revoked.

In any case, your changes are still subject to `code review`_ (either before or
after they are committed, depending on the nature of the change).  You are
encouraged to review other peoples' patches as well, but you aren't required
to do so.

.. _discuss the change/gather consensus:

Making a Major Change
---------------------

When a developer begins a major new project with the aim of contributing it back
to LLVM, they should inform the community with an email to the `llvm-dev
<http://lists.llvm.org/mailman/listinfo/llvm-dev>`_ email list, to the extent
possible. The reason for this is to:

#. keep the community informed about future changes to LLVM,

#. avoid duplication of effort by preventing multiple parties working on the
   same thing and not knowing about it, and

#. ensure that any technical issues around the proposed work are discussed and
   resolved before any significant work is done.

The design of LLVM is carefully controlled to ensure that all the pieces fit
together well and are as consistent as possible. If you plan to make a major
change to the way LLVM works or want to add a major new extension, it is a good
idea to get consensus with the development community before you start working on
it.

Once the design of the new feature is finalized, the work itself should be done
as a series of `incremental changes`_, not as a long-term development branch.

.. _incremental changes:

Incremental Development
-----------------------

In the LLVM project, we do all significant changes as a series of incremental
patches.  We have a strong dislike for huge changes or long-term development
branches.  Long-term development branches have a number of drawbacks:

#. Branches must have mainline merged into them periodically.  If the branch
   development and mainline development occur in the same pieces of code,
   resolving merge conflicts can take a lot of time.

#. Other people in the community tend to ignore work on branches.

#. Huge changes (produced when a branch is merged back onto mainline) are
   extremely difficult to `code review`_.

#. Branches are not routinely tested by our nightly tester infrastructure.

#. Changes developed as monolithic large changes often don't work until the
   entire set of changes is done.  Breaking it down into a set of smaller
   changes increases the odds that any of the work will be committed to the main
   repository.

To address these problems, LLVM uses an incremental development style and we
require contributors to follow this practice when making a large/invasive
change.  Some tips:

* Large/invasive changes usually have a number of secondary changes that are
  required before the big change can be made (e.g. API cleanup, etc).  These
  sorts of changes can often be done before the major change is done,
  independently of that work.

* The remaining inter-related work should be decomposed into unrelated sets of
  changes if possible.  Once this is done, define the first increment and get
  consensus on what the end goal of the change is.

* Each change in the set can be stand alone (e.g. to fix a bug), or part of a
  planned series of changes that works towards the development goal.

* Each change should be kept as small as possible. This simplifies your work
  (into a logical progression), simplifies code review and reduces the chance
  that you will get negative feedback on the change. Small increments also
  facilitate the maintenance of a high quality code base.

* Often, an independent precursor to a big change is to add a new API and slowly
  migrate clients to use the new API.  Each change to use the new API is often
  "obvious" and can be committed without review.  Once the new API is in place
  and used, it is much easier to replace the underlying implementation of the
  API.  This implementation change is logically separate from the API
  change.

If you are interested in making a large change, and this scares you, please make
sure to first `discuss the change/gather consensus`_ then ask about the best way
to go about making the change.

Attribution of Changes
----------------------

When contributors submit a patch to an LLVM project, other developers with
commit access may commit it for the author once appropriate (based on the
progression of code review, etc.). When doing so, it is important to retain
correct attribution of contributions to their contributors. However, we do not
want the source code to be littered with random attributions "this code written
by J. Random Hacker" (this is noisy and distracting). In practice, the revision
control system keeps a perfect history of who changed what, and the CREDITS.txt
file describes higher-level contributions. If you commit a patch for someone
else, please follow the attribution of changes in the simple manner as outlined
by the `commit messages`_ section. Overall, please do not add contributor names
to the source code.

Also, don't commit patches authored by others unless they have submitted the
patch to the project or you have been authorized to submit them on their behalf
(you work together and your company authorized you to contribute the patches,
etc.). The author should first submit them to the relevant project's commit
list, development list, or LLVM bug tracker component. If someone sends you
a patch privately, encourage them to submit it to the appropriate list first.


IR Backwards Compatibility
--------------------------

When the IR format has to be changed, keep in mind that we try to maintain some
backwards compatibility. The rules are intended as a balance between convenience
for llvm users and not imposing a big burden on llvm developers:

* The textual format is not backwards compatible. We don't change it too often,
  but there are no specific promises.

* The bitcode format produced by a X.Y release will be readable by all following
  X.Z releases and the (X+1).0 release.

* Newer releases can ignore features from older releases, but they cannot
  miscompile them. For example, if nsw is ever replaced with something else,
  dropping it would be a valid way to upgrade the IR.

* Debug metadata is special in that it is currently dropped during upgrades.

* Non-debug metadata is defined to be safe to drop, so a valid way to upgrade
  it is to drop it. That is not very user friendly and a bit more effort is
  expected, but no promises are made.

.. _copyright-license-patents:

Copyright, License, and Patents
===============================

.. note::

   This section deals with legal matters but does not provide legal advice.  We
   are not lawyers --- please seek legal counsel from an attorney.

This section addresses the issues of copyright, license and patents for the LLVM
project.  The copyright for the code is held by the individual contributors of
the code and the terms of its license to LLVM users and developers is the
`University of Illinois/NCSA Open Source License
<http://www.opensource.org/licenses/UoI-NCSA.php>`_ (with portions dual licensed
under the `MIT License <http://www.opensource.org/licenses/mit-license.php>`_,
see below).  As contributor to the LLVM project, you agree to allow any
contributions to the project to licensed under these terms.

Copyright
---------

The LLVM project does not require copyright assignments, which means that the
copyright for the code in the project is held by its respective contributors who
have each agreed to release their contributed code under the terms of the `LLVM
License`_.

An implication of this is that the LLVM license is unlikely to ever change:
changing it would require tracking down all the contributors to LLVM and getting
them to agree that a license change is acceptable for their contribution.  Since
there are no plans to change the license, this is not a cause for concern.

As a contributor to the project, this means that you (or your company) retain
ownership of the code you contribute, that it cannot be used in a way that
contradicts the license (which is a liberal BSD-style license), and that the
license for your contributions won't change without your approval in the
future.

.. _LLVM License:

License
-------

We intend to keep LLVM perpetually open source and to use a liberal open source
license. **As a contributor to the project, you agree that any contributions be
licensed under the terms of the corresponding subproject.** All of the code in
LLVM is available under the `University of Illinois/NCSA Open Source License
<http://www.opensource.org/licenses/UoI-NCSA.php>`_, which boils down to
this:

* You can freely distribute LLVM.
* You must retain the copyright notice if you redistribute LLVM.
* Binaries derived from LLVM must reproduce the copyright notice (e.g. in an
  included readme file).
* You can't use our names to promote your LLVM derived products.
* There's no warranty on LLVM at all.

We believe this fosters the widest adoption of LLVM because it **allows
commercial products to be derived from LLVM** with few restrictions and without
a requirement for making any derived works also open source (i.e.  LLVM's
license is not a "copyleft" license like the GPL). We suggest that you read the
`License <http://www.opensource.org/licenses/UoI-NCSA.php>`_ if further
clarification is needed.

In addition to the UIUC license, the runtime library components of LLVM
(**compiler_rt, libc++, and libclc**) are also licensed under the `MIT License
<http://www.opensource.org/licenses/mit-license.php>`_, which does not contain
the binary redistribution clause.  As a user of these runtime libraries, it
means that you can choose to use the code under either license (and thus don't
need the binary redistribution clause), and as a contributor to the code that
you agree that any contributions to these libraries be licensed under both
licenses.  We feel that this is important for runtime libraries, because they
are implicitly linked into applications and therefore should not subject those
applications to the binary redistribution clause. This also means that it is ok
to move code from (e.g.)  libc++ to the LLVM core without concern, but that code
cannot be moved from the LLVM core to libc++ without the copyright owner's
permission.

Note that the LLVM Project does distribute dragonegg, **which is
GPL.** This means that anything "linked" into dragonegg must itself be compatible
with the GPL, and must be releasable under the terms of the GPL.  This implies
that **any code linked into dragonegg and distributed to others may be subject to
the viral aspects of the GPL** (for example, a proprietary code generator linked
into dragonegg must be made available under the GPL).  This is not a problem for
code already distributed under a more liberal license (like the UIUC license),
and GPL-containing subprojects are kept in separate SVN repositories whose
LICENSE.txt files specifically indicate that they contain GPL code.

We have no plans to change the license of LLVM.  If you have questions or
comments about the license, please contact the `LLVM Developer's Mailing
List <mailto:llvm-dev@lists.llvm.org>`_.

Patents
-------

To the best of our knowledge, LLVM does not infringe on any patents (we have
actually removed code from LLVM in the past that was found to infringe).  Having
code in LLVM that infringes on patents would violate an important goal of the
project by making it hard or impossible to reuse the code for arbitrary purposes
(including commercial use).

When contributing code, we expect contributors to notify us of any potential for
patent-related trouble with their changes (including from third parties).  If
you or your employer own the rights to a patent and would like to contribute
code to LLVM that relies on it, we require that the copyright owner sign an
agreement that allows any other user of LLVM to freely use your patent.  Please
contact the `oversight group <mailto:llvm-oversight@cs.uiuc.edu>`_ for more
details.
