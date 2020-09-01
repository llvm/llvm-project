===================
LLVM Bug Life Cycle
===================

.. contents::
   :local:



Introduction - Achieving consistency in how we deal with bug reports
====================================================================

We aim to achieve a basic level of consistency in how reported bugs evolve from
being reported, to being worked on, and finally getting closed out. The
consistency helps reporters, developers and others to gain a better
understanding of what a particular bug state actually means and what to expect
might happen next.

At the same time, we aim to not over-specify the life cycle of bugs in the
`the LLVM Bug Tracking System <https://bugs.llvm.org/enter_bug.cgi>`_, as the
overall goal is to make it easier to work with and understand the bug reports.

The main parts of the life cycle documented here are:

#. `Reporting`_
#. `Triaging`_
#. `Actively working on fixing`_
#. `Closing`_

Furthermore, some of the metadata in the bug tracker, such as who to notify on
newly reported bugs or what the breakdown into products & components is we use,
needs to be maintained. See the following for details:

#. `Maintenance of Bug products/component metadata`_
#. `Maintenance of cc-by-default settings`_


.. _Reporting:

Reporting bugs
==============

See :doc:`HowToSubmitABug` on further details on how to submit good bug reports.

Make sure that you have one or more people on cc on the bug report that you
think will react to it. We aim to automatically add specific people on cc for
most products/components, but may not always succeed in doing so.

If you know the area of LLVM code the root cause of the bug is in, good
candidates to add as cc may be the same people you'd ask for a code review in
that area. See :ref:`finding-potential-reviewers` for more details.


.. _Triaging:

Triaging bugs
=============

Bugs with status NEW indicate that they still need to be triaged.
When triage is complete, the status of the bug is moved to CONFIRMED.

The goal of triaging a bug is to make sure a newly reported bug ends up in a
good, actionable, state. Try to answer the following questions while triaging.

* Is the reported behavior actually wrong?

  * E.g. does a miscompile example depend on undefined behavior?

* Can you easily reproduce the bug?

  * If not, are there reasonable excuses why it cannot easily be reproduced?

* Is it related to an already reported bug?

  * Use the "See also"/"depends on"/"blocks" fields if so.
  * Close it as a duplicate if so, pointing to the issue it duplicates.

* Are the following fields filled in correctly?

  * Product
  * Component
  * Title

* CC others not already cc’ed that you happen to know would be good to pull in.
* Add the "beginner" keyword if you think this would be a good bug to be fixed
  by someone new to LLVM.

.. _Actively working on fixing:

Actively working on fixing bugs
===============================

Please remember to assign the bug to yourself if you're actively working on
fixing it and to unassign it when you're no longer actively working on it.  You
unassign a bug by setting the Assignee field to "unassignedbugs@nondot.org".

.. _Closing:

Resolving/Closing bugs
======================

For simplicity, we only have 1 status for all resolved or closed bugs:
RESOLVED.

Resolving bugs is good! Make sure to properly record the reason for resolving.
Examples of reasons for resolving are:

* Revision NNNNNN fixed the bug.
* The bug cannot be reproduced with revision NNNNNN.
* The circumstances for the bug don't apply anymore.
* There is a sound reason for not fixing it (WONTFIX).
* There is a specific and plausible reason to think that a given bug is
  otherwise inapplicable or obsolete.

  * One example is an old open bug that doesn't contain enough information to
    clearly understand the problem being reported (e.g. not reproducible). It is
    fine to resolve such a bug e.g. with resolution WORKSFORME and leaving a
    comment to encourage the reporter to reopen the bug with more information
    if it's still reproducible on their end.

If a bug is resolved, please fill in the revision number it was fixed in in the
"Fixed by Commit(s)" field.


.. _Maintenance of Bug products/component metadata:

Maintenance of products/components metadata
===========================================

Please raise a bug against "Bugzilla Admin"/"Products" to request any changes
to be made to the breakdown of products & components modeled in Bugzilla.


.. _Maintenance of cc-by-default settings:

Maintenance of cc-by-default settings
=====================================

Please raise a bug against "Bugzilla Admin"/"Products" to request any changes
to be made to the cc-by-default settings for specific components.
