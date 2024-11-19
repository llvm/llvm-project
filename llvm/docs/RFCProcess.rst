=================================
Request For Comment (RFC) process
=================================

.. contents::
   :local:
   :depth: 1

Introduction
============
Substantive changes to LLVM projects need to be acceptable to the wider
community, which requires gaining community consensus to adopt the changes.
This is done by posting an RFC and obtaining feedback about the proposal.

Process
=======

Writing an RFC
--------------
The process begins with writing a proposal for the changes you'd like to see
made. The proposal should include:

* a detailed overview of the proposed changes,
* the motivation for why the changes are being proposed,
* the impact on different parts of the project, and
* any open questions the community should address.

The proposal should be posted to the appropriate forum on
`Discourse <https://discourse.llvm.org/>`_.

Feedback Period
---------------
Once the RFC is posted, the community will provide feedback on the proposal.
The feedback period is a collaborative effort between the community and the
proposal authors. Authors should take the community's feedback into
consideration and edit the original post to incorporate relevant changes they
agree to. Edits should be made such that it's clear what has changed. Editing
the original post makes it easier for the community to understand the proposal
without having to read every comment on the thread, though this can make
reading the comment thread somewhat more difficult as comments may be referring
to words no longer in the proposal.

There is not a set time limit to the feedback period; it lasts as long as
discussion is actively continuing on the proposal.

Trivial Acceptance or Rejection
-------------------------------
If the proposal has obvious consensus (for or against), a maintainer for each
of the impacted parts of the project will explicitly accept or reject the RFC
by leaving a comment stating their decision and possibly detailing any
provisions for their acceptance. Overall consensus is determined once a
maintainer from each impacted part of the project has accepted the proposal.

Low Engagement Level
~~~~~~~~~~~~~~~~~~~~
If a proposal gets little or no engagement by the community, it is a sign that
the proposal does not have consensus and is rejected. Engagement means comments
on the proposal. If there are few or no comments but the are a lot of people
pressing the like/heart button on the post, maintainers can make a value
judgement on whether to accept or reject.

After Acceptance
----------------
Once an RFC has been accepted, the authors may begin merging pull requests
related to the proposal. While the RFC process typically makes reviewing the
pull requests go more smoothly, the review process may identify additional
necessary changes to the proposal. Minor changes to the proposal do not require
an additional RFC. However, if the proposal changes significantly in a material
way, the authors may be asked to run another RFC.

After Rejection
---------------
Any rejected RFC can be brought back to the community as a new RFC in the
future. The new RFC should either clearly identify new information that may
change the community's perception of the proposal and/or explicitly address the
concerns previously raised by the community. It is helpful to explicitly call
out such information in the subsequent RFC.
