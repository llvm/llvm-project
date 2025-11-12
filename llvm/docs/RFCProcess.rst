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

After posting a major proposal, it is common to receive lots of conflicting
feedback from different parties, or no feedback at all, leaving authors without
clear next steps. As a community, we are aiming for `"rough consensus"
<https://en.wikipedia.org/wiki/Rough_consensus>`_, similar in spirit to what is
described in `IETF RFC7282 <https://datatracker.ietf.org/doc/html/rfc7282>`_.
This requires considering and addressing all of the objections to the RFC, and
confirming that we can all live with the tradeoffs embodied in the proposal.

The LLVM Area Teams (defined in `LP0004
<https://github.com/llvm/llvm-www/blob/main/proposals/LP0004-project-governance.md>`_)
are responsible for facilitating project decision making. In cases where there
isn't obvious agreement, area teams should step in to restate their perceived
consensus. In cases of deeper disagreement, area teams should try to identify
the next steps for the proposal, such as gathering more data, changing the
proposal, or rejecting it in the absence of major changes in the design or
context. They can also act as moderators by scheduling calls for participants
to speak directly to resolve disagreements, subject to normal
:ref:`Code of Conduct <LLVM Community Code of Conduct>` guidelines.

Once the design of the new feature is finalized, the work itself should be done
as a series of :ref:`incremental changes <incremental-changes>`, not as a long-term development branch.


Trivial Acceptance or Rejection
-------------------------------
Some proposals have obvious consensus (for or against) after discussion in the
community. It is acceptable to presume a post which appears to have obvious
consensus has been accepted.

Non-trivial Acceptance or Rejection
-----------------------------------
If the proposal does not have obvious consensus after community discussion,
a maintainer for each of the impacted parts of the project should explicitly
accept or reject the RFC by leaving a comment stating their decision and
possibly detailing any provisions for their acceptance. Overall consensus is
determined once a maintainer from each impacted part of the project has
accepted the proposal.

Low Engagement Level
~~~~~~~~~~~~~~~~~~~~
If the proposal gets little or no engagement by the community, it is a sign that
the proposal does not have consensus and is rejected. Engagement means comments
on the proposal. If there are few or no comments but the are a lot of people
pressing the like/heart button on the post, the appropriate area team can make
a value judgement on whether to accept or reject.

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

Suggestions on Getting a Change Accepted
----------------------------------------
These are some suggestions for how to get a major change accepted:

* Make it targeted, and avoid touching components irrelevant to the task.

* Explain how the change improves LLVM for other stakeholders rather than
  focusing on your specific use case.

* As discussion evolves, periodically summarize the current state of the
  discussion and clearly separate points where consensus seems to emerge from
  those where further discussion is necessary.

* Compilers are foundational infrastructure, so there is a high quality bar,
  and the burden of proof is on the proposer. If reviewers repeatedly ask for
  an unreasonable amount of evidence or data, proposal authors can escalate to
  the area team to resolve disagreements.