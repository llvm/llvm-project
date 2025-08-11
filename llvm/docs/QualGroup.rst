========================
LLVM Qualification Group
========================

Introduction
============

The LLVM Qualification Group is an open working group within the LLVM community. 
It was created to coordinate efforts around enabling the use of LLVM components 
in safety-critical applications such as those governed by ISO 26262 (automotive), 
DO-178C (aerospace), and EN 50128 (railways).

Motivation
==========

LLVM is increasingly used in safety-critical domains (e.g., automotive, aerospace, medical),
but currently lacks a shared structure to address the specific needs of functional safety, 
such as systematic assurance arguments, tool qualification artifacts, and their associated 
documentation.

A more open, upstream, reusable, and collaborative approach would benefit the wider ecosystem.
This group serves as a public forum for those interested in improving LLVM’s suitability for
use in such environments.

Goals
=====

The Qualification Group aims to:

* Facilitate open discussion around tool confidence and qualification practices
* Identify areas for upstream improvements (e.g., traceability hooks, quality evidence)
* Share safety-relevant documentation and argumentation templates
* Coordinate efforts across users and vendors working toward similar goals
* Act as a point of contact for safety-related collaboration

The group is non-enforcing and does not control any part of the codebase.
All technical decisions remain subject to the standard LLVM review and governance process.

Participation
=============

Participation is open to anyone interested. There are several ways to get involved:

* Join discussions on the `LLVM Discourse <https://discourse.llvm.org/>`_ forum, under the "Community" category.
* Engage in conversations on the LLVM Community Discord in the `#fusa-qual-wg <https://discord.com/channels/636084430946959380/1389362444169773117>`_ channel.
* Join our monthly sync-up calls. Details on working sessions and meeting minutes are shared on the :doc:`GettingInvolved` page.
* Contribute ideas, feedback, or patches via GitHub, Discourse, or directly in working documents.

We welcome contributors from diverse backgrounds, organizations, and experience levels.

Meeting Minutes
===============

Meeting notes for the LLVM Qualification Working Group are published on the 
LLVM Discourse forum. These notes provide a summary of topics discussed, 
decisions made, and next steps. 

You can access all minutes here:
https://discourse.llvm.org/t/llvm-qualification-wg-sync-ups-meeting-minutes/87148

Contributors
============

The LLVM Qualification Working Group is a collaborative effort involving participants 
from across the LLVM ecosystem. These include community members and industry contributors
with experience in compiler development, tool qualification, and functional safety.

While contributor names are recorded in the `Meeting Minutes`_ for those who attend 
sync-up calls, we also recognize contributions made asynchronously via Discord, GitHub, 
and other discussion channels.

All forms of constructive participation are valued and acknowledged.

Presentation Slides
===================

Slides used to support discussions during sync-up meetings are stored in the
 `qual-wg/slides/` directory of the LLVM repository.

 Available slides:

* :download:`July 2025 <qual-wg/slides/202507_llvm_qual_wg.pdf>`
* (add future entries here)

Code of Conduct
===============

We are committed to fostering a respectful, inclusive, and constructive environment 
where contributors from diverse backgrounds and organizations can collaborate 
on qualification-related efforts in the LLVM ecosystem. 
To support this goal, we adopt the following principles:

Let's Build This Together
-------------------------
This is a space for shared ownership and mutual learning. If you're here, you belong. 
Help us shape a group where trust, technical rigor, and collaboration go hand in hand.

Respect and Inclusion
---------------------
* Treat all participants with respect and dignity, regardless of background, experience level, employer, or role in the community.
* Be welcoming and supportive. We value a diversity of opinions and expertise.
* Assume good intent, and ask questions before drawing conclusions.

Constructive Collaboration
--------------------------
* Keep discussions focused, technical, and solution-oriented.
* Provide thoughtful, actionable feedback. Avoid sarcasm, dismissive remarks, or personal criticism.
* Recognize that contributors have different constraints and priorities. Seek alignment, not perfection.

Transparency and Openness
-------------------------
* Share relevant information openly to enable others to contribute effectively.
* Document decisions and rationales so others can understand and build on them.
* Clearly distinguish between personal opinions, organizational positions, and community consensus.

Unacceptable Behavior
---------------------
We will not tolerate:
* Harassment, discrimination, or exclusionary behavior.
* Disruptive conduct in meetings or communication channels.
* Using this group for marketing, lobbying, or promoting non-collaborative commercial agendas.

Safety and Trust
----------------
* We aim to build qualification artifacts that others can trust. Similarly, we aim to be trustworthy collaborators.
* If you see something concerning, speak up respectfully or contact the group organizer(s) privately.
* We follow the LLVM Community :doc:`Code of Conduct <CodeOfConduct>`, which applies across all official LLVM communication spaces.

Contact
=======

For more information or to get involved:

* Refer to our initial `RFC: Proposal to Establish a Safety Group in LLVM <https://discourse.llvm.org/t/rfc-proposal-to-establish-a-safety-group-in-llvm/86916>`_ on the LLVM Discourse forum.
* Join the conversation on the LLVM Community Discord in the `#fusa-qual-wg <https://discord.com/channels/636084430946959380/1389362444169773117>`_ channel.
