.. CHANGE TRACKER for reference
.. Purpose: Fixed document location and added Current Topics & Backlog
.. Author: Carlos Andres Ramirez
.. Last updated: 2025-09-08 by Carlos Ramirez

========================
LLVM Qualification Group
========================

Introduction
============

The LLVM Qualification Group is an open working group within the LLVM community. 
It was created to coordinate efforts around enabling the use of LLVM components 
in safety-critical applications governed by functional safety standards 
such as IEC 61508 (for general E/E/PE systems), IEC 62304 (medical devices), 
ISO 26262 (automotive), DO-178C (aerospace), and EN 50716 (railways).

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

Group Composition
=================

Group Members
-------------

The members of the LLVM Qualification Group represent a diverse cross-section of the LLVM community, including individual contributors, researchers, vendor representatives, and experts in the field of software qualification, including reliability, quality, safety, and/or security.
They meet the criteria for inclusion below. Knowing their handles help us keep track of who’s who across platforms, coordinate activities, and recognize contributions.

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Name
     - Affiliation
     - Discourse handle
     - Discord handle
     - GitHub handle
   * - Alan Phipps
     - Texas Instruments
     - evodius96
     - alanphipps
     - evodius96
   * - Carlos Andrés Ramírez
     - Woven by Toyota
     - CarlosAndresRamirez
     - carlos\_andres\_ramirez
     - CarlosAndresRamirez
   * - Davide Cunial
     - BMW A.G.
     - capitan-davide
     - capitan_davide
     - capitan-davide
   * - Jorge Pinto Sousa
     - Critical Techworks
     - sousajo-cc
     - sousajo-cc
     - sousajo-cc
   * - José Rui Simões
     - Critical Software
     - jr-simoes
     - jr_simoes
     - iznogoud-zz
   * - Oscar Slotosch
     - Validas
     - slotosch
     - oscarslotosch_66740
     - \-
   * - Petar Jovanovic
     - HTECH
     - petarj
     - petarjovanovic_18635
     - petar-jovanovic
   * - Petter Berntsson
     - Arm Limited
     - petbernt
     - petbernt
     - petbernt
   * - Wendi Urribarri
     - Woven by Toyota
     - uwendi
     - uwendi
     - uwendi
   * - YoungJun Lee
     - NSHC
     - YoungJunLee
     - YoungJunLee
     - IamYJLee
   * - Zaky Hermawan
     - No Affiliation
     - ZakyHermawan
     - quarkz99
     - zakyHermawan


Organizations are limited to three representatives within the group to maintain diversity.

Participation
-------------

There are several ways to participate:

* Join discussions on the `LLVM Discourse <https://discourse.llvm.org/>`_ forum, under the "Community" category.
* Engage in conversations on the LLVM Community Discord in the `#fusa-qual-wg <https://discord.com/channels/636084430946959380/1389362444169773117>`_ channel. Note: You need to join the community's `Discord chat server <https://llvm.org/docs/GettingInvolved.html#discord>`_ first.
* Join our monthly sync-up calls. Details on working sessions and meeting minutes are shared on the :doc:`GettingInvolved` page.
* Contribute ideas, feedback, or patches via GitHub, Discourse, or directly in working documents.

Contribution Principles
-----------------------

We understand that most members contribute in a limited capacity due to their primary responsibilities. This initiative is volunteer-driven, and we operate with the following shared principles:

* **Acknowledgement of limited bandwidth:** We recognize that no one is working full-time on this group, and participation will vary based on individual availability and priorities.
* **Small and consistent contributions are valuable:** We believe that steady ongoing contributions, even if minimal, are crucial for long-term success, as long as there is coordination and respect for each other's time. Even small contributions (e.g., a few hours per month) can significantly advance the group's goals and have an impact. 
* **Realistic progress expectations:** Given the voluntary nature and no full-time involvement, we expect our progress to be slow. This group was initiated in July 2025. Concrete outcomes in 1-2 years would be considered excellent for this type of cross-company and voluntary collaboration.
* **Respect for differing capacities:** We value every member’s engagement, whether large or small, often or sporadically, as it all contributes to the overall effort. Even contributions that may seem small, such as sharing an idea or pointing out a relevant resource, are meaningful and important.

However, we need a balance between flexibility, structure, and enough organization to move forward together. Members are expected to remain engaged through one or more of the following:

* Regular participation in meetings or asynchronous discussions.
* Contributions to qualification artifacts, methodologies, or documentation.
* Active involvement in at least one qualification-related task over the past year.

Membership Criteria
-------------------

Membership in the LLVM Qualification Group is intended for individuals with relevant experience or active engagement in qualification-related efforts. Categories include:

**Individual Contributors**

  * Experience in software/tool qualification (e.g., reliability, quality, safety, security); OR  
  * Active involvement in LLVM-related qualification efforts; OR  
  * Significant LLVM contributions related to qualification in the past year (code, discussion, resolving related challenges).

**Researchers**

  * Active research, publication, or development of methodologies, frameworks, or tools aimed at improving LLVM quality and reliability.

**Vendor Contacts**

  * Represent organizations building or using LLVM-based tools in safety-critical environments; OR  
  * Require involvement due to organizational role in qualification or compliance.

Nomination Process
------------------

Individuals may nominate themselves or be nominated by an existing member. Nominations should:

* Explain the nominee’s background and relevance to qualification efforts.
* Be submitted via this form: `Participant Introduction & Membership <https://forms.gle/cE1kHjqkKNtafUrD7>`_
* Be communicated to an active LLVM Qualification Group member (e.g., on the Discord channel).

Nominations are discussed within the group. If consensus is reached, the nominee is accepted. Otherwise, a majority vote will decide.

Membership Review
-----------------

To ensure the group remains active and focused, member participation will be reviewed every six months. Inactive members may be removed following this review.

Current Topics & Backlog
========================

Our working group is actively engaged in discussions about the project's
direction and tackling technical challenges. You can find our current 
discussions, challenges, and the project backlog in the following 
document: `Backlog <https://docs.google.com/document/d/10YZZ72ba09Ck_OiJaP9C4-7DeUiveaIKTE3IkaSKjzA/edit?usp=sharing>`_

This document serves as our central hub for all ongoing topics and will
be updated regularly to reflect our progress. We welcome your 
contributions and feedback.

Meeting Materials
=================

Agendas, meeting notes, and presentation slides for the sync-ups are shared to ensure transparency and continuity.

Upcoming and past meeting agendas, and meeting minutes are published in a dedicated thread
on the LLVM Discourse forum: `Meeting Agendas and Minutes <https://discourse.llvm.org/t/llvm-qualification-wg-sync-ups-meeting-minutes/87148>`_ 

Slides used to support discussions during sync-up meetings are stored in LLVM's GitHub repository.

Available slides:

* `September 2025 <https://docs.google.com/presentation/d/1SZAE-QHfJED6CxJCCtBkPDxcw7XU9ORX54TJyXe1ppc/edit?usp=sharing>`_
* `August 2025 <https://docs.google.com/presentation/d/1K8GWoRm8ZAeyyGvTeV5f-sMOhMr7WHiEk6_Nm5Fk10o/edit?usp=sharing>`_
* `July 2025 <https://docs.google.com/presentation/d/1ktURe9qz5ggbdOQYK-2ISpiC18B-Y_35WvGyAnnxEpw/edit?usp=sharing>`_
* (add future entries here)

AI Transcription Policy
=======================

Objective
---------

The LLVM Qualification Group may enable AI auto-transcription (currently using Gemini) during sync-up calls in order to:

* Make complex discussions easier to follow.
* Reduce the effort of manual note-taking.
* Support inclusivity for participants who are not native English speakers.

Usage
-----

The purpose of auto-transcripts is to:

* Ensure participants can remain engaged during the sync-up meeting (particularly helpful for non-native English speakers or when audio clarity is limited).
* Serve as an aid for preparing the meeting minutes that are published on Discourse:
  `Meeting Agendas and Minutes <https://discourse.llvm.org/t/llvm-qualification-wg-sync-ups-meeting-minutes/87148>`_

Additional safeguards include:

* Transcript files are private to the note-taker(s) and never circulated to attendees or the public.
* Transcript files are permanently deleted once the minutes are posted.
* The meeting chair or scribe remains responsible for reviewing the transcript, ensuring accuracy, and editing out sensitive details in the official minutes.

Vendor Retention
----------------

Long-term storage or model-training settings are disabled on the account used for organizing the working group calls.

However, according to Google’s Gemini documentation, even with all history features disabled, voice and transcript data may be retained on Google’s servers for up to ~72 hours for the purpose of "*keeping Gemini safe and secure, including with help from human reviewers*" before deletion.

This retention period cannot currently be shortened.

Consent
-------

* At the start of each sync-up, participants will be asked if they are comfortable with enabling auto-transcription.
* If any participant objects, auto-transcription will be disabled for that meeting.
* Participants may also request at any point that parts of the discussion not be transcribed.

Recordings
----------

* Meetings are not recorded by default.
* Exceptions are made only when explicit approval from attendees is obtained (e.g., for a special-hosted demo).

Transparency & Feedback
-----------------------

We want to ensure this practice remains transparent and comfortable for everyone. If any group members have concerns (e.g., about names appearing in transcripts or minutes), they are encouraged to raise them on Discourse or Discord so they can be addressed.

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
