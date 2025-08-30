..
  Purpose: Track and document the ongoing discussions and backlog for the LLVM Qualification Group.
  Author: Carlos Andres Ramirez Catano
  Last updated: 2025-08-30 by Carlos Andres Ramirez Catano

================================================
Current Topics & Backlog for Qualification Group
================================================

This document serves as the central hub for our working group's current
activities, including ongoing discussions, key challenges, and our 
prioritized backlog.

This is a living document and will be updated as our work evolves.

---

Ongoing Discussions
===================

This section outlines topics the working group is currently analyzing or
deliberating on.

* **Compiler Qualification Language & Standard:**
    * **Topic:** Defining the initial scope for Clang qualification.
    * **Challenge:** Deciding between C and C++ for the first qualification target.
    * **Sub-topics:**
        * Which C standard (e.g., C11, C23) is most suitable?
        * Creating an open-source conformance test suite for C seems unfeasible. Any alternatives? Is this a requirement?
        * Would it suffice to do testing using Alive2?
        * Criteria for deciding the hardware architecture (target)
        * Criteria for deciding specific compiler use case and options enabled
        * Qualification standards usually require a specific, frozen tool version and a controlled development process to be qualified for a particular use case. The continuous development model of most open-source projects doesn't align well with this requirement. Would it be possible to have a Long Term Support (LTS) release of LLVM?

* **Qualification Standard and Tool Confidence Level:**
    * **Topic:** Defining the initial safety framework scope for qualification (ISO26262, IEC61508, etc).
    * **Challenge:** We need enough members for each framework we want to include in scope.
    * **Sub-topics:**
        * Shall we initially focus on a specific standard, or provide templates for different standards so members can work on their own implementations?
        * Criteria for choosing tool confidence level

* **Toolchain Integration with Open Source Tools:**
    * **Topic:** Investigating how to demonstrate confidence in LLVM's development process.
    * **Challenge:** Bridging the gap between traditional safety-critical processes and open-source development models.
    * **Sub-topics:**
        * Can we leverage existing test suites?
        * Identifying a path to continuous validation within a CI/CD pipeline.
        * Integrating Alive2 tests
        * Integrating a human centric approach (tests and documentation of the LLVM's development process)

---

Prioritized Backlog
===================

This is a list of tasks and topics we have agreed to address in the near future, ranked by priority.

1.  **Define Language & Standard for Qualification:** Finalize the decision on whether to qualify a specific version of the C or C++ standard.
2.  **Research C Conformance Test Suites:** Identify and evaluate existing open-source or commercial C conformance test suites.
3.  **Propose a Community RFC:** Draft a Request for Comments (RFC) to the LLVM community to formalize our group's goals and request support.
4.  **Create a Public Wiki:** Establish a public-facing wiki or documentation site to track our progress and share findings.

---

Future Ideas & Unsorted Topics
===============================

This section is a parking lot for new ideas or topics that have been raised but are not yet being actively discussed or prioritized.

* Investigating qualification for the LLVM linker (lld).
* Exploring the use of static analysis tools (e.g., clang-tidy, Malleus) in a safety-critical context.
* Documenting the process for a minimal, safety-critical build of Clang.

---
