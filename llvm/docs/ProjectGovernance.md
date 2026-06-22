# Project Governance

```{contents}
:local:
```

## Area Teams

### Role and Responsibilities

*Area teams* have three main responsibilities.

First, they are responsible for electing from among themselves a team secretary
who will take notes of any team meetings and a team chair who facilitates team
meetings and represents the team on the *project council*.

Second, *area teams* are responsible for maintaining an up-to-date and
comprehensive list of maintainers for their area of the project. They can
nominate any individual they deem appropriate as maintainer of any area they
are responsible for. The role of maintainer remains a volunteer role, and any
individual can accept, decline, or resign the role for themselves as they feel
appropriate.

Finally, *area teams* are responsible for facilitating decision making for
their area of the project. Facilitating decision making can take any number of
forms ranging from contributing to RFC discussions, helping mediate
disagreements.

*Area teams* should prepare a meeting agenda by collecting all the active RFCs
in the community or significant disagreements in pull requests. During the team
meeting, the *area team* should try to identify actionable next steps or
information to gather so the RFC or pull request can proceed. An *area team*
may escalate to the *project council* as needed.

When acting to facilitate decision making the *area team* should act as a
mediator between different perspectives helping find common ground and
recognizing that decisions need not be binary. The *area team* should seek to
find the best solution to the framed problem, which may not be any of the
proposed alternatives. If agreement cannot be reached, the *area team* may act
as the final decision maker. In that capacity decisions of an *area team* are
considered final, but can be overruled by a 2/3 majority vote of the project
council or the *area team* itself revisiting the issue. If an *area team*
cannot reach consensus, it may request the *project council* to resolve the
disagreement.

A fast "no" is often a better outcome than an indefinite "maybe". In
recognition of that, an area team, when acting as the facilitator of decision
making, will publicly communicate a timeline for discussion and decision
making. The *area team* will communicate when a topic is on the agenda for a
meeting with sufficient notice for relevant parties to participate.

*Area teams* are not intended to be direction setters or primary maintainers of
their areas, although individuals on an *area team* may fulfill that role
separately. The area team's role is as a steward and moderator ensuring the
health and smooth operation of the area.

### Elections and Composition

To be a *voting contributor* an individual must be a member of the LLVM GitHub
Organization, and either have a public email address on their GitHub profile or
have made a commit to the LLVM project using a public email address. The email
address on the GitHub public profile or retrieved via commit metadata will be
used for all election-related communication.

Each *area team* will have an odd number of members with a minimum of three (3)
members and a maximum of nine (9) elected by the *voting contributors*.
Candidates for *area teams* must be a voting contributor and self-nominated. An
individual cannot serve on two *area teams*. Members of an *area team* are
elected for 1 year terms.

An *area team* with less than nine members may increase its size up to nine
members with a majority vote. The *area team* may then appoint members to fill
any vacancies as normal. If at the beginning of an election there are
insufficient candidates to fill all vacancies on an area team, the team size
will decrease to the largest odd number that all the candidates can fill. If
less than three candidates run for election for an *area team* the project
council will either recruit members or disband the team.

The *area team* will take an active role in identifying potential candidates to
join the *area team* in future terms. In this capacity, the *area team* should
keep a focus on growth and development of contributors in the community, and
the community values promoting diversity and inclusivity.

Elections for area teams occur in January of each year. A two week long
nomination period begins the second Monday in January. During the nomination
period any *voting contributor* can nominate themselves or another *voting
contributor* to run for any one area team. No individual can run for more than
one area team in a single election. An individual nominated for more than one
area team will be responsible for choosing which team they want to run for.
Nominations will be recorded publicly for community visibility. Unsuccessful
results in an election do not impact nomination eligibility in subsequent
elections.

Voting begins the fourth Monday in January and continues for 2 weeks. Election
results will be announced no later than two days after voting closes. The term
of the newly elected area team begins the first Monday in March. Each area team
will meet during the first week in March to elect from themselves the team
secretary and chair to re-constitute the project council.

### Vacancies

A member of an *area team* can resign at any time. As life can sometimes happen
unexpectedly, a member of an *area team* may be unable to fulfill their duties or
resign. In that case, a majority of the remaining *area team* may vote to declare
the member removed in absentia after a 90-day absence.

If someone resigns or is otherwise removed from an *area team*, the remaining
members of the *area team* may appoint a replacement to serve the remainder of
the term through any process they choose.

### Active Area Teams

There are currently four *area teams*:

- **LLVM** - Covering `llvm` source area.
- **Clang** - Covering `clang` source area.
- **MLIR** - Covering `mlir` source area.
- **Infrastructure** - Covering project-wide automation and other infrastructure.

## Project Council

### Role and Responsibilities

The *project council* is composed of the chair from each of the *area teams*.

The *project council* has a mandate to:

- Prioritize the long term health of the LLVM project and community.
- Shape the community to be accessible, inclusive, and sustainable.
- Maintain the relationship between the LLVM Community and the LLVM Foundation.
- Assist *area teams* in identifying and growing community leaders.
- Facilitate seeking consensus among the LLVM Community and *area teams*.
- Act as, or delegate to, an *area team* for all issues that are not covered by an area team, or span across multiple project areas.
- As a last resort, act as the final decision maker on debates.

The *project council* will elect from among themselves a secretary who will
take notes of all meetings, a chair who facilitates meetings, and a liaison to
the LLVM Foundation to manage the relationship between the *project council*
and the LLVM Foundation.

Representatives to the *project council* are also term limited. An individual
may not serve on the *project council* for more than two consecutive terms.
This limit may also be waived by the *project council* if and only if the
respective team is unable to produce a different representative.

The *project council* has the power to form and dissolve *area teams*. Forming
an *area team* requires a majority vote. Any changes to the *area team*
structures must be publicly disclosed including the motivation for the changes.
Dissolving an area team, or altering the boundaries of an *area team* requires
a consenting vote of the chair of the area team(s) being altered and a majority
vote of the *project council*.

## Governance Meetings

Each *area team* and the *project council* should have scheduled public
meetings. The date of the scheduled meetings should be on the LLVM Community
Calendar. The calendar invite will have a link to a public meeting agenda. The
teams may have non-public meetings for discussion, deliberation, planning or
other purposes. The team may cancel a meeting if no items are on the agenda or
to accommodate member schedules (holidays, personal time, etc).

Notes from all *area team* and *project council* meetings will be publicly
posted. Notes will exclude reference to any private information, or information
that otherwise needs to be confidential.

## Current Composition (2026)

### Project Council

- **Chair:** Aaron Ballman (@AaronBallman) - representing Clang

- **Secretary:** Alex Zinenko (@ftynse) - representing MLIR

- **Members:**

  - Nikita Popov (@nikic) - representing LLVM
  - Reid Kleckner (@rnk) - representing Infrastructure

### Area Teams

- **LLVM Area Team**

  - Nikita Popov (**Chair**)
  - Matt Arsenault
  - Florian Hahn

- **Clang Area Team**

  - Aaron Ballman (**Chair**)
  - Eli Friedman
  - Erich Keane
  - Corentin Jabot
  - Shafik Yaghmour

- **MLIR Area Team**

  - Alex Zinenko (**Chair**)
  - Renato Golin
  - Matthias Springer

- **Infrastructure Area Team**

  - Reid Kleckner (**Chair**)
  - Petr Hosek
  - David Blaikie

### Next Election Cycle

- **January 11, 2027:** Nominations begin.
- **January 25, 2027:** Nominations close and voting begins.
- **February 8, 2027:** Voting closes.
- **February 10, 2027:** Election results announced no later than this date.
- **March 1, 2027:** New area team terms begin.
- **March 1-5, 2027:** Area teams meet to elect chairs and secretaries.

## Meetings and Contact Information

### Project Council

- **Meetings:** First Wednesday of each month at 9:00 AM PT / 4:00 PM UTC / 6:00 PM CET.
- **Contact:** Tag `@project-council` on [LLVM Discourse](https://discourse.llvm.org/).

### Area Teams

- **LLVM Area Team**

  - **Meetings:** Bi-weekly on Wednesdays at 16:00-17:00 CET.
  - **Contact:** Tag `@llvm-area-team` on [LLVM Discourse](https://discourse.llvm.org/).

- **Clang Area Team**

  - **Meetings:** Closed administrative meeting bi-weekly on Thursdays at 10:00 AM PT. Open meetings are scheduled flexibly.
  - **Contact:** Tag `@clang-area-team` on [LLVM Discourse](https://discourse.llvm.org/).

- **MLIR Area Team**

  - **Meetings:** Announced on Discourse (not regularly scheduled at a fixed interval).
  - **Contact:** Tag `@mlir-area-team` on [LLVM Discourse](https://discourse.llvm.org/).

- **Infrastructure Area Team**

  - **Meetings:** Bi-weekly on Thursdays at 9:30 AM PT.
  - **Contact:** Tag `@infrastructure-area-team` on [LLVM Discourse](https://discourse.llvm.org/).

All meetings are listed on the [LLVM Community Calendar](https://calendar.google.com/calendar/u/0/embed?src=calendar@llvm.org).

