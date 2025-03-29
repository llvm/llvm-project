============================
LLVM Incident Response Guide
============================

Purpose
=======

The purpose of this document is to outline how a project administrator should respond to
malicious or unwanted content that appears on LLVM infrastructure.  This includes but
is not limited to: malicious code checked into the GitHub repository,  unauthorized access
to LLVM controlled servers, or compromise of community owned resources like buildbots
or GitHub Actions runners.

General Principles
==================

We trust our project administrators to use good judgement when responding to an incident,
so we want to avoid creating regulations or rules that will slow down or limit their ability to
quickly resolve it.  However, we do want to provide some general guidelines for admins
to follow during an incident, mainly to ensure that the problem and the steps taken to
resolve it are being communicated effectively.  Here is a checklist admins should follow
when addressing an issue:

1. Communicate the problem to another admin.
2. Decide on a short term solution to minimize the impact.
3. Communicate the solution to another admin.
4. Take action and implement the solution.
5. Notify the community of what was done.
6. Meet with one or more admins to discuss long-term solution.
7. Implement long-term solution.
8. Publish a retrospective for the community.

1. Communicate the problem to another admin
-------------------------------------------

It's important to let someone else know what is going on.  It can be an email,
slack, or Discord message, and you don't have to wait for a response before
taking action.

2. Decide on a short term solution to minimize the impact
---------------------------------------------------------

For a short-term solution the goal should be to protect the community or users from
being impacted by the incident.  An example of a short-term action would be to
block read access to the GitHub repository if something malicious was committed.

Ideally, multiple admins would discuss a solution together, but it's OK if one
admin comes up with the plan on their own if they can't get in touch with
someone else.

3. Communicate the short-term solution to another admin
-------------------------------------------------------

Make sure that someone else is aware of what changes will be made.  This is important in
case the admin making the changes goes offline and another admin needs to cover for
them.

4. Take action and implement the short-term solution
----------------------------------------------------

A single admin can do this on their own as long as they've communicated what they're doing to
someone else.  There are cases where waiting for confirmation could leave users or community
members at risk so admins should try to take action as quickly as possible.

5. Notify the community for what was done and why
-------------------------------------------------

This should be done in a Discourse post in the LLVM Project category.

6. Meet with other admins to discuss long-term solution
-------------------------------------------------------

Once the immediate risk has been eliminate, admins should meet together and discuss
a long-term solution.  Unlike the short-term solution, this conversation should be
done with two or more admins.  The discussion could also include key community members
or even the entire community.

7. Implement the long-term solution
-----------------------------------

Once there is consensus on a long-term solution, the admins should implement it.

8. Publish a retrospective for the community
--------------------------------------------

Once the problem is resolved, the admins should publish a retrospective about the incident
and decide on any changes that need to be made to prevent further incidents.
