===========================================
LLVM Loop Terminology (and Canonical Forms)
===========================================

.. contents::
   :local:

Introduction
============

Loops are a core concept in any optimizer.  This page spells out some
of the common terminology used within LLVM code to describe loop
structures.

First, let's start with the basics.  In LLVM, a Loop is a cycle within
the control flow graph (CFG) where there exists one block (the loop
header block) which dominates all other blocks within the cycle.

Note that there are some important implications of this definition:

* Not all cycles are loops.  There exist cycles that do not meet the
  dominance requirement and such are not considered loops.  

* Loops can contain non-loop cycles and non-loop cycles may contain
  loops.  Loops may also contain sub-loops.

* Given the use of dominance in the definition, all loops are
  statically reachable from the entry of the function.  

* Every loop must have a header block, and some set of predecessors
  outside the loop.  A loop is allowed to be statically infinite, so
  there need not be any exiting edges.

* Any two loops are either fully disjoint (no intersecting blocks), or
  one must be a sub-loop of the other.

A loop may have an arbitrary number of exits, both explicit (via
control flow) and implicit (via throwing calls which transfer control
out of the containing function).  There is no special requirement on
the form or structure of exit blocks (the block outside the loop which
is branched to).  They may have multiple predecessors, phis, etc...

Key Terminology
===============

Header Block - The basic block which dominates all other blocks
contained within the loop.  As such, it is the first one executed if
the loop executes at all.  Note that a block can be the header of
two separate loops at the same time, but only if one is a sub-loop
of the other.

Exiting Block - A basic block contained within a given loop which has
at least one successor outside of the loop and one successor inside the
loop.  (The latter is required for the block to be contained within the
cycle which makes up the loop.)  That is, it has a successor which is
an Exit Block.  

Exit Block - A basic block outside of the associated loop which has a
predecessor inside the loop.  That is, it has a predecessor which is
an Exiting Block.

Latch Block - A basic block within the loop whose successors include
the header block of the loop.  Thus, a latch is a source of backedge.
A loop may have multiple latch blocks.  A latch block may be either
conditional or unconditional.

Backedge(s) - The edge(s) in the CFG from latch blocks to the header
block.  Note that there can be multiple such edges, and even multiple
such edges leaving a single latch block.  

Loop Predecessor -  The predecessor blocks of the loop header which
are not contained by the loop itself.  These are the only blocks
through which execution can enter the loop.  When used in the
singular form implies that there is only one such unique block. 

Preheader Block - A preheader is a (singular) loop predecessor which
ends in an unconditional transfer of control to the loop header.  Note
that not all loops have such blocks.

Backedge Taken Count - The number of times the backedge will execute
before some interesting event happens.  Commonly used without
qualification of the event as a shorthand for when some exiting block
branches to some exit block. May be zero, or not statically computable.

Iteration Count - The number of times the header will execute before
some interesting event happens.  Commonly used without qualification to
refer to the iteration count at which the loop exits.  Will always be
one greater than the backedge taken count.  *Warning*: Preceding
statement is true in the *integer domain*; if you're dealing with fixed
width integers (such as LLVM Values or SCEVs), you need to be cautious
of overflow when converting one to the other.

It's important to note that the same basic block can play multiple
roles in the same loop, or in different loops at once.  For example, a
single block can be the header for two nested loops at once, while
also being an exiting block for the inner one only, and an exit block
for a sibling loop.  Example:

.. code-block:: C

  while (..) {
    for (..) {}
    do {
      do {
        // <-- block of interest
        if (exit) break;
      } while (..);
    } while (..)
  }

LoopInfo
========

LoopInfo is the core analysis for obtaining information about loops.
There are few key implications of the definitions given above which
are important for working successfully with this interface.

* LoopInfo does not contain information about non-loop cycles.  As a
  result, it is not suitable for any algorithm which requires complete
  cycle detection for correctness.

* LoopInfo provides an interface for enumerating all top level loops
  (e.g. those not contained in any other loop).  From there, you may
  walk the tree of sub-loops rooted in that top level loop.

* Loops which become statically unreachable during optimization *must*
  be removed from LoopInfo. If this can not be done for some reason,
  then the optimization is *required* to preserve the static
  reachability of the loop.
  

Loop Simplify Form
==================

TBD


Loop Closed SSA (LCSSA)
=======================

TBD

"More Canonical" Loops
======================

TBD
