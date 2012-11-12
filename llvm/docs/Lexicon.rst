.. _lexicon:

================
The LLVM Lexicon
================

.. note::

    This document is a work in progress!

Definitions
===========

A
-

**ADCE**
    Aggressive Dead Code Elimination

B
-

**BB Vectorization**
    Basic Block Vectorization

**BURS**
    Bottom Up Rewriting System --- A method of instruction selection for code
    generation.  An example is the `BURG
    <http://www.program-transformation.org/Transform/BURG>`_ tool.

C
-

**CSE**
    Common Subexpression Elimination. An optimization that removes common
    subexpression compuation. For example ``(a+b)*(a+b)`` has two subexpressions
    that are the same: ``(a+b)``. This optimization would perform the addition
    only once and then perform the multiply (but only if it's compulationally
    correct/safe).

D
-

**DAG**
    Directed Acyclic Graph

.. _derived pointer:
.. _derived pointers:

**Derived Pointer**
    A pointer to the interior of an object, such that a garbage collector is
    unable to use the pointer for reachability analysis. While a derived pointer
    is live, the corresponding object pointer must be kept in a root, otherwise
    the collector might free the referenced object. With copying collectors,
    derived pointers pose an additional hazard that they may be invalidated at
    any `safe point`_. This term is used in opposition to `object pointer`_.

**DSA**
    Data Structure Analysis

**DSE**
    Dead Store Elimination

F
-

**FCA**
    First Class Aggregate

G
-

**GC**
    Garbage Collection. The practice of using reachability analysis instead of
    explicit memory management to reclaim unused memory.

H
-

.. _heap:

**Heap**
    In garbage collection, the region of memory which is managed using
    reachability analysis.

I
-

**IPA**
    Inter-Procedural Analysis. Refers to any variety of code analysis that
    occurs between procedures, functions or compilation units (modules).

**IPO**
    Inter-Procedural Optimization. Refers to any variety of code optimization
    that occurs between procedures, functions or compilation units (modules).

**ISel**
    Instruction Selection

L
-

**LCSSA**
    Loop-Closed Static Single Assignment Form

**LICM**
    Loop Invariant Code Motion

**Load-VN**
    Load Value Numbering

**LTO**
    Link-Time Optimization

M
-

**MC**
    Machine Code

O
-
.. _object pointer:
.. _object pointers:

**Object Pointer**
    A pointer to an object such that the garbage collector is able to trace
    references contained within the object. This term is used in opposition to
    `derived pointer`_.

P
-

**PRE**
    Partial Redundancy Elimination

R
-

**RAUW**

    Replace All Uses With. The functions ``User::replaceUsesOfWith()``,
    ``Value::replaceAllUsesWith()``, and
    ``Constant::replaceUsesOfWithOnConstant()`` implement the replacement of one
    Value with another by iterating over its def/use chain and fixing up all of
    the pointers to point to the new value.  See
    also `def/use chains <ProgrammersManual.html#iterate_chains>`_.

**Reassociation**
    Rearranging associative expressions to promote better redundancy elimination
    and other optimization.  For example, changing ``(A+B-A)`` into ``(B+A-A)``,
    permitting it to be optimized into ``(B+0)`` then ``(B)``.

.. _roots:
.. _stack roots:

**Root**
    In garbage collection, a pointer variable lying outside of the `heap`_ from
    which the collector begins its reachability analysis. In the context of code
    generation, "root" almost always refers to a "stack root" --- a local or
    temporary variable within an executing function.

**RPO**
    Reverse postorder

S
-

.. _safe point:

**Safe Point**
    In garbage collection, it is necessary to identify `stack roots`_ so that
    reachability analysis may proceed. It may be infeasible to provide this
    information for every instruction, so instead the information may is
    calculated only at designated safe points. With a copying collector,
    `derived pointers`_ must not be retained across safe points and `object
    pointers`_ must be reloaded from stack roots.

**SDISel**
    Selection DAG Instruction Selection.

**SCC**
    Strongly Connected Component

**SCCP**
    Sparse Conditional Constant Propagation

**SRoA**
    Scalar Replacement of Aggregates

**SSA**
    Static Single Assignment

**Stack Map**
    In garbage collection, metadata emitted by the code generator which
    identifies `roots`_ within the stack frame of an executing function.

T
-

**TBAA**
    Type-Based Alias Analysis

