.. _amdgpu-memmodel:

=====================
 AMDGPU Memory Model
=====================

.. contents::
   :local:

Introduction
============

The :ref:`LLVM memory model<memmodel>` provides broad guarantees that are
sufficient to implement inter-thread communication via memory. But in most
communication patterns, not all memory accesses performed by a thread need to be
exposed to other threads. Even when they do need to be exposed, not all threads
may need to observe these memory accesses. This document describes the *AMDGPU
memory model* that allows the user to control how the side-effects of memory
accesses are propagated across threads. The programmer expresses this using
**new intrinsics and metadata** as described below, and the implementation can
then choose a more efficient mechanism to complete them, such as the cache
policy bits in an AMDGPU device.

The AMDGPU memory model allows executions that are not allowed by the LLVM
memory model. At the same time, a simple mapping can be used to implement these
new intrinsics and metadata using operations defined in the default LLVM memory
model. Thus, **there exists a safe-by-default implementation** that produces
executions that are valid in both models.

Terminology
===========

Memory Accesses
  Operations that read or write locations in memory are termed as *memory
  accesses*. Typical examples are ``load``, ``store`` and atomic instructions,
  as well as many intrinsics.

Synchronization Operations
  Synchronization operations control how the side-effects of memory accesses are
  propagated in the system. Typical examples are atomic operations (including
  fences) with at least ``release`` or ``acquire`` ordering.

.. _amdgpu-scopes:

Scopes
======

A *scope* is an abstract description of sets of memory accesses and
synchronization operations in a multi-threaded execution environment. Each such
set is called an *instance* of that scope, or a *scope instance* for short.

- Each memory access or synchronization operation belongs to at most one
  instance of every scope defined by the target.
- When an operation ``X`` specifies a scope ``S``, it indicates the instance of
  ``S`` that contains ``X``. This scope instance is also termed as *X's instance
  of scope S*, or just *X's scope instance* when ``S`` is implied by the
  context.
- When an operation does not specify a scope, it indicates the *system*
  scope defined below.

LLVM scopes
-----------

The LLVM Language Reference defines the following :ref:`scopes<syncscope>`:

*system scope* (empty string "")
  There exists a single instance of this scope that contains the memory accesses
  and synchronization operations performed by all threads.

"singlethread" scope
  Each thread corresponds to a "singlethread" scope instance that contains the
  memory accesses and synchronization operations performed by that thread.

AMDGPU scopes
-------------

The AMDGPU backend further refines the LLVM scopes with the following
target-defined scopes and constraints:

- *system scope* (same as LLVM)
- "agent" scope
- "cluster" scope
- "workgroup" scope
- "wavefront" scope
- "singlethread" scope (same as LLVM)

These are arranged from largest scope (*system scope*) to smallest scope
("singlethread").

- Every instance ``X`` of some scope ``S1`` other than "singlethread" scope is
  partitioned by the scope ``S2`` one level below it. Each subset defined by this
  partition is an instance of ``S2`` and is called a *subscope instance* of ``X``.
- It follows that if two scope instances ``X`` and ``Y`` intersect, then their
  intersection is the smaller of ``X`` and ``Y``.
- A scope ``S1`` is a *subscope* of a scope ``S2`` if every instance of ``S1``
  is a subscope instance of some instance of ``S2``.

**Inclusive Scopes**: Two operations ``X`` and ``Y`` are said to have *inclusive
scopes* if the scope instance of each operation contains the other operation. In
that case, the *common scope instance* ``S'`` of ``X`` and ``Y`` is the
intersection of their scope instances. The scope corresponding to ``S'`` is also
termed as the *common scope* of ``X`` and ``Y``.

Availability and Visibility
===========================

The AMDGPU memory model is built on top of the :ref:`happens-before<memmodel>`
order defined by the LLVM memory model. But when one of the new intrinsics or
metadata is used, **happens-before by itself is not sufficient** to describe its
observable effects. Instead, the AMDGPU model uses *availability* and
*visibility* to describe how the side-effects of these operations propagate to
other threads.

Availability determines how *far* the side-effects of a write have been
forwarded in the system relative to that write. Visibility determines how
*close* the side-effects of the same write have reached relative to an observer
operation (typically a read).

The AMDGPU memory model *does not change the structure of happens-before*, but
changes the rules that determine how operations may observe the side-effects of
other operations that *happen-before* them.

Consider a write ``W`` that ``happens-before`` a read ``R`` to the same address:

- ``R`` can potentially observe the side-effects of ``W`` **only if W is
  visible** to ``R``.
- ``W`` can potentially be visible to ``R`` **only if W is first made
  available** to ``R``.

The instructions used in the default LLVM memory model automatically satisfy
these necessary conditions, and hence they can be explained using the rules from
either memory model. But the new intrinsics and metadata *opt out* of the LLVM
memory model, and can only be explained using the AMDGPU memory model.

.. _amdgpu-store-available:

store-available
---------------

.. code-block:: llvm

   @llvm.amdgcn.av.global.store.b128(ptr, value, scope)
   store atomic [syncscope("<target-scope>")]
   atomicrmw    [syncscope("<target-scope>")]
   cmpxchg      [syncscope("<target-scope>")]

The ``@llvm.amdgcn.av.global.store.b128`` intrinsic performs a non-atomic
*store-available* operation on ``ptr`` with scope ``scope``.

An atomic operation that results in a store operation is a *store-available*
operation with scope ``syncscope``.

.. _amdgpu-load-visible:

load-visible
------------

.. code-block:: llvm

   @llvm.amdgcn.av.global.load.b128(ptr, scope)
   load atomic  [syncscope("<target-scope>")]
   atomicrmw    [syncscope("<target-scope>")]
   cmpxchg      [syncscope("<target-scope>")]

The ``@llvm.amdgcn.av.global.load.b128`` intrinsic performs a non-atomic
*load-visible* operation on ``ptr`` with scope ``scope``.

An atomic operation that results in a read operation is a *load-visible*
operation with scope ``syncscope``.

.. note::

   Metadata cannot be used to model this using ordinary load/store operations,
   because the scope is necessary for correctness. In a hypothetical operation
   like this:

   .. code-block:: llvm

      store ptr, data, !mmra !{!"amdgcn-av", !"workgroup"}

   If the metadata is dropped or ignored, there is no guarantee that the store
   will become available at the intended scope. In implementation terms, the
   store may be completed at a nearer cache than the one required for that
   scope. A corresponding *load-visible* that does not access the same near
   cache will fail to observe this store.

MakeAvailable and MakeVisible
-----------------------------

.. code-block:: llvm

   store atomic [syncscope("<target-scope>")] <ordering> [, !mmra !{!"amdgcn-av", !"none"}]
   load atomic  [syncscope("<target-scope>")] <ordering> [, !mmra !{!"amdgcn-av", !"none"}]
   atomicrmw    [syncscope("<target-scope>")] <ordering> [, !mmra !{!"amdgcn-av", !"none"}]
   cmpxchg      [syncscope("<target-scope>")] <ordering> [, !mmra !{!"amdgcn-av", !"none"}]
   fence        [syncscope("<target-scope>")] <ordering> [, !mmra !{!"amdgcn-av", !"none"}]

A synchronization operation with at least ``release`` ordering is a
``MakeAvailable`` operation with scope ``syncscope``, if it is not marked as
``!{!"amdgcn-av", !"none"}``.

A synchronization operation with at least ``acquire`` ordering is a
``MakeVisible`` operation with scope ``syncscope``, if it is not marked as
``!{!"amdgcn-av", !"none"}``.

These operations include ``MakeVisible`` and ``MakeAvailable`` operations by
default. The presence of this metadata removes this ability and essentially
creates *non-av* ordering operations, i.e., ordering operations that do not
establish availability or visibility.

For an atomic operation which itself accesses memory (e.g., ``store atomic``
or ``load atomic``), the metadata does not affect the availability or the
visibility of the access performed by the operation itself. It only affects
the ordering of other memory accesses.

.. code-block:: llvm

   ; This includes the following operations:
   ; - The atomic store at "agent" scope,
   ; - A store-available operation at "agent" scope on `ptr`,
   ; - A `MakeAvailable` operation at "agent" scope that affects previous memory accesses.
   store atomic syncscope("agent") release ptr

   ; This includes the following operations:
   ; - The atomic store at "agent" scope,
   ; - A store-available operation at "agent" scope on `ptr`.
   ; Noteably, it does not include a `MakeAvailable` operation on other memory accesses.
   store atomic syncscope("agent") release ptr, !mmra !{!"amdgcn-av", !"none"}

Ordering
========

.. note::

   **TODO:** These ordering operations affect all address spaces. We need to
   eventually make that a parameter similar to the storage class parameter on
   operations and orders in Vulkan.

Availability Operation
----------------------

An operation ``X`` is an *availability operation* on a write ``W`` if one of the
following holds:

- ``X`` is ``W`` itself, and ``W`` is a *store-available* operation, or,
- ``X`` is a ``MakeAvailable`` operation that follows ``W`` in program order,
  or,
- ``X`` is a ``MakeAvailable`` operation whose scope instance includes ``W``,
  and there is an availability operation ``Z`` on ``W`` such that:

  - ``Z`` happens-before ``X``, and,
  - ``Z``'s scope instance includes ``X``.

Then ``X`` makes ``W`` available in its own scope instance ``S`` and every
subscope instance of ``S`` that also includes ``W``.

Visibility Operation
--------------------

An operation ``Y`` is a *visibility operation* on a write ``W`` if ``Y`` is a
*load-visible* operation to the same address, or a ``MakeVisible`` operation,
and one of the following holds:

- There exists an *availability* operation ``X`` on write ``W`` such that:

  - ``X`` happens-before ``Y``, and,
  - ``X`` and ``Y`` specify inclusive scopes.

  Then ``Y`` makes ``W`` visible in the common scope instance ``S`` of ``X`` and
  ``Y``, and every subscope instance of ``S`` that includes ``Y``.

- There exists a *visibility* operation ``X`` on write ``W`` such that:

  - ``X`` happens-before ``Y``, and,
  - ``X`` makes ``W`` visible in a scope instance ``S1`` that includes ``Y``, and,
  - ``X`` is included in the scope instance ``S2`` of ``Y``.

  Then ``Y`` makes ``W`` visible in the intersection ``S`` of ``S1`` and ``S2``,
  and every subscope instance of ``S`` that includes ``Y``.

Location Order
--------------

A write ``W`` is *location-ordered* before an access ``Y`` to the same address
if ``W`` is program-ordered before ``Y``.

A write ``W`` is *location-ordered* before a write ``W1`` to the same address if
there exists an availability operation ``Z`` on ``W`` such that:

- ``Z`` happens-before ``W1``, and,
- ``W1`` is included in ``Z``'s scope instance.

A write ``W`` is *location-ordered* before a read ``R`` to the same address if
there exists a visibility operation ``Z`` on write ``W`` such that:

- ``Z`` is ``R`` itself, or,
- ``Z`` precedes ``R`` in program order.

The AMDGPU memory model overrides the definition of each byte in the
:ref:`LLVM memory model<memmodel>` as follows.

Every (defined) read operation ``R`` reads a series of bytes written by
(defined) write operations. Each initialized global is assumed to have an
initial *system scoped* atomic write operation that is *location-ordered* before
any other read or write to that same location.

For each byte of a read ``R``, ``R`` may see any write to the same byte, except:

- If a write ``W1`` is *location-ordered* before a write ``W2``, and ``W2`` is
  *location-ordered* before a read ``R``, then ``R`` may not see ``W1``.
- If a read ``R`` happens-before a write ``W3``, then ``R`` may not see ``W3``.

The value returned by ``R`` is then defined as follows:

- If no write is *location-ordered* before a read ``R``, then ``R`` returns
  ``undef``.
- Otherwise if the set consisting of ``R`` and all writes that ``R`` may see
  contains only atomic operations with inclusive scopes, then ``R`` returns the
  value written by one of those writes.
- Otherwise, if ``R`` may see some write that is not *location-ordered* before
  ``R``, then ``R`` returns ``undef``.
- Otherwise, if ``R`` may see exactly one write ``W``, then ``R`` returns the
  value written by ``W``.
- Otherwise, ``R`` returns ``undef``.

Properties
==========

.. tip::

   This section is informational.

The following properties follow from the definitions above:

1. **Happens-before is necessary for location-order.** A write ``W`` is
   *location-ordered* before a read ``R`` only if ``W`` happens-before ``R``.
   This follows from the definition of availability and visibility operations,
   which always require a happens-before link with the preceding operation in
   the chain.

2. **A write cannot be made available in a scope that does not contain it.** The
   definition of an availability operation ``X`` requires that ``X``'s scope
   instance includes ``W`` as a precondition. Since every scope instance that
   includes ``X`` also includes ``W``, availability cannot reach a scope
   instance that excludes ``W``. In other words, availability can only "expand
   outwards" into progressively larger scopes.

3. **Visibility is bounded by availability.** When a write is available in a
   scope instance, it can be made visible in that scope instance by a visibility
   operation with the corresponding scope. Subsequent ``MakeVisible`` operations
   make that write visible into narrower scope instances towards the observer.

4. **A write can be made visible in a scope instance that does not contain it.**
   The definition of a *visibility operation* anchors scope instances to the
   observer (``Y``), not to the original write. The only precondition is that the
   write must already be visible or available in the scope instance of the
   visibility operation.

5. **Availability and visibility chains.** For a write ``W`` to be visible to a
   read ``R`` anywhere in the system, the sufficient condition is a chain of
   happens-before edges that include availability and visibility operations with
   inclusive scopes. It is not necessary that ``W`` and ``R`` themselves have
   inclusive scopes. Each link in the availability and visibility definitions
   only checks the immediate predecessor, so intermediate operations can bridge
   scope gaps that the endpoints cannot satisfy directly. Such a chain passes
   through at least one availability operation and at least one visibility
   operation with inclusive scopes, such that their common scope includes both
   ``W`` and ``R``.

.. _amdgcn-av-vulkan:

The Vulkan Memory Model
=======================

The AMDGPU memory model draws heavily on the Vulkan memory model. In
particular, the following instructions are equivalent.

.. csv-table::
   :header: "LLVM", "SPIRV", "Available/Visible Semantics"
   :widths: 20, 20, 60

   "``load``", "``OpLoad NonPrivatePointer``", "\-"
   "``load-visible``", "``OpLoad NonPrivatePointer``", "``MakePointerVisible``"
   "``store``", "``OpStore NonPrivatePointer``", "\-"
   "``store-available``", "``OpStore NonPrivatePointer``", "``MakePointerAvailable``"
   "``load atomic``", "``OpAtomicLoad``", "``MakePointerVisible``. Also ``MakeVisible`` when order is at least ``acquire``."
   "``load atomic !{!""amdgcn-av"", !""none""}``", "``OpAtomicLoad``", "``MakePointerVisible``"
   "``store atomic``", "``OpAtomicStore``", "``MakePointerAvailable``. Also ``MakeAvailable`` when order is at least ``release``."
   "``store atomic !{!""amdgcn-av"", !""none""}``", "``OpAtomicStore``", "``MakePointerAvailable``"
   "``fence``", "``OpMemoryBarrier``", "``MakeAvailable`` when order is at least ``release``, and ``MakeVisible`` when order is at least ``acquire``."
   "``fence !{!""amdgcn-av"", !""none""}``", "``OpMemoryBarrier``", "\-"

.. note::

   The above table is representative only, and does not aim to be exhaustive. In
   particular, it does not list composite atomic operations like ``rmw`` and
   ``cmpxchg``. The ordering and semantics of these operations can be determined
   by combining suitable rules such as:

   - "``MakeAvailable`` if the order is at least ``release``, and the operation
     results in a store",
   - "Only if it is not marked as ``!{!"amdgcn-av", !"none"}``", etc.

The AMDGPU memory model is a special case of the Vulkan memory model:

a. LLVM fence/atomic ordering operations have ``MakeAvailable`` /
   ``MakeVisible`` semantics by default, thus satisfying the availability and
   visibility chains required in Vulkan. Hence the LLVM memory model is a
   "strong" subset of the Vulkan memory model.
b. The AMDGPU memory model described here makes it possible to opt-out of the
   default ``MakeAvailable`` and ``MakeVisible`` semantics, and instead specify
   it on select places including the new *load-visible* and *store-available*
   operations. This expands the subset of the Vulkan memory model that can now
   be expressed in LLVM IR.
