.. _amdgpu-execution-synchronization:

================================
AMDGPU Execution Synchronization
================================

.. contents::
   :local:

.. _amdgpu-execution-synchronization-barriers:

This document covers different ways of synchronizing execution of threads on AMD GPUs.

.. note::

  This document is not exhaustive. There may be more ways of synchronizing execution
  that are not covered by this document.

Barriers
========

This section covers execution synchronization using barrier-style primitives.

.. _amdgpu-execution-synchronization-barriers-execution-model:

Execution Model
---------------

This section contains a formal execution model that can be used to model the behavior of
barriers on AMDGPU targets.

.. note::

  The barrier execution model is experimental and subject to change.

Threads can synchronize execution by performing barrier operations on barrier *objects* as described below:

* Each barrier *object* has the following state:

  * An unsigned positive integer *expected count*: counts the number of *arrive* operations
    expected for this barrier *object*.
  * An unsigned non-negative integer *arrive count*: counts the number of *arrive* operations
    already performed on this barrier *object*.

      * The initial value of *arrive count* is zero.
      * When an operation causes *arrive count* to be equal to *expected count*, the barrier is completed,
        and the *arrive count* is reset to zero.

* Barrier *objects* exist within a *scope* (see :ref:`amdgpu-amdhsa-llvm-sync-scopes-table`),
  and each instance of a barrier *object* can only be accessed by threads in the same *scope* instance.
* *Barrier-mutually-exclusive* is a symmetric relation between barrier *objects* that share resources
  in a way that restricts how a thread can use them at the same time.
* Barrier operations are performed on barrier *objects*. A barrier operation is a dynamic instance
  of one of the following:

  * Barrier *init*

    * Barrier *init* takes an additional unsigned positive integer argument *k*.
    * Sets the *expected count* of the *barrier object* to *k*.
    * Resets the *arrive count* of the *barrier object* to zero.

  * Barrier *join*.

    * Allow the thread that executes the operation to *wait* on a barrier *object*.

  * Barrier *drop*.

    * Decrements *expected count* of the barrier *object* by one.

  * Barrier *arrive*.

    * Increments the *arrive count* of the barrier *object* by one.
    * If supported, an additional argument to  *arrive* can also update the *expected count* of the
      barrier *object* before the *arrive count* is incremented;
      the new *expected count* cannot be less than or equal to the *arrive count*,
      otherwise the behavior is undefined.

  * Barrier *wait*.

    * Introduces execution dependencies between threads; this operation depends on
      other barrier operations to complete.

* Barrier modification operations are barrier operations that modify the barrier *object* state:

  * Barrier *init*.
  * Barrier *drop*.
  * Barrier *arrive*.

* *Thread-barrier-order<BO>* is the subset of *program-order* that only
  relates barrier operations performed on a barrier *object* ``BO``.
* All barrier modification operations on a barrier *object* ``BO`` occur in a strict total order called
  *barrier-modification-order<BO>*; it is the order in which ``BO`` observes barrier
  operations that change its state. For any valid *barrier-modification-order<BO>*, the
  following must be true:

  * Let ``A`` and ``B`` be two barrier modification operations where ``A -> B`` in
    *thread-barrier-order<BO>*, then ``A -> B`` is also in *barrier-modification-order<BO>*.
  * The first element in *barrier-modification-order<BO>* is always a barrier *init*, otherwise
    the behavior is undefined.

* *barrier-participates-in* relates barrier operations to the barrier *waits* that depend on them
  to complete. A barrier operation ``X`` *barrier-participates-in* a barrier *wait* ``W``
  if and only if all of the following is true:

  * ``X`` and ``W`` are both performed on the same barrier *object* ``BO``.
  * ``X`` is a barrier *arrive* or *drop* operation.
  * ``X`` does not *barrier-participate-in* another distinct barrier *wait* ``W'`` in the same thread as ``W``.
  * ``W -> X`` not in *thread-barrier-order<BO>*.
  * All dependent constraint and relations are satisfied as well. [0]_

* For the set ``S`` consisting of all barrier operations that *barrier-participate-in* a barrier *wait* ``W`` for some
  barrier *object* ``BO``:

  * The elements of ``S`` all exist in a continuous, uninterrupted interval of *barrier-modification-order<BO>*.
  * The *arrive count* of ``BO`` is zero before the first operation of ``S`` in *barrier-modification-order<BO>*.
  * The *arrive count* and *expected count* of ``BO`` are equal after the last operation of ``S`` in
    *barrier-modification-order<BO>*. The *arrive count* and *expected count* of ``BO`` cannot
    equal at any other point in ``S``.

* A barrier *join* ``J`` is *barrier-joined-before* a barrier operation ``X`` if and only if all
  of the following is true:

  * ``J -> X`` in *thread-barrier-order<BO>*.
  * ``X`` is not a barrier *join*.
  * There is no barrier *join* or *drop* ``JD`` where ``J -> JD -> X`` in *thread-barrier-order<BO>*.
  * There is no barrier *join* ``J'`` on a distinct barrier *object* ``BO'`` such that ``J -> J' -> X`` in
    *program-order*, and ``BO`` *barrier-mutually-exclusive* ``BO'``.

* A barrier operation ``A`` *barrier-executes-before* another barrier operation ``B`` if any of the
  following is true:

  * ``A -> B`` in *program-order*.
  * ``A -> B`` in *barrier-participates-in*.
  * ``A`` *barrier-executes-before* some barrier operation ``X``, and ``X``
    *barrier-executes-before* ``B``.

* *Barrier-executes-before* is consistent with *barrier-modification-order<BO>*
  for every barrier object ``BO``.
* For every barrier *drop* ``D`` performed on a barrier *object* ``BO``:

  * There is a barrier *join* ``J`` such that ``J -> D`` in *barrier-joined-before*;
    otherwise, the behavior is undefined.
  * ``D`` cannot cause the *expected count* of ``BO`` to become negative; otherwise, the behavior is undefined.

* For every pair of barrier *arrive* ``A`` and barrier *drop* ``D`` performed on a barrier *object*
  ``BO``, such that ``A -> D`` in *thread-barrier-order<BO>*, one of the following must be true:

  * ``A`` does not *barrier-participates-in* any barrier *wait*.
  * ``A`` *barrier-participates-in* at least one barrier *wait* ``W``
    such that  ``W -> D`` in *barrier-executes-before*.

* For every barrier *wait* ``W`` performed on a barrier *object* ``BO``:

  * There is a barrier *join* ``J`` such that ``J -> W`` in *barrier-joined-before*, and
    ``J`` must *barrier-executes-before* at least one operation ``X`` that
    *barrier-participates-in* ``W``; otherwise, the behavior is undefined.

* *barrier-phase-with* is a symmetric relation over barrier operations defined as the
  transitive closure of: *barrier-participates-in* and its inverse relation.
* For every barrier operation ``A`` that *barrier-participates-in* a barrier *wait* ``W`` on a barrier *object* ``BO``:

  * There is no barrier operation ``X`` on ``BO`` such that ``A -> X -> W`` in
    *barrier-executes-before*, and ``X`` *barrier-phase-with* a non-empty set of operations
    that does not include ``W``.

.. note::

  Barriers only synchronize execution and do not affect the visibility of memory operations between threads.
  Refer to the :ref:`execution barriers memory model<amdgpu-amdhsa-execution-barriers-memory-model>`
  to determine how to synchronize memory operations through *barrier-executes-before*.


.. [0] The definition of *barrier-participates-in* (in its current state) is non-deterministic and
       will be improved in the future: Within a valid execution, there may be multiple ways
       to build *barrier-participates-in*, however there is only one way to build it that also satisfies all
       other relations and constraints that depend on *barrier-participates-in* and relations derived from it.

Informational Notes
~~~~~~~~~~~~~~~~~~~

Informally, we can deduce from the above formal model that execution barriers behave as follows:

* *Barrier-executes-before* relates the dynamic instances of operations from different threads together.
  For example, if ``A -> B`` in *barrier-executes-before*, then the execution of ``A`` must complete
  before the execution of ``B`` can complete.

  * This property can also be combined with *program-order*. For example, let two (non-barrier) operations
    ``X`` and ``Y`` where ``X -> A`` and ``B -> Y`` in *program-order*, then we know that the execution
    of ``X`` completes before the execution of ``Y`` does.

* Barriers do not complete "out-of-thin-air"; a barrier *wait* ``W`` cannot depend on a barrier operation
  ``X`` to complete if ``W -> X`` in *barrier-executes-before*.
* It is undefined behavior to operate on an uninitialized barrier object.
* It is undefined behavior for a barrier *wait* to never complete.
* It is not mandatory to *drop* a barrier after *joining* it.
* A thread may not *arrive* and then *drop* a barrier *object* unless the barrier completes before the
  barrier *drop*. Incrementing the *arrive count* and decrementing the *expected count* directly
  after may cause undefined behavior.
* *Joining* a barrier is only useful if the thread will *wait* on that same barrier *object* later.

Barrier Implementations on AMDGPU Targets
-----------------------------------------

``s_barrier``
~~~~~~~~~~~~~

``s_barrier`` are the primary barrier implementation of AMD GPUs.

``s_barrier`` instructions can only be used to synchronize threads at a wavefront granularity.
``s_barrier`` instructions are convergent within a wave, and thus can only be performed
in wave-uniform control flow.

The ``s_barrier`` family of instructions is available in some form on all GFX targets,
and has evolved over time. The sub-sections below cover the capabilities offered by every major
iteration of this feature separately.

GFX6-11
+++++++

Targets from GFX6 through GFX11 included do not have the "split barrier" feature.
The barrier *arrive* and barrier *wait* operations **cannot** be performed independently
using ``s_barrier``.

There is only one *workgroup barrier* object of ``workgroup`` scope that is implicitly used
by all ``s_barrier`` instructions.

The following code sequences can be used to implement the barrier operations defined by the
:ref:`execution synchronization model<amdgpu-execution-synchronization-barriers-execution-model>` using
``s_barrier`` on GFX6 through GFX11:

.. table:: s_barrier GFX6-11
    :name: amdgpu-execution-synchronization-barriers-sbarrier-gfx6-11
    :widths: 15 15 70

    ===================== ====================== ===========================================================
    Barrier Operation(s)  Barrier *Object*       AMDGPU Machine Code
    ===================== ====================== ===========================================================
    **Init, Join and Drop**
    --------------------------------------------------------------------------------------------------------
    *init*                - *Workgroup barrier*  Automatically initialized by the hardware when a workgroup
                                                 is launched. The *expected count* of this barrier is set
                                                 to the number of waves in the workgroup.

    *join*                - *Workgroup barrier*  Any thread launched within a workgroup automatically *joins*
                                                 this barrier *object*.

    *drop*                - *Workgroup barrier*  When a thread ends, it automatically *drops* this barrier
                                                 *object* if it had previously *joined* it.

    **Arrive and Wait**
    --------------------------------------------------------------------------------------------------------
    *arrive* then *wait*  - *Workgroup barrier*  | **BackOffBarrier**
                                                 | ``s_barrier``
                                                 | **No BackOffBarrier**
                                                 | ``s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)``
                                                 | ``s_waitcnt_vscnt null, 0x0``
                                                 | ``s_barrier``

                                                 - If the target does not have the BackOffBarrier feature,
                                                   then there cannot be any outstanding memory operations
                                                   before issuing the ``s_barrier`` instruction.
                                                 - The waitcnts can independently be moved earlier, or
                                                   removed entirely as long as the associated
                                                   counter remains at zero before issuing the
                                                   ``s_barrier`` instruction.
                                                 - The ``s_barrier`` instruction cannot complete
                                                   before all waves of the workgroup have launched.

    *arrive*              - *Workgroup barrier*  Not available separately, see *arrive* then *wait*

    *wait*                - *Workgroup barrier*  Not available separately, see *arrive* then *wait*
    ===================== ====================== ===========================================================

GFX12
+++++

GFX12 targets have the split-barrier feature, and also allow ``s_barrier`` instructions to use
one of multiple barrier *objects* available per workgroup. ``s_barrier`` instruction use the
barrier ID operand to determine the barrier *object* they operate on.

GFX12.5 additionally introduces new barrier *objects* that offer more flexibility for synchronizing the execution
of a subset of waves of a workgroup, or synchronizing execution across workgroups within a workgroup cluster, via
``s_barrier``.

.. note::

  Check the :ref:`the table below<amdgpu-execution-synchronization-barriers-sbarrier-ids-gfx12>` to determine
  which barrier IDs are available to ``s_barrier`` instructions on a given target.

The following code sequences can be used to implement the barrier operations defined by the
:ref:`execution synchronization model<amdgpu-execution-synchronization-barriers-execution-model>` using
``s_barrier`` on GFX12.0 and up:

.. table:: s_barrier GFX12
    :name: amdgpu-execution-synchronization-barriers-sbarrier-gfx2
    :widths: 15 15 70

    ===================== =========================== ===========================================================
    Barrier Operation(s)  Barrier ID                  AMDGPU Machine Code
    ===================== =========================== ===========================================================
    **Init, Join and Drop**
    -------------------------------------------------------------------------------------------------------------
    *init*                - ``-2``, ``-1``            Automatically initialized by the hardware when a workgroup
                                                      is launched. The *expected count* of this barrier is set
                                                      to the number of waves in the workgroup.

    *init*                - ``-4``, ``-3``            Automatically initialized by the hardware when a workgroup
                                                      is launched as part of a workgroup cluster.
                                                      The *expected count* of this barrier is set to the number
                                                      of workgroups in the workgroup cluster.

    *init*                - ``0``                     Automatically initialized by the hardware and always
                                                      available. This barrier *object* is opaque and immutable
                                                      as all operations other than barrier *join* are no-ops.

    *init*                - ``[1, 16]``               | ``s_barrier_init <N>``

                                                      - ``<N>`` is an immediate constant, or stored in the lower
                                                        half of ``m0``.
                                                      - The value to set as the *expected count* of the barrier
                                                        is stored in the upper half of ``m0``.

    *join*                - ``-2``, ``-1``            Any thread launched within a workgroup automatically *joins*
                                                      this barrier *object*.

    *join*                - ``-4``, ``-3``            Any thread launched within a workgroup cluster
                                                      automatically *joins* this barrier *object*.

    *join*                - ``0``                     | ``s_barrier_join <N>``
                          - ``[1, 16]``
                                                      - ``<N>`` is an immediate constant, or stored in the lower
                                                        half of ``m0``.

    *drop*                - ``0``                     | ``s_barrier_leave``
                          - ``[1, 16]``
                                                      - ``s_barrier_leave`` takes no operand. It can only be used
                                                        to *drop* a barrier *object* ``BO`` if ``BO`` was
                                                        previously *joined* using ``s_barrier_join``.
                                                      - *Drops* the barrier *object* ``BO`` if and only if
                                                        there is a barrier *join* ``J`` such that ``J`` is
                                                        *barrier-joined-before* this barrier
                                                        *drop* operation.

    *drop*                - ``-2``, ``-1``            When a thread ends, it automatically *drops* this barrier
                          - ``-4``, ``-3``            *object* if it had previously *joined* it.

    **Arrive and Wait**
    -------------------------------------------------------------------------------------------------------------

    *arrive*              - ``-4``, ``-3``            | ``s_barrier_signal <N>``
                          - ``-2``, ``-1``            | Or
                          - ``0``                     | ``s_barrier_signal_isfirst <N>``
                          - ``[1, 16]``
                                                      - ``<N>`` is an immediate constant, or stored in bits ``[4:0]`` of ``m0``.
                                                      - The ``_isfirst`` variant sets ``SCC=1`` if this wave is the first
                                                        to signal the barrier, otherwise ``SCC=0``.
                                                      - For barrier *objects* ``[1, 16]``: When using ``m0`` as an operand,
                                                        if there is a non-zero value contained in the bits ``[22:16]`` of ``m0``,
                                                        the *expected count* of the barrier *object* is set to that value before
                                                        the *arrive count* of the barrier *object* is incremented.
                                                        The new *expected count* value must be greater than or equal to the
                                                        *arrive count*, otherwise the behavior is undefined.
                                                      - For barrier *objects* ``-4`` and ``-3``
                                                        (``cluster`` barriers): only one wave
                                                        per workgroup may arrive at the barrier on behalf of
                                                        its entire workgroup. However, any wave within the workgroup
                                                        cluster can then *wait* on this barrier *object*.
                                                      - This is a no-op on the *NULL named barrier object*
                                                        (barrier *object* ``0``).

    *wait*                - ``-4``, ``-3``            ``s_barrier_wait <N>``.
                          - ``-2``, ``-1``
                          - ``0``                     - ``<N>`` is an immediate constant.
                          - ``[1, 16]``               - For barrier *objects* ``-2`` and ``-1``: This instruction
                                                        cannot complete before all waves of the
                                                        workgroup have launched.
                                                      - For barrier *objects* ``-4`` and ``-3`` (``cluster`` barriers):
                                                        This instruction cannot complete before all waves of the
                                                        workgroup cluster have launched.
                                                      - This is a no-op on the *NULL named barrier object*
                                                        (barrier *object* ``0``).
                                                      - For *named barrier objects*, this instruction always waits on the
                                                        last *named barrier object* that the thread has *joined*, even
                                                        if it is different from the *barrier object* passed to the
                                                        instruction.
    ===================== =========================== ===========================================================


The following barrier IDs are available:

.. table:: s_barrier IDs GFX12
    :name: amdgpu-execution-synchronization-barriers-sbarrier-ids-gfx12
    :widths: 15 15 15 55

    =============== ============== ============ ==============================================================
    Barrier ID      Scope          Availability Description
    =============== ============== ============ ==============================================================
    ``-4``          ``cluster``    GFX12.5      *Cluster trap barrier*; *cluster barrier object* for use by
                                                all workgroups of a workgroup cluster. Dedicated for the trap
                                                handler and only available in privileged execution mode
                                                (not accessible by the shader).

    ``-3``          ``cluster``    GFX12.5      *Cluster user barrier*; *cluster barrier object* for use by
                                                all workgroups of a workgroup cluster.

    ``-2``          ``workgroup``  GFX12 (all)  *Workgroup trap barrier*, dedicated for the trap handler and
                                                only available in privileged execution mode
                                                (not accessible by the shader).

    ``-1``          ``workgroup``  GFX12 (all)  *Workgroup barrier*.

    ``0``           ``workgroup``  GFX12.5      *NULL named barrier object*. *Barrier-mutually-exclusive* with
                                                barriers ``[1, 16]``.

    ``[1, 16]``     ``workgroup``  GFX12.5      *Named barrier object*. All barrier *objects* in this range are
                                                *barrier-mutually-exclusive* with other barriers in ``[0, 16]``.
    =============== ============== ============ ==============================================================


Informally, we can note that:

* All operations on the *NULL named barrier object* other than *join* are no-ops.

  * As the *NULL named barrier object* (barrier ID ``0``) is *barrier-mutually-exclusive* with all other
    *named barrier objects* (barrier IDs ``[1, 16]``), a thread can use a *join* on the *NULL*
    barrier as a way to "unjoin" a *named barrier* (break *barrier-joined-before*) without
    having to use a *drop* operation.

* When a thread ends, it does **not** implicitly *drop* any *named barrier objects*
  (barrier IDs ``[0, 16]``) it has *joined*.
