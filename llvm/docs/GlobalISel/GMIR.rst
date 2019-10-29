.. _gmir:

Generic Machine IR
==================

Machine IR operates on physical registers, register classes, and (mostly)
target-specific instructions.

To bridge the gap with LLVM IR, GlobalISel introduces "generic" extensions to
Machine IR:

.. contents::
   :local:

``NOTE``:
The generic MIR (GMIR) representation still contains references to IR
constructs (such as ``GlobalValue``).  Removing those should let us write more
accurate tests, or delete IR after building the initial MIR.  However, it is
not part of the GlobalISel effort.

.. _gmir-instructions:

Generic Instructions
--------------------

The main addition is support for pre-isel generic machine instructions (e.g.,
``G_ADD``).  Like other target-independent instructions (e.g., ``COPY`` or
``PHI``), these are available on all targets.

``TODO``:
While we're progressively adding instructions, one kind in particular exposes
interesting problems: compares and how to represent condition codes.
Some targets (x86, ARM) have generic comparisons setting multiple flags,
which are then used by predicated variants.
Others (IR) specify the predicate in the comparison and users just get a single
bit.  SelectionDAG uses SETCC/CONDBR vs BR_CC (and similar for select) to
represent this.

The ``MachineIRBuilder`` class wraps the ``MachineInstrBuilder`` and provides
a convenient way to create these generic instructions.

.. _gmir-gvregs:

Generic Virtual Registers
-------------------------

Generic instructions operate on a new kind of register: "generic" virtual
registers.  As opposed to non-generic vregs, they are not assigned a Register
Class.  Instead, generic vregs have a :ref:`gmir-llt`, and can be assigned
a :ref:`gmir-regbank`.

``MachineRegisterInfo`` tracks the same information that it does for
non-generic vregs (e.g., use-def chains).  Additionally, it also tracks the
:ref:`gmir-llt` of the register, and, instead of the ``TargetRegisterClass``,
its :ref:`gmir-regbank`, if any.

For simplicity, most generic instructions only accept generic vregs:

* instead of immediates, they use a gvreg defined by an instruction
  materializing the immediate value (see :ref:`irtranslator-constants`).
* instead of physical register, they use a gvreg defined by a ``COPY``.

``NOTE``:
We started with an alternative representation, where MRI tracks a size for
each gvreg, and instructions have lists of types.
That had two flaws: the type and size are redundant, and there was no generic
way of getting a given operand's type (as there was no 1:1 mapping between
instruction types and operands).
We considered putting the type in some variant of MCInstrDesc instead:
See `PR26576 <http://llvm.org/PR26576>`_: [GlobalISel] Generic MachineInstrs
need a type but this increases the memory footprint of the related objects

.. _gmir-regbank:

Register Bank
-------------

A Register Bank is a set of register classes defined by the target.
A bank has a size, which is the maximum store size of all covered classes.

In general, cross-class copies inside a bank are expected to be cheaper than
copies across banks.  They are also coalesceable by the register coalescer,
whereas cross-bank copies are not.

Also, equivalent operations can be performed on different banks using different
instructions.

For example, X86 can be seen as having 3 main banks: general-purpose, x87, and
vector (which could be further split into a bank per domain for single vs
double precision instructions).

Register banks are described by a target-provided API,
:ref:`RegisterBankInfo <api-registerbankinfo>`.

.. _gmir-llt:

Low Level Type
--------------

Additionally, every generic virtual register has a type, represented by an
instance of the ``LLT`` class.

Like ``EVT``/``MVT``/``Type``, it has no distinction between unsigned and signed
integer types.  Furthermore, it also has no distinction between integer and
floating-point types: it mainly conveys absolutely necessary information, such
as size and number of vector lanes:

* ``sN`` for scalars
* ``pN`` for pointers
* ``<N x sM>`` for vectors
* ``unsized`` for labels, etc..

``LLT`` is intended to replace the usage of ``EVT`` in SelectionDAG.

Here are some LLT examples and their ``EVT`` and ``Type`` equivalents:

   =============  =========  ======================================
   LLT            EVT        IR Type
   =============  =========  ======================================
   ``s1``         ``i1``     ``i1``
   ``s8``         ``i8``     ``i8``
   ``s32``        ``i32``    ``i32``
   ``s32``        ``f32``    ``float``
   ``s17``        ``i17``    ``i17``
   ``s16``        N/A        ``{i8, i8}``
   ``s32``        N/A        ``[4 x i8]``
   ``p0``         ``iPTR``   ``i8*``, ``i32*``, ``%opaque*``
   ``p2``         ``iPTR``   ``i8 addrspace(2)*``
   ``<4 x s32>``  ``v4f32``  ``<4 x float>``
   ``s64``        ``v1f64``  ``<1 x double>``
   ``<3 x s32>``  ``v3i32``  ``<3 x i32>``
   ``unsized``    ``Other``  ``label``
   =============  =========  ======================================


Rationale: instructions already encode a specific interpretation of types
(e.g., ``add`` vs. ``fadd``, or ``sdiv`` vs. ``udiv``).  Also encoding that
information in the type system requires introducing bitcast with no real
advantage for the selector.

Pointer types are distinguished by address space.  This matches IR, as opposed
to SelectionDAG where address space is an attribute on operations.
This representation better supports pointers having different sizes depending
on their addressspace.

``NOTE``:
Currently, LLT requires at least 2 elements in vectors, but some targets have
the concept of a '1-element vector'.  Representing them as their underlying
scalar type is a nice simplification.

``TODO``:
Currently, non-generic virtual registers, defined by non-pre-isel-generic
instructions, cannot have a type, and thus cannot be used by a pre-isel generic
instruction.  Instead, they are given a type using a COPY.  We could relax that
and allow types on all vregs: this would reduce the number of MI required when
emitting target-specific MIR early in the pipeline.  This should purely be
a compile-time optimization.

