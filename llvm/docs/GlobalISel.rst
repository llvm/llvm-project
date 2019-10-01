============================
Global Instruction Selection
============================

.. contents::
   :local:
   :depth: 1

.. warning::
   This document is a work in progress.  It reflects the current state of the
   implementation, as well as open design and implementation issues.

Introduction
============

GlobalISel is a framework that provides a set of reusable passes and utilities
for instruction selection --- translation from LLVM IR to target-specific
Machine IR (MIR).

GlobalISel is intended to be a replacement for SelectionDAG and FastISel, to
solve three major problems:

* **Performance** --- SelectionDAG introduces a dedicated intermediate
  representation, which has a compile-time cost.

  GlobalISel directly operates on the post-isel representation used by the
  rest of the code generator, MIR.
  It does require extensions to that representation to support arbitrary
  incoming IR: :ref:`gmir`.

* **Granularity** --- SelectionDAG and FastISel operate on individual basic
  blocks, losing some global optimization opportunities.

  GlobalISel operates on the whole function.

* **Modularity** --- SelectionDAG and FastISel are radically different and share
  very little code.

  GlobalISel is built in a way that enables code reuse. For instance, both the
  optimized and fast selectors share the :ref:`pipeline`, and targets can
  configure that pipeline to better suit their needs.


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

.. _pipeline:

Core Pipeline
=============

There are four required passes, regardless of the optimization mode:

.. contents::
   :local:

Additional passes can then be inserted at higher optimization levels or for
specific targets. For example, to match the current SelectionDAG set of
transformations: MachineCSE and a better MachineCombiner between every pass.

``NOTE``:
In theory, not all passes are always necessary.
As an additional compile-time optimization, we could skip some of the passes by
setting the relevant MachineFunction properties.  For instance, if the
IRTranslator did not encounter any illegal instruction, it would set the
``legalized`` property to avoid running the :ref:`milegalizer`.
Similarly, we considered specializing the IRTranslator per-target to directly
emit target-specific MI.
However, we instead decided to keep the core pipeline simple, and focus on
minimizing the overhead of the passes in the no-op cases.


.. _irtranslator:

IRTranslator
------------

This pass translates the input LLVM IR ``Function`` to a GMIR
``MachineFunction``.

``TODO``:
This currently doesn't support the more complex instructions, in particular
those involving control flow (``switch``, ``invoke``, ...).
For ``switch`` in particular, we can initially use the ``LowerSwitch`` pass.

.. _api-calllowering:

API: CallLowering
^^^^^^^^^^^^^^^^^

The ``IRTranslator`` (using the ``CallLowering`` target-provided utility) also
implements the ABI's calling convention by lowering calls, returns, and
arguments to the appropriate physical register usage and instruction sequences.

.. _irtranslator-aggregates:

Aggregates
^^^^^^^^^^

Aggregates are lowered to a single scalar vreg.
This differs from SelectionDAG's multiple vregs via ``GetValueVTs``.

``TODO``:
As some of the bits are undef (padding), we should consider augmenting the
representation with additional metadata (in effect, caching computeKnownBits
information on vregs).
See `PR26161 <http://llvm.org/PR26161>`_: [GlobalISel] Value to vreg during
IR to MachineInstr translation for aggregate type

.. _irtranslator-constants:

Constant Lowering
^^^^^^^^^^^^^^^^^

The ``IRTranslator`` lowers ``Constant`` operands into uses of gvregs defined
by ``G_CONSTANT`` or ``G_FCONSTANT`` instructions.
Currently, these instructions are always emitted in the entry basic block.
In a ``MachineFunction``, each ``Constant`` is materialized by a single gvreg.

This is beneficial as it allows us to fold constants into immediate operands
during :ref:`instructionselect`, while still avoiding redundant materializations
for expensive non-foldable constants.
However, this can lead to unnecessary spills and reloads in an -O0 pipeline, as
these vregs can have long live ranges.

``TODO``:
We're investigating better placement of these instructions, in fast and
optimized modes.


.. _milegalizer:

Legalizer
---------

This pass transforms the generic machine instructions such that they are legal.

A legal instruction is defined as:

* **selectable** --- the target will later be able to select it to a
  target-specific (non-generic) instruction.

* operating on **vregs that can be loaded and stored** -- if necessary, the
  target can select a ``G_LOAD``/``G_STORE`` of each gvreg operand.

As opposed to SelectionDAG, there are no legalization phases.  In particular,
'type' and 'operation' legalization are not separate.

Legalization is iterative, and all state is contained in GMIR.  To maintain the
validity of the intermediate code, instructions are introduced:

* ``G_MERGE_VALUES`` --- concatenate multiple registers of the same
  size into a single wider register.

* ``G_UNMERGE_VALUES`` --- extract multiple registers of the same size
  from a single wider register.

* ``G_EXTRACT`` --- extract a simple register (as contiguous sequences of bits)
  from a single wider register.

As they are expected to be temporary byproducts of the legalization process,
they are combined at the end of the :ref:`milegalizer` pass.
If any remain, they are expected to always be selectable, using loads and stores
if necessary.

The legality of an instruction may only depend on the instruction itself and
must not depend on any context in which the instruction is used. However, after
deciding that an instruction is not legal, using the context of the instruction
to decide how to legalize the instruction is permitted. As an example, if we
have a ``G_FOO`` instruction of the form::

  %1:_(s32) = G_CONSTANT i32 1
  %2:_(s32) = G_FOO %0:_(s32), %1:_(s32)

it's impossible to say that G_FOO is legal iff %1 is a ``G_CONSTANT`` with
value ``1``. However, the following::

  %2:_(s32) = G_FOO %0:_(s32), i32 1

can say that it's legal iff operand 2 is an immediate with value ``1`` because
that information is entirely contained within the single instruction.

.. _api-legalizerinfo:

API: LegalizerInfo
^^^^^^^^^^^^^^^^^^

The recommended [#legalizer-legacy-footnote]_ API looks like this::

  getActionDefinitionsBuilder({G_ADD, G_SUB, G_MUL, G_AND, G_OR, G_XOR, G_SHL})
      .legalFor({s32, s64, v2s32, v4s32, v2s64})
      .clampScalar(0, s32, s64)
      .widenScalarToNextPow2(0)
      .clampNumElements(0, v2s32, v4s32)
      .clampNumElements(0, v2s64, v2s64)
      .moreElementsToNextPow2(0);

and describes a set of rules by which we can either declare an instruction legal
or decide which action to take to make it more legal.

At the core of this ruleset is the ``LegalityQuery`` which describes the
instruction. We use a description rather than the instruction to both allow other
passes to determine legality without having to create an instruction and also to
limit the information available to the predicates to that which is safe to rely
on. Currently, the information available to the predicates that determine
legality contains:

* The opcode for the instruction

* The type of each type index (see ``type0``, ``type1``, etc.)

* The size in bytes and atomic ordering for each MachineMemOperand

Rule Processing and Declaring Rules
"""""""""""""""""""""""""""""""""""

The ``getActionDefinitionsBuilder`` function generates a ruleset for the given
opcode(s) that rules can be added to. If multiple opcodes are given, they are
all permanently bound to the same ruleset. The rules in a ruleset are executed
from top to bottom and will start again from the top if an instruction is
legalized as a result of the rules. If the ruleset is exhausted without
satisfying any rule, then it is considered unsupported.

When it doesn't declare the instruction legal, each pass over the rules may
request that one type changes to another type. Sometimes this can cause multiple
types to change but we avoid this as much as possible as making multiple changes
can make it difficult to avoid infinite loops where, for example, narrowing one
type causes another to be too small and widening that type causes the first one
to be too big.

In general, it's advisable to declare instructions legal as close to the top of
the rule as possible and to place any expensive rules as low as possible. This
helps with performance as testing for legality happens more often than
legalization and legalization can require multiple passes over the rules.

As a concrete example, consider the rule::

  getActionDefinitionsBuilder({G_ADD, G_SUB, G_MUL, G_AND, G_OR, G_XOR, G_SHL})
      .legalFor({s32, s64, v2s32, v4s32, v2s64})
      .clampScalar(0, s32, s64)
      .widenScalarToNextPow2(0);

and the instruction::

  %2:_(s7) = G_ADD %0:_(s7), %1:_(s7)

this doesn't meet the predicate for the :ref:`.legalFor() <legalfor>` as ``s7``
is not one of the listed types so it falls through to the
:ref:`.clampScalar() <clampscalar>`. It does meet the predicate for this rule
as the type is smaller than the ``s32`` and this rule instructs the legalizer
to change type 0 to ``s32``. It then restarts from the top. This time it does
satisfy ``.legalFor()`` and the resulting output is::

  %3:_(s32) = G_ANYEXT %0:_(s7)
  %4:_(s32) = G_ANYEXT %1:_(s7)
  %5:_(s32) = G_ADD %3:_(s32), %4:_(s32)
  %2:_(s7) = G_TRUNC %5:_(s32)

where the ``G_ADD`` is legal and the other instructions are scheduled for
processing by the legalizer.

Rule Actions
""""""""""""

There are various rule factories that append rules to a ruleset but they have a
few actions in common:

.. _legalfor:

* ``legalIf()``, ``legalFor()``, etc. declare an instruction to be legal if the
  predicate is satisfied.

* ``narrowScalarIf()``, ``narrowScalarFor()``, etc. declare an instruction to be illegal
  if the predicate is satisfied and indicates that narrowing the scalars in one
  of the types to a specific type would make it more legal. This action supports
  both scalars and vectors.

* ``widenScalarIf()``, ``widenScalarFor()``, etc. declare an instruction to be illegal
  if the predicate is satisfied and indicates that widening the scalars in one
  of the types to a specific type would make it more legal. This action supports
  both scalars and vectors.

* ``fewerElementsIf()``, ``fewerElementsFor()``, etc. declare an instruction to be
  illegal if the predicate is satisfied and indicates reducing the number of
  vector elements in one of the types to a specific type would make it more
  legal. This action supports vectors.

* ``moreElementsIf()``, ``moreElementsFor()``, etc. declare an instruction to be illegal
  if the predicate is satisfied and indicates increasing the number of vector
  elements in one of the types to a specific type would make it more legal.
  This action supports vectors.

* ``lowerIf()``, ``lowerFor()``, etc. declare an instruction to be illegal if the
  predicate is satisfied and indicates that replacing it with equivalent
  instruction(s) would make it more legal. Support for this action differs for
  each opcode.

* ``libcallIf()``, ``libcallFor()``, etc. declare an instruction to be illegal if the
  predicate is satisfied and indicates that replacing it with a libcall would
  make it more legal. Support for this action differs for
  each opcode.

* ``customIf()``, ``customFor()``, etc. declare an instruction to be illegal if the
  predicate is satisfied and indicates that the backend developer will supply
  a means of making it more legal.

* ``unsupportedIf()``, ``unsupportedFor()``, etc. declare an instruction to be illegal
  if the predicate is satisfied and indicates that there is no way to make it
  legal and the compiler should fail.

* ``fallback()`` falls back on an older API and should only be used while porting
  existing code from that API.

Rule Predicates
"""""""""""""""

The rule factories also have predicates in common:

* ``legal()``, ``lower()``, etc. are always satisfied.

* ``legalIf()``, ``narrowScalarIf()``, etc. are satisfied if the user-supplied
  ``LegalityPredicate`` function returns true. This predicate has access to the
  information in the ``LegalityQuery`` to make its decision.
  User-supplied predicates can also be combined using ``all(P0, P1, ...)``.

* ``legalFor()``, ``narrowScalarFor()``, etc. are satisfied if the type matches one in
  a given set of types. For example ``.legalFor({s16, s32})`` declares the
  instruction legal if type 0 is either s16 or s32. Additional versions for two
  and three type indices are generally available. For these, all the type
  indices considered together must match all the types in one of the tuples. So
  ``.legalFor({{s16, s32}, {s32, s64}})`` will only accept ``{s16, s32}``, or
  ``{s32, s64}`` but will not accept ``{s16, s64}``.

* ``legalForTypesWithMemSize()``, ``narrowScalarForTypesWithMemSize()``, etc. are
  similar to ``legalFor()``, ``narrowScalarFor()``, etc. but additionally require a
  MachineMemOperand to have a given size in each tuple.

* ``legalForCartesianProduct()``, ``narrowScalarForCartesianProduct()``, etc. are
  satisfied if each type index matches one element in each of the independent
  sets. So ``.legalForCartesianProduct({s16, s32}, {s32, s64})`` will accept
  ``{s16, s32}``, ``{s16, s64}``, ``{s32, s32}``, and ``{s32, s64}``.

Composite Rules
"""""""""""""""

There are some composite rules for common situations built out of the above facilities:

* ``widenScalarToNextPow2()`` is like ``widenScalarIf()`` but is satisfied iff the type
  size in bits is not a power of 2 and selects a target type that is the next
  largest power of 2. 

.. _clampscalar:

* ``minScalar()`` is like ``widenScalarIf()`` but is satisfied iff the type
  size in bits is smaller than the given minimum and selects the minimum as the
  target type. Similarly, there is also a ``maxScalar()`` for the maximum and a
  ``clampScalar()`` to do both at once. 

* ``minScalarSameAs()`` is like ``minScalar()`` but the minimum is taken from another
  type index.

* ``moreElementsToNextMultiple()`` is like ``moreElementsToNextPow2()`` but is based on
  multiples of X rather than powers of 2.

Other Information
"""""""""""""""""

``TODO``:
An alternative worth investigating is to generalize the API to represent
actions using ``std::function`` that implements the action, instead of explicit
enum tokens (``Legal``, ``WidenScalar``, ...).

``TODO``:
Moreover, we could use TableGen to initially infer legality of operation from
existing patterns (as any pattern we can select is by definition legal).
Expanding that to describe legalization actions is a much larger but
potentially useful project.

.. rubric:: Footnotes

.. [#legalizer-legacy-footnote] An API is broadly similar to
   SelectionDAG/TargetLowering is available but is not recommended as a more
   powerful API is available.

.. _regbankselect:

RegBankSelect
-------------

This pass constrains the :ref:`gmir-gvregs` operands of generic
instructions to some :ref:`gmir-regbank`.

It iteratively maps instructions to a set of per-operand bank assignment.
The possible mappings are determined by the target-provided
:ref:`RegisterBankInfo <api-registerbankinfo>`.
The mapping is then applied, possibly introducing ``COPY`` instructions if
necessary.

It traverses the ``MachineFunction`` top down so that all operands are already
mapped when analyzing an instruction.

This pass could also remap target-specific instructions when beneficial.
In the future, this could replace the ExeDepsFix pass, as we can directly
select the best variant for an instruction that's available on multiple banks.

.. _api-registerbankinfo:

API: RegisterBankInfo
^^^^^^^^^^^^^^^^^^^^^

The ``RegisterBankInfo`` class describes multiple aspects of register banks.

* **Banks**: ``addRegBankCoverage`` --- which register bank covers each
  register class.

* **Cross-Bank Copies**: ``copyCost`` --- the cost of a ``COPY`` from one bank
  to another.

* **Default Mapping**: ``getInstrMapping`` --- the default bank assignments for
  a given instruction.

* **Alternative Mapping**: ``getInstrAlternativeMapping`` --- the other
  possible bank assignments for a given instruction.

``TODO``:
All this information should eventually be static and generated by TableGen,
mostly using existing information augmented by bank descriptions.

``TODO``:
``getInstrMapping`` is currently separate from ``getInstrAlternativeMapping``
because the latter is more expensive: as we move to static mapping info,
both methods should be free, and we should merge them.

.. _regbankselect-modes:

RegBankSelect Modes
^^^^^^^^^^^^^^^^^^^

``RegBankSelect`` currently has two modes:

* **Fast** --- For each instruction, pick a target-provided "default" bank
  assignment.  This is the default at -O0.

* **Greedy** --- For each instruction, pick the cheapest of several
  target-provided bank assignment alternatives.

We intend to eventually introduce an additional optimizing mode:

* **Global** --- Across multiple instructions, pick the cheapest combination of
  bank assignments.

``NOTE``:
On AArch64, we are considering using the Greedy mode even at -O0 (or perhaps at
backend -O1):  because :ref:`gmir-llt` doesn't distinguish floating point from
integer scalars, the default assignment for loads and stores is the integer
bank, introducing cross-bank copies on most floating point operations.


.. _instructionselect:

InstructionSelect
-----------------

This pass transforms generic machine instructions into equivalent
target-specific instructions.  It traverses the ``MachineFunction`` bottom-up,
selecting uses before definitions, enabling trivial dead code elimination.

.. _api-instructionselector:

API: InstructionSelector
^^^^^^^^^^^^^^^^^^^^^^^^

The target implements the ``InstructionSelector`` class, containing the
target-specific selection logic proper.

The instance is provided by the subtarget, so that it can specialize the
selector by subtarget feature (with, e.g., a vector selector overriding parts
of a general-purpose common selector).
We might also want to parameterize it by MachineFunction, to enable selector
variants based on function attributes like optsize.

The simple API consists of:

  .. code-block:: c++

    virtual bool select(MachineInstr &MI)

This target-provided method is responsible for mutating (or replacing) a
possibly-generic MI into a fully target-specific equivalent.
It is also responsible for doing the necessary constraining of gvregs into the
appropriate register classes as well as passing through COPY instructions to
the register allocator.

The ``InstructionSelector`` can fold other instructions into the selected MI,
by walking the use-def chain of the vreg operands.
As GlobalISel is Global, this folding can occur across basic blocks.

SelectionDAG Rule Imports
^^^^^^^^^^^^^^^^^^^^^^^^^

TableGen will import SelectionDAG rules and provide the following function to
execute them:

  .. code-block:: c++

    bool selectImpl(MachineInstr &MI)

The ``--stats`` option can be used to determine what proportion of rules were
successfully imported. The easiest way to use this is to copy the
``-gen-globalisel`` tablegen command from ``ninja -v`` and modify it.

Similarly, the ``--warn-on-skipped-patterns`` option can be used to obtain the
reasons that rules weren't imported. This can be used to focus on the most
important rejection reasons.

PatLeaf Predicates
^^^^^^^^^^^^^^^^^^

PatLeafs cannot be imported because their C++ is implemented in terms of
``SDNode`` objects. PatLeafs that handle immediate predicates should be
replaced by ``ImmLeaf``, ``IntImmLeaf``, or ``FPImmLeaf`` as appropriate.

There's no standard answer for other PatLeafs. Some standard predicates have
been baked into TableGen but this should not generally be done.

Custom SDNodes
^^^^^^^^^^^^^^

Custom SDNodes should be mapped to Target Pseudos using ``GINodeEquiv``. This
will cause the instruction selector to import them but you will also need to
ensure the target pseudo is introduced to the MIR before the instruction
selector. Any preceding pass is suitable but the legalizer will be a
particularly common choice.

ComplexPatterns
^^^^^^^^^^^^^^^

ComplexPatterns cannot be imported because their C++ is implemented in terms of
``SDNode`` objects. GlobalISel versions should be defined with
``GIComplexOperandMatcher`` and mapped to ComplexPattern with
``GIComplexPatternEquiv``.

The following predicates are useful for porting ComplexPattern:

* isBaseWithConstantOffset() - Check for base+offset structures
* isOperandImmEqual() - Check for a particular constant
* isObviouslySafeToFold() - Check for reasons an instruction can't be sunk and folded into another.

There are some important points for the C++ implementation:

* Don't modify MIR in the predicate
* Renderer lambdas should capture by value to avoid use-after-free. They will be used after the predicate returns.
* Only create instructions in a renderer lambda. GlobalISel won't clean up things you create but don't use.


.. _maintainability:

Maintainability
===============

.. _maintainability-iterative:

Iterative Transformations
-------------------------

Passes are split into small, iterative transformations, with all state
represented in the MIR.

This differs from SelectionDAG (in particular, the legalizer) using various
in-memory side-tables.


.. _maintainability-mir:

MIR Serialization
-----------------

.. FIXME: Update the MIRLangRef to include GMI additions.

:ref:`gmir` is serializable (see :doc:`MIRLangRef`).
Combined with :ref:`maintainability-iterative`, this enables much finer-grained
testing, rather than requiring large and fragile IR-to-assembly tests.

The current "stage" in the :ref:`pipeline` is represented by a set of
``MachineFunctionProperties``:

* ``legalized``
* ``regBankSelected``
* ``selected``


.. _maintainability-verifier:

MachineVerifier
---------------

The pass approach lets us use the ``MachineVerifier`` to enforce invariants.
For instance, a ``regBankSelected`` function may not have gvregs without
a bank.

``TODO``:
The ``MachineVerifier`` being monolithic, some of the checks we want to do
can't be integrated to it:  GlobalISel is a separate library, so we can't
directly reference it from CodeGen.  For instance, legality checks are
currently done in RegBankSelect/InstructionSelect proper.  We could #ifdef out
the checks, or we could add some sort of verifier API.


.. _progress:

Progress and Future Work
========================

The initial goal is to replace FastISel on AArch64.  The next step will be to
replace SelectionDAG as the optimized ISel.

``NOTE``:
While we iterate on GlobalISel, we strive to avoid affecting the performance of
SelectionDAG, FastISel, or the other MIR passes.  For instance, the types of
:ref:`gmir-gvregs` are stored in a separate table in ``MachineRegisterInfo``,
that is destroyed after :ref:`instructionselect`.

.. _progress-fastisel:

FastISel Replacement
--------------------

For the initial FastISel replacement, we intend to fallback to SelectionDAG on
selection failures.

Currently, compile-time of the fast pipeline is within 1.5x of FastISel.
We're optimistic we can get to within 1.1/1.2x, but beating FastISel will be
challenging given the multi-pass approach.
Still, supporting all IR (via a complete legalizer) and avoiding the fallback
to SelectionDAG in the worst case should enable better amortized performance
than SelectionDAG+FastISel.

``NOTE``:
We considered never having a fallback to SelectionDAG, instead deciding early
whether a given function is supported by GlobalISel or not.  The decision would
be based on :ref:`milegalizer` queries.
We abandoned that for two reasons:
a) on IR inputs, we'd need to basically simulate the :ref:`irtranslator`;
b) to be robust against unforeseen failures and to enable iterative
improvements.

.. _progress-targets:

Support For Other Targets
-------------------------

In parallel, we're investigating adding support for other - ideally quite
different - targets.  For instance, there is some initial AMDGPU support.


.. _porting:

Porting GlobalISel to A New Target
==================================

There are four major classes to implement by the target:

* :ref:`CallLowering <api-calllowering>` --- lower calls, returns, and arguments
  according to the ABI.
* :ref:`RegisterBankInfo <api-registerbankinfo>` --- describe
  :ref:`gmir-regbank` coverage, cross-bank copy cost, and the mapping of
  operands onto banks for each instruction.
* :ref:`LegalizerInfo <api-legalizerinfo>` --- describe what is legal, and how
  to legalize what isn't.
* :ref:`InstructionSelector <api-instructionselector>` --- select generic MIR
  to target-specific MIR.

Additionally:

* ``TargetPassConfig`` --- create the passes constituting the pipeline,
  including additional passes not included in the :ref:`pipeline`.

.. _other_resources:

Resources
=========

* `Global Instruction Selection - A Proposal by Quentin Colombet @LLVMDevMeeting 2015 <https://www.youtube.com/watch?v=F6GGbYtae3g>`_
* `Global Instruction Selection - Status by Quentin Colombet, Ahmed Bougacha, and Tim Northover @LLVMDevMeeting 2016 <https://www.youtube.com/watch?v=6tfb344A7w8>`_
* `GlobalISel - LLVM's Latest Instruction Selection Framework by Diana Picus @FOSDEM17 <https://www.youtube.com/watch?v=d6dF6E4BPeU>`_
* GlobalISel: Past, Present, and Future by Quentin Colombet and Ahmed Bougacha @LLVMDevMeeting 2017
* Head First into GlobalISel by Daniel Sanders, Aditya Nandakumar, and Justin Bogner @LLVMDevMeeting 2017
