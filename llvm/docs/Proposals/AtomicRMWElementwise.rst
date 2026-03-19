AtomicRMW Elementwise
=====================

Overview
--------

This proposal adds an ``elementwise`` modifier to the LLVM IR
``atomicrmw`` instruction:

.. code-block:: llvm

   %old = atomicrmw elementwise fadd ptr %ptr, <4 x float> %val seq_cst

The modifier changes vector ``atomicrmw`` from whole-value atomic semantics to
per-lane atomic semantics.

Motivation
----------

LLVM IR currently gives vector ``atomicrmw`` whole-value atomic semantics.
That is stronger than the semantics provided by most hardware vector atomic
instructions, which are atomic per lane and do not guarantee atomicity across
the entire vector value.

This mismatch creates two distinct use cases:

- A language or frontend may require whole-vector atomicity. That use case
  should keep using plain ``atomicrmw`` and lower to a whole-value
  implementation such as a compare-exchange loop when legal.
- A language or frontend may want hardware-style vector atomics. That use case
  needs an IR representation whose semantics match per-lane atomics.

The new ``elementwise`` modifier makes that distinction explicit in the IR.

Proposed Syntax
---------------

The textual syntax becomes:

.. code-block:: text

   atomicrmw [volatile] [elementwise] <operation> ptr <pointer>, <ty> <value>
             [syncscope("<target-scope>")] <ordering>[, align <alignment>]

Examples:

.. code-block:: llvm

   %old0 = atomicrmw elementwise add ptr %p, <4 x i32> %v monotonic
   %old1 = atomicrmw volatile elementwise fadd ptr %p, <4 x float> %v seq_cst

Semantics
---------

Without ``elementwise``, ``atomicrmw`` keeps its current whole-value atomic
semantics.

With ``elementwise``, the instruction is only valid for fixed vectors, and it
behaves as if it were expanded into one scalar ``atomicrmw`` per vector lane,
executed in lane order by the issuing thread. The result is the vector formed
from the old values returned by those scalar operations.

For example:

.. code-block:: llvm

   %old = atomicrmw elementwise fadd ptr %p, <4 x float> %v seq_cst

has the same semantics as:

.. code-block:: llvm

   %p0 = getelementptr <4 x float>, ptr %p, i64 0, i64 0
   %p1 = getelementptr <4 x float>, ptr %p, i64 0, i64 1
   %p2 = getelementptr <4 x float>, ptr %p, i64 0, i64 2
   %p3 = getelementptr <4 x float>, ptr %p, i64 0, i64 3
   %v0 = extractelement <4 x float> %v, i64 0
   %v1 = extractelement <4 x float> %v, i64 1
   %v2 = extractelement <4 x float> %v, i64 2
   %v3 = extractelement <4 x float> %v, i64 3
   %o0 = atomicrmw fadd ptr %p0, float %v0 seq_cst
   %o1 = atomicrmw fadd ptr %p1, float %v1 seq_cst
   %o2 = atomicrmw fadd ptr %p2, float %v2 seq_cst
   %o3 = atomicrmw fadd ptr %p3, float %v3 seq_cst
   %t0 = insertelement <4 x float> poison, float %o0, i64 0
   %t1 = insertelement <4 x float> %t0, float %o1, i64 1
   %t2 = insertelement <4 x float> %t1, float %o2, i64 2
   %old = insertelement <4 x float> %t2, float %o3, i64 3

No atomicity is guaranteed across lanes.

Type Rules
----------

The element type of an ``elementwise`` vector must be valid for the
corresponding scalar ``atomicrmw`` operation:

- ``xchg``: integer, floating-point, or pointer element type
- ``add/sub/and/nand/or/xor/max/min/umax/umin/uinc_wrap/udec_wrap/usub_cond/usub_sat``:
  integer element type
- ``fadd/fsub/fmax/fmin/fmaximum/fminimum``: floating-point element type

Scalable vectors are not supported.

Memory Model
------------

The ordering, synchronization scope, volatility, and alignment apply to each
scalar lane operation.

``seq_cst`` remains per-lane ``seq_cst``. The modifier does not create a new
cross-lane atomicity guarantee or a compound transaction.

Lowering Strategy
-----------------

The proposed lowering split is:

- Plain vector ``atomicrmw`` keeps whole-value semantics and may lower to a
  whole-value compare-exchange loop.
- ``atomicrmw elementwise`` is preserved into codegen for targets that opt in
  to late native lowering, and otherwise scalarizes into one scalar
  ``atomicrmw`` per lane in IR.

This keeps the semantic split explicit and target-independent:

- targets that have native vector atomics can lower the preserved instruction
  directly
- targets that have native scalar atomics get native code after scalarization
- targets that need scalar compare-exchange loops reuse existing scalar atomic
  expansion
- whole-value vector atomics remain available for targets that can implement
  them

Prototype Scope
---------------

The prototype implementation in this tree:

- adds the ``elementwise`` modifier to ``atomicrmw``
- preserves existing plain ``atomicrmw`` semantics
- preserves ``atomicrmw elementwise`` into codegen for NVPTX vector ``fadd``
  cases and scalarizes the rest in ``AtomicExpand``
- keeps ``AtomicExpand`` responsible for the scalar fallback and for splitting
  preserved ``seq_cst`` elementwise RMWs into fences plus a weaker atomic
- keeps NVPTX whole-vector lowering on the plain instruction path

Open Questions
--------------

- Whether the final syntax should keep ``elementwise`` as a modifier on
  ``atomicrmw`` or use a separate instruction spelling.
- Whether future work should add elementwise support to ``cmpxchg``.
- Whether the LangRef should define lane order explicitly or describe the
  modifier purely in terms of observational equivalence to scalar expansion.
