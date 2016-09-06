===========================
LLVM Branch Weight Metadata
===========================

.. contents::
   :local:

Introduction
============

Branch Weight Metadata represents branch weights as its likeliness to be taken
(see :doc:`BlockFrequencyTerminology`). Metadata is assigned to the
``TerminatorInst`` as a ``MDNode`` of the ``MD_prof`` kind. The first operator
is always a ``MDString`` node with the string "branch_weights".  Number of
operators depends on the terminator type.

Branch weights might be fetch from the profiling file, or generated based on
`__builtin_expect`_ instruction.

All weights are represented as an unsigned 32-bit values, where higher value
indicates greater chance to be taken.

Supported Instructions
======================

``BranchInst``
^^^^^^^^^^^^^^

Metadata is only assigned to the conditional branches. There are two extra
operands for the true and the false branch.

.. code-block:: none

  !0 = metadata !{
    metadata !"branch_weights",
    i32 <TRUE_BRANCH_WEIGHT>,
    i32 <FALSE_BRANCH_WEIGHT>
  }

``SwitchInst``
^^^^^^^^^^^^^^

Branch weights are assigned to every case (including the ``default`` case which
is always case #0).

.. code-block:: none

  !0 = metadata !{
    metadata !"branch_weights",
    i32 <DEFAULT_BRANCH_WEIGHT>
    [ , i32 <CASE_BRANCH_WEIGHT> ... ]
  }

``IndirectBrInst``
^^^^^^^^^^^^^^^^^^

Branch weights are assigned to every destination.

.. code-block:: none

  !0 = metadata !{
    metadata !"branch_weights",
    i32 <LABEL_BRANCH_WEIGHT>
    [ , i32 <LABEL_BRANCH_WEIGHT> ... ]
  }

Other
^^^^^

Other terminator instructions are not allowed to contain Branch Weight Metadata.

.. _\__builtin_expect:

Built-in ``expect`` Instructions
================================

``__builtin_expect(long exp, long c)`` instruction provides branch prediction
information. The return value is the value of ``exp``.

It is especially useful in conditional statements. Currently Clang supports two
conditional statements:

``if`` statement
^^^^^^^^^^^^^^^^

The ``exp`` parameter is the condition. The ``c`` parameter is the expected
comparison value. If it is equal to 1 (true), the condition is likely to be
true, in other case condition is likely to be false. For example:

.. code-block:: c++

  if (__builtin_expect(x > 0, 1)) {
    // This block is likely to be taken.
  }

``switch`` statement
^^^^^^^^^^^^^^^^^^^^

The ``exp`` parameter is the value. The ``c`` parameter is the expected
value. If the expected value doesn't show on the cases list, the ``default``
case is assumed to be likely taken.

.. code-block:: c++

  switch (__builtin_expect(x, 5)) {
  default: break;
  case 0:  // ...
  case 3:  // ...
  case 5:  // This case is likely to be taken.
  }

CFG Modifications
=================

Branch Weight Metatada is not proof against CFG changes. If terminator operands'
are changed some action should be taken. In other case some misoptimizations may
occur due to incorrect branch prediction information.

Function Entry Counts
=====================

To allow comparing different functions during inter-procedural analysis and
optimization, ``MD_prof`` nodes can also be assigned to a function definition.
The first operand is a string indicating the name of the associated counter.

Currently, one counter is supported: "function_entry_count". This is a 64-bit
counter that indicates the number of times that this function was invoked (in
the case of instrumentation-based profiles). In the case of sampling-based
profiles, this counter is an approximation of how many times the function was
invoked.

For example, in the code below, the instrumentation for function foo()
indicates that it was called 2,590 times at runtime.

.. code-block:: llvm

  define i32 @foo() !prof !1 {
    ret i32 0
  }
  !1 = !{!"function_entry_count", i64 2590}
