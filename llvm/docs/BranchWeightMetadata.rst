.. _branch_weight:

===========================
LLVM Branch Weight Metadata
===========================

.. contents::
   :local:

Introduction
============

Branch Weight Metadata represents branch weights as its likeliness to be
taken. Metadata is assigned to the ``TerminatorInst`` as a ``MDNode`` of the
``MD_prof`` kind. The first operator is always a ``MDString`` node with the
string "branch_weights". Number of operators depends on the terminator type.

Branch weights might be fetch from the profiling file, or generated based on
`__builtin_expect`_ instruction.

All weights are represented as an unsigned 32-bit values, where higher value
indicates greater chance to be taken.

Supported Instructions
======================

``BranchInst``
^^^^^^^^^^^^^^

Metadata is only assign to the conditional branches. There are two extra
operarands, for the true and the false branch.

.. code-block:: llvm

  !0 = metadata !{
    metadata !"branch_weights",
    i32 <TRUE_BRANCH_WEIGHT>,
    i32 <FALSE_BRANCH_WEIGHT>
  }

``SwitchInst``
^^^^^^^^^^^^^^

Branch weights are assign to every case (including ``default`` case which is
always case #0).

.. code-block:: llvm

  !0 = metadata !{
    metadata !"branch_weights",
    i32 <DEFAULT_BRANCH_WEIGHT>
    [ , i32 <CASE_BRANCH_WEIGHT> ... ]
  }

``IndirectBrInst``
^^^^^^^^^^^^^^^^^^

Branch weights are assign to every destination.

.. code-block:: llvm

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
occur due to incorrent branch prediction information.
