.. _irtranslator:

IRTranslator
============

.. contents::
  :local:

This pass translates the input LLVM-IR ``Function`` to a GMIR
``MachineFunction``. This is typically a direct translation but does
occasionally get a bit more involved. For example:

.. code-block:: llvm

  %2 = add i32 %0, %1

becomes:

.. code-block:: none

  %2:_(s32) = G_ADD %0:_(s32), %1:_(s32)

whereas

.. code-block:: llvm

  call i32 @puts(i8* %cast210)

is translated according to the ABI rules of the target.

.. note::

  The currently implemented portion of the :doc:`../LangRef` is sufficient for
  many compilations but it is not 100% complete. Users seeking to compile
  LLVM-IR containing some of the rarer features may need to implement the
  translation.

Target Intrinsics
-----------------

There has been some (off-list) debate about whether to add target hooks for
translating target intrinsics. Among those who discussed it, it was generally
agreed that the IRTranslator should be able to lower target intrinsics in a
customizable way but no work has happened to implement this at the time of
writing.

.. _translator-call-lower:

Translating Function Calls
--------------------------

The ``IRTranslator`` also implements the ABI's calling convention by lowering
calls, returns, and arguments to the appropriate physical register usage and
instruction sequences. This is achieved using the ``CallLowering``
implementation,

.. _irtranslator-aggregates:

Aggregates
^^^^^^^^^^

.. caution::

  This has changed since it was written and is no longer accurate. It has not
  been refreshed in this pass of improving the documentation as I haven't
  worked much in this part of the codebase and it should have attention from
  someone more knowledgeable about it.

Aggregates are lowered to a single scalar vreg.
This differs from SelectionDAG's multiple vregs via ``GetValueVTs``.

``TODO``:
As some of the bits are undef (padding), we should consider augmenting the
representation with additional metadata (in effect, caching computeKnownBits
information on vregs).
See `PR26161 <http://llvm.org/PR26161>`_: [GlobalISel] Value to vreg during
IR to MachineInstr translation for aggregate type

.. _irtranslator-constants:

Translation of Constants
------------------------

Constant operands are translated as a use of a virtual register that is defined
by a ``G_CONSTANT`` or ``G_FCONSTANT`` instruction. These instructions are
placed in the entry block to allow them to be subject to the continuous CSE
implementation (``CSEMIRBuilder``). Their debug location information is removed
to prevent this from confusing debuggers.

This is beneficial as it allows us to fold constants into immediate operands
during :ref:`instructionselect`, while still avoiding redundant materializations
for expensive non-foldable constants. However, this can lead to unnecessary
spills and reloads in an -O0 pipeline, as these virtual registers can have long
live ranges. This can be mitigated by running a `localizer <https://github.com/llvm/llvm-project/blob/master/llvm/lib/CodeGen/GlobalISel/Localizer.cpp>`_
after the translator.
