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

