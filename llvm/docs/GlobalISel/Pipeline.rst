.. _pipeline:

Core Pipeline
=============

There are four required passes, regardless of the optimization mode:

.. toctree::
  :maxdepth: 1

  IRTranslator
  Legalizer
  RegBankSelect
  InstructionSelect

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

:ref:`gmir` is serializable (see :doc:`../MIRLangRef`).
Combined with :ref:`maintainability-iterative`, this enables much finer-grained
testing, rather than requiring large and fragile IR-to-assembly tests.

The current "stage" in the :ref:`pipeline` is represented by a set of
``MachineFunctionProperties``:

* ``legalized``
* ``regBankSelected``
* ``selected``

