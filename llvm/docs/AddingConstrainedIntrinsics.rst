==================================================
How To Add A Constrained Floating-Point Intrinsic
==================================================

.. contents::
   :local:

.. warning::
  This is a work in progress.

Add the intrinsic
=================

Multiple files need to be updated when adding a new constrained intrinsic.

Add the new intrinsic to the table of intrinsics.::

  include/llvm/IR/Intrinsics.td

Update class ConstrainedFPIntrinsic to know about the intrinsics.::

  include/llvm/IR/IntrinsicInst.h

Functions like ConstrainedFPIntrinsic::isUnaryOp() or
ConstrainedFPIntrinsic::isTernaryOp() may need to know about the new
intrinsic.::

  lib/IR/IntrinsicInst.cpp

Update the IR verifier::

  lib/IR/Verifier.cpp

Add SelectionDAG node types
===========================

Add the new STRICT version of the node type to the ISD::NodeType enum.::

  include/llvm/CodeGen/ISDOpcodes.h

In class SDNode update isStrictFPOpcode()::

  include/llvm/CodeGen/SelectionDAGNodes.h

A mapping from the STRICT SDnode type to the non-STRICT is done in
TargetLoweringBase::getStrictFPOperationAction(). This allows STRICT
nodes to be legalized similarly to the non-STRICT node type.::

  include/llvm/CodeGen/TargetLowering.h

Building the SelectionDAG
-------------------------

The switch statement in SelectionDAGBuilder::visitIntrinsicCall() needs
to be updated to call SelectionDAGBuilder::visitConstrainedFPIntrinsic().
That function, in turn, needs to be updated to know how to create the
SDNode for the intrinsic. The new STRICT node will eventually be converted
to the matching non-STRICT node. For this reason it should have the same
operands and values as the non-STRICT version but should also use the chain.
This makes subsequent sharing of code for STRICT and non-STRICT code paths
easier.::

  lib/CodeGen/SelectionDAG/SelectionDAGBuilder.cpp

Most of the STRICT nodes get legalized the same as their matching non-STRICT
counterparts. A new STRICT node with this property must get added to the
switch in SelectionDAGLegalize::LegalizeOp().::

  lib/CodeGen/SelectionDAG/LegalizeDAG.cpp

Other parts of the legalizer may need to be updated as well. Look for
places where the non-STRICT counterpart is legalized and update as needed.
Be careful of the chain since STRICT nodes use it but their counterparts
often don't.

The code to do the conversion or mutation of the STRICT node to a non-STRICT
version of the node happens in SelectionDAG::mutateStrictFPToFP(). Be
careful updating this function since some nodes have the same return type
as their input operand, but some are different. Both of these cases must
be properly handled.::

  lib/CodeGen/SelectionDAG/SelectionDAG.cpp

However, the mutation may not happen if the new node has not been registered
in TargetLoweringBase::initActions(). If the corresponding non-STRICT node
is Legal but a target does not know about STRICT nodes then the STRICT
node will default to Legal and mutation will be bypassed with a "Cannot
select" error. Register the new STRICT node as Expand to avoid this bug.::

  lib/CodeGen/TargetLoweringBase.cpp

To make debug logs readable it is helpful to update the SelectionDAG's
debug logger:::

  lib/CodeGen/SelectionDAG/SelectionDAGDumper.cpp

Add documentation and tests
===========================

::

  docs/LangRef.rst
