//===-- ParasolISelDAGToDAG.cpp - A Dag to Dag Inst Selector for Parasol --===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the Parasol target.
//
//===----------------------------------------------------------------------===//

#include "ParasolISelDAGToDAG.h"
#include "ParasolSubtarget.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SelectionDAGISel.h"

using namespace llvm;

#define DEBUG_TYPE "parasol-isel"

char ParasolDAGToDAGISel::ID = 0;

bool ParasolDAGToDAGISel::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &static_cast<const ParasolSubtarget &>(MF.getSubtarget());
  return SelectionDAGISel::runOnMachineFunction(MF);
}

void ParasolDAGToDAGISel::Select(SDNode *Node) {
  unsigned Opcode = Node->getOpcode();

  // If we have a custom node, we already have selected!
  if (Node->isMachineOpcode()) {
    LLVM_DEBUG(errs() << "== "; Node->dump(CurDAG); errs() << "\n");
    Node->setNodeId(-1);
    return;
  }

  // Instruction Selection not handled by the auto-generated tablegen selection
  // should be handled here.
  switch (Opcode) {
  default:
    break;
  }

  // Select the default instruction
  SelectCode(Node);
}
