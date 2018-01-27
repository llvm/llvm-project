//===-- Nios2ISelDAGToDAG.cpp - A Dag to Dag Inst Selector for Nios2 ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the NIOS2 target.
//
//===----------------------------------------------------------------------===//

#include "Nios2.h"
#include "Nios2TargetMachine.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "nios2-isel"

//===----------------------------------------------------------------------===//
// Instruction Selector Implementation
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Nios2DAGToDAGISel - NIOS2 specific code to select NIOS2 machine
// instructions for SelectionDAG operations.
//===----------------------------------------------------------------------===//

namespace {

class Nios2DAGToDAGISel : public SelectionDAGISel {
  /// Subtarget - Keep a pointer to the Nios2 Subtarget around so that we can
  /// make the right decision when generating code for different targets.
  const Nios2Subtarget *Subtarget;

public:
  explicit Nios2DAGToDAGISel(Nios2TargetMachine &TM, CodeGenOpt::Level OL)
      : SelectionDAGISel(TM, OL) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    Subtarget = &MF.getSubtarget<Nios2Subtarget>();
    return SelectionDAGISel::runOnMachineFunction(MF);
  }

  void Select(SDNode *N) override;

  // Pass Name
  StringRef getPassName() const override {
    return "NIOS2 DAG->DAG Pattern Instruction Selection";
  }

#include "Nios2GenDAGISel.inc"
};
} // namespace

// Select instructions not customized! Used for
// expanded, promoted and normal instructions
void Nios2DAGToDAGISel::Select(SDNode *Node) {

  // If we have a custom node, we already have selected!
  if (Node->isMachineOpcode()) {
    DEBUG(errs() << "== "; Node->dump(CurDAG); errs() << "\n");
    Node->setNodeId(-1);
    return;
  }

  // Select the default instruction
  SelectCode(Node);
}

FunctionPass *llvm::createNios2ISelDag(Nios2TargetMachine &TM,
                                       CodeGenOpt::Level OptLevel) {
  return new Nios2DAGToDAGISel(TM, OptLevel);
}
