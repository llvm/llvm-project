//===-- CSKYISelDAGToDAG.cpp - A dag to dag inst selector for CSKY---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the CSKY target.
//
//===----------------------------------------------------------------------===//

#include "CSKY.h"
#include "CSKYSubtarget.h"
#include "CSKYTargetMachine.h"
#include "MCTargetDesc/CSKYMCTargetDesc.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"

using namespace llvm;

#define DEBUG_TYPE "csky-isel"

namespace {
class CSKYDAGToDAGISel : public SelectionDAGISel {
  const CSKYSubtarget *Subtarget;

public:
  explicit CSKYDAGToDAGISel(CSKYTargetMachine &TM) : SelectionDAGISel(TM) {}

  StringRef getPassName() const override {
    return "CSKY DAG->DAG Pattern Instruction Selection";
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    // Reset the subtarget each time through.
    Subtarget = &MF.getSubtarget<CSKYSubtarget>();
    SelectionDAGISel::runOnMachineFunction(MF);
    return true;
  }

  void Select(SDNode *N) override;

#include "CSKYGenDAGISel.inc"
};
} // namespace

void CSKYDAGToDAGISel::Select(SDNode *N) {
  // If we have a custom node, we have already selected
  if (N->isMachineOpcode()) {
    LLVM_DEBUG(dbgs() << "== "; N->dump(CurDAG); dbgs() << "\n");
    N->setNodeId(-1);
    return;
  }

  SDLoc Dl(N);
  unsigned Opcode = N->getOpcode();
  bool IsSelected = false;

  switch (Opcode) {
  default:
    break;
    // FIXME: Add selection nodes needed later.
  }

  if (IsSelected)
    return;

  // Select the default instruction.
  SelectCode(N);
}

FunctionPass *llvm::createCSKYISelDag(CSKYTargetMachine &TM) {
  return new CSKYDAGToDAGISel(TM);
}
