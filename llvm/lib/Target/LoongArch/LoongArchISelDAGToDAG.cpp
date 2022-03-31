//=- LoongArchISelDAGToDAG.cpp - A dag to dag inst selector for LoongArch -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the LoongArch target.
//
//===----------------------------------------------------------------------===//

#include "LoongArchISelDAGToDAG.h"
#include "MCTargetDesc/LoongArchMCTargetDesc.h"

using namespace llvm;

#define DEBUG_TYPE "loongarch-isel"

void LoongArchDAGToDAGISel::Select(SDNode *Node) {
  // If we have a custom node, we have already selected.
  if (Node->isMachineOpcode()) {
    LLVM_DEBUG(dbgs() << "== "; Node->dump(CurDAG); dbgs() << "\n");
    Node->setNodeId(-1);
    return;
  }

  // Instruction Selection not handled by the auto-generated tablegen selection
  // should be handled here.
  unsigned Opcode = Node->getOpcode();
  SDLoc DL(Node);

  switch (Opcode) {
  default:
    break;
    // TODO: Add selection nodes needed later.
  }

  // Select the default instruction.
  SelectCode(Node);
}
// This pass converts a legalized DAG into a LoongArch-specific DAG, ready
// for instruction scheduling.
FunctionPass *llvm::createLoongArchISelDag(LoongArchTargetMachine &TM) {
  return new LoongArchDAGToDAGISel(TM);
}
