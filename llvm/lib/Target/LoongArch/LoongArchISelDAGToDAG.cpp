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
#include "MCTargetDesc/LoongArchMatInt.h"

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
  MVT GRLenVT = Subtarget->getGRLenVT();
  SDLoc DL(Node);

  switch (Opcode) {
  default:
    break;
  case ISD::Constant: {
    int64_t Imm = cast<ConstantSDNode>(Node)->getSExtValue();
    SDNode *Result = nullptr;
    SDValue SrcReg = CurDAG->getRegister(LoongArch::R0, GRLenVT);

    // The instructions in the sequence are handled here.
    for (LoongArchMatInt::Inst &Inst : LoongArchMatInt::generateInstSeq(Imm)) {
      SDValue SDImm = CurDAG->getTargetConstant(Inst.Imm, DL, GRLenVT);
      if (Inst.Opc == LoongArch::LU12I_W)
        Result = CurDAG->getMachineNode(LoongArch::LU12I_W, DL, GRLenVT, SDImm);
      else
        Result = CurDAG->getMachineNode(Inst.Opc, DL, GRLenVT, SrcReg, SDImm);
      SrcReg = SDValue(Result, 0);
    }

    ReplaceNode(Node, Result);
    return;
  }
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
