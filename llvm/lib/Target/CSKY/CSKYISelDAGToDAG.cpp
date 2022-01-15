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
  bool selectAddCarry(SDNode *N);
  bool selectSubCarry(SDNode *N);

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
  case ISD::ADDCARRY:
    IsSelected = selectAddCarry(N);
    break;
  case ISD::SUBCARRY:
    IsSelected = selectSubCarry(N);
    break;
  case ISD::GLOBAL_OFFSET_TABLE: {
    Register GP = Subtarget->getInstrInfo()->getGlobalBaseReg(*MF);
    ReplaceNode(N, CurDAG->getRegister(GP, N->getValueType(0)).getNode());

    IsSelected = true;
    break;
  }
  case ISD::FrameIndex: {
    SDValue Imm = CurDAG->getTargetConstant(0, Dl, MVT::i32);
    int FI = cast<FrameIndexSDNode>(N)->getIndex();
    SDValue TFI = CurDAG->getTargetFrameIndex(FI, MVT::i32);
    ReplaceNode(N, CurDAG->getMachineNode(Subtarget->hasE2() ? CSKY::ADDI32
                                                             : CSKY::ADDI16XZ,
                                          Dl, MVT::i32, TFI, Imm));

    IsSelected = true;
    break;
  }
  }

  if (IsSelected)
    return;

  // Select the default instruction.
  SelectCode(N);
}

bool CSKYDAGToDAGISel::selectAddCarry(SDNode *N) {
  MachineSDNode *NewNode = nullptr;
  auto Type0 = N->getValueType(0);
  auto Type1 = N->getValueType(1);
  auto Op0 = N->getOperand(0);
  auto Op1 = N->getOperand(1);
  auto Op2 = N->getOperand(2);

  SDLoc Dl(N);

  if (isNullConstant(Op2)) {
    auto *CA = CurDAG->getMachineNode(
        Subtarget->has2E3() ? CSKY::CLRC32 : CSKY::CLRC16, Dl, Type1);
    NewNode = CurDAG->getMachineNode(
        Subtarget->has2E3() ? CSKY::ADDC32 : CSKY::ADDC16, Dl, {Type0, Type1},
        {Op0, Op1, SDValue(CA, 0)});
  } else if (isOneConstant(Op2)) {
    auto *CA = CurDAG->getMachineNode(
        Subtarget->has2E3() ? CSKY::SETC32 : CSKY::SETC16, Dl, Type1);
    NewNode = CurDAG->getMachineNode(
        Subtarget->has2E3() ? CSKY::ADDC32 : CSKY::ADDC16, Dl, {Type0, Type1},
        {Op0, Op1, SDValue(CA, 0)});
  } else {
    NewNode = CurDAG->getMachineNode(Subtarget->has2E3() ? CSKY::ADDC32
                                                         : CSKY::ADDC16,
                                     Dl, {Type0, Type1}, {Op0, Op1, Op2});
  }
  ReplaceNode(N, NewNode);
  return true;
}

static SDValue InvertCarryFlag(const CSKYSubtarget *Subtarget,
                               SelectionDAG *DAG, SDLoc Dl, SDValue OldCarry) {
  auto NewCarryReg =
      DAG->getMachineNode(Subtarget->has2E3() ? CSKY::MVCV32 : CSKY::MVCV16, Dl,
                          MVT::i32, OldCarry);
  auto NewCarry =
      DAG->getMachineNode(Subtarget->hasE2() ? CSKY::BTSTI32 : CSKY::BTSTI16,
                          Dl, OldCarry.getValueType(), SDValue(NewCarryReg, 0),
                          DAG->getTargetConstant(0, Dl, MVT::i32));
  return SDValue(NewCarry, 0);
}

bool CSKYDAGToDAGISel::selectSubCarry(SDNode *N) {
  MachineSDNode *NewNode = nullptr;
  auto Type0 = N->getValueType(0);
  auto Type1 = N->getValueType(1);
  auto Op0 = N->getOperand(0);
  auto Op1 = N->getOperand(1);
  auto Op2 = N->getOperand(2);

  SDLoc Dl(N);

  if (isNullConstant(Op2)) {
    auto *CA = CurDAG->getMachineNode(
        Subtarget->has2E3() ? CSKY::SETC32 : CSKY::SETC16, Dl, Type1);
    NewNode = CurDAG->getMachineNode(
        Subtarget->has2E3() ? CSKY::SUBC32 : CSKY::SUBC16, Dl, {Type0, Type1},
        {Op0, Op1, SDValue(CA, 0)});
  } else if (isOneConstant(Op2)) {
    auto *CA = CurDAG->getMachineNode(
        Subtarget->has2E3() ? CSKY::CLRC32 : CSKY::CLRC16, Dl, Type1);
    NewNode = CurDAG->getMachineNode(
        Subtarget->has2E3() ? CSKY::SUBC32 : CSKY::SUBC16, Dl, {Type0, Type1},
        {Op0, Op1, SDValue(CA, 0)});
  } else {
    auto CarryIn = InvertCarryFlag(Subtarget, CurDAG, Dl, Op2);
    NewNode = CurDAG->getMachineNode(Subtarget->has2E3() ? CSKY::SUBC32
                                                         : CSKY::SUBC16,
                                     Dl, {Type0, Type1}, {Op0, Op1, CarryIn});
  }
  auto CarryOut = InvertCarryFlag(Subtarget, CurDAG, Dl, SDValue(NewNode, 1));

  ReplaceUses(SDValue(N, 0), SDValue(NewNode, 0));
  ReplaceUses(SDValue(N, 1), CarryOut);
  CurDAG->RemoveDeadNode(N);

  return true;
}

FunctionPass *llvm::createCSKYISelDag(CSKYTargetMachine &TM) {
  return new CSKYDAGToDAGISel(TM);
}
