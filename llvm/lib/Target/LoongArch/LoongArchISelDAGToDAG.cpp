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
#include "LoongArchISelLowering.h"
#include "MCTargetDesc/LoongArchMCTargetDesc.h"
#include "MCTargetDesc/LoongArchMatInt.h"
#include "llvm/Support/KnownBits.h"

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
  MVT VT = Node->getSimpleValueType(0);

  switch (Opcode) {
  default:
    break;
  case ISD::Constant: {
    int64_t Imm = cast<ConstantSDNode>(Node)->getSExtValue();
    if (Imm == 0 && VT == GRLenVT) {
      SDValue New = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), DL,
                                           LoongArch::R0, GRLenVT);
      ReplaceNode(Node, New.getNode());
      return;
    }
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
  case ISD::FrameIndex: {
    SDValue Imm = CurDAG->getTargetConstant(0, DL, GRLenVT);
    int FI = cast<FrameIndexSDNode>(Node)->getIndex();
    SDValue TFI = CurDAG->getTargetFrameIndex(FI, VT);
    unsigned ADDIOp =
        Subtarget->is64Bit() ? LoongArch::ADDI_D : LoongArch::ADDI_W;
    ReplaceNode(Node, CurDAG->getMachineNode(ADDIOp, DL, VT, TFI, Imm));
    return;
  }
    // TODO: Add selection nodes needed later.
  }

  // Select the default instruction.
  SelectCode(Node);
}

bool LoongArchDAGToDAGISel::SelectBaseAddr(SDValue Addr, SDValue &Base) {
  // If this is FrameIndex, select it directly. Otherwise just let it get
  // selected to a register independently.
  if (auto *FIN = dyn_cast<FrameIndexSDNode>(Addr))
    Base =
        CurDAG->getTargetFrameIndex(FIN->getIndex(), Subtarget->getGRLenVT());
  else
    Base = Addr;
  return true;
}

bool LoongArchDAGToDAGISel::selectShiftMask(SDValue N, unsigned ShiftWidth,
                                            SDValue &ShAmt) {
  // Shift instructions on LoongArch only read the lower 5 or 6 bits of the
  // shift amount. If there is an AND on the shift amount, we can bypass it if
  // it doesn't affect any of those bits.
  if (N.getOpcode() == ISD::AND && isa<ConstantSDNode>(N.getOperand(1))) {
    const APInt &AndMask = N->getConstantOperandAPInt(1);

    // Since the max shift amount is a power of 2 we can subtract 1 to make a
    // mask that covers the bits needed to represent all shift amounts.
    assert(isPowerOf2_32(ShiftWidth) && "Unexpected max shift amount!");
    APInt ShMask(AndMask.getBitWidth(), ShiftWidth - 1);

    if (ShMask.isSubsetOf(AndMask)) {
      ShAmt = N.getOperand(0);
      return true;
    }

    // SimplifyDemandedBits may have optimized the mask so try restoring any
    // bits that are known zero.
    KnownBits Known = CurDAG->computeKnownBits(N->getOperand(0));
    if (ShMask.isSubsetOf(AndMask | Known.Zero)) {
      ShAmt = N.getOperand(0);
      return true;
    }
  } else if (N.getOpcode() == LoongArchISD::BSTRPICK) {
    // Similar to the above AND, if there is a BSTRPICK on the shift amount, we
    // can bypass it.
    assert(isPowerOf2_32(ShiftWidth) && "Unexpected max shift amount!");
    assert(isa<ConstantSDNode>(N.getOperand(1)) && "Illegal msb operand!");
    assert(isa<ConstantSDNode>(N.getOperand(2)) && "Illegal lsb operand!");
    uint64_t msb = N.getConstantOperandVal(1), lsb = N.getConstantOperandVal(2);
    if (lsb == 0 && Log2_32(ShiftWidth) <= msb + 1) {
      ShAmt = N.getOperand(0);
      return true;
    }
  } else if (N.getOpcode() == ISD::SUB &&
             isa<ConstantSDNode>(N.getOperand(0))) {
    uint64_t Imm = N.getConstantOperandVal(0);
    // If we are shifting by N-X where N == 0 mod Size, then just shift by -X to
    // generate a NEG instead of a SUB of a constant.
    if (Imm != 0 && Imm % ShiftWidth == 0) {
      SDLoc DL(N);
      EVT VT = N.getValueType();
      SDValue Zero =
          CurDAG->getCopyFromReg(CurDAG->getEntryNode(), DL, LoongArch::R0, VT);
      unsigned NegOpc = VT == MVT::i64 ? LoongArch::SUB_D : LoongArch::SUB_W;
      MachineSDNode *Neg =
          CurDAG->getMachineNode(NegOpc, DL, VT, Zero, N.getOperand(1));
      ShAmt = SDValue(Neg, 0);
      return true;
    }
  }

  ShAmt = N;
  return true;
}

// This pass converts a legalized DAG into a LoongArch-specific DAG, ready
// for instruction scheduling.
FunctionPass *llvm::createLoongArchISelDag(LoongArchTargetMachine &TM) {
  return new LoongArchDAGToDAGISel(TM);
}
