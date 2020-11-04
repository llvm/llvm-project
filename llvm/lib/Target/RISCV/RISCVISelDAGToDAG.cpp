//===-- RISCVISelDAGToDAG.cpp - A dag to dag inst selector for RISCV ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the RISCV target.
//
//===----------------------------------------------------------------------===//

#include "RISCVISelDAGToDAG.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "Utils/RISCVMatInt.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-isel"

void RISCVDAGToDAGISel::PostprocessISelDAG() {
  doPeepholeLoadStoreADDI();
}

static SDNode *selectImm(SelectionDAG *CurDAG, const SDLoc &DL, int64_t Imm,
                         MVT XLenVT) {
  RISCVMatInt::InstSeq Seq;
  RISCVMatInt::generateInstSeq(Imm, XLenVT == MVT::i64, Seq);

  SDNode *Result = nullptr;
  SDValue SrcReg = CurDAG->getRegister(RISCV::X0, XLenVT);
  for (RISCVMatInt::Inst &Inst : Seq) {
    SDValue SDImm = CurDAG->getTargetConstant(Inst.Imm, DL, XLenVT);
    if (Inst.Opc == RISCV::LUI)
      Result = CurDAG->getMachineNode(RISCV::LUI, DL, XLenVT, SDImm);
    else
      Result = CurDAG->getMachineNode(Inst.Opc, DL, XLenVT, SrcReg, SDImm);

    // Only the first instruction has X0 as its source.
    SrcReg = SDValue(Result, 0);
  }

  return Result;
}

// Returns true if the Node is an ISD::AND with a constant argument. If so,
// set Mask to that constant value.
static bool isConstantMask(SDNode *Node, uint64_t &Mask) {
  if (Node->getOpcode() == ISD::AND &&
      Node->getOperand(1).getOpcode() == ISD::Constant) {
    Mask = cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue();
    return true;
  }
  return false;
}

void RISCVDAGToDAGISel::Select(SDNode *Node) {
  // If we have a custom node, we have already selected.
  if (Node->isMachineOpcode()) {
    LLVM_DEBUG(dbgs() << "== "; Node->dump(CurDAG); dbgs() << "\n");
    Node->setNodeId(-1);
    return;
  }

  // Instruction Selection not handled by the auto-generated tablegen selection
  // should be handled here.
  unsigned Opcode = Node->getOpcode();
  MVT XLenVT = Subtarget->getXLenVT();
  SDLoc DL(Node);
  EVT VT = Node->getValueType(0);

  switch (Opcode) {
  case ISD::ADD: {
    // Optimize (add r, imm) to (addi (addi r, imm0) imm1) if applicable. The
    // immediate must be in specific ranges and have a single use.
    if (auto *ConstOp = dyn_cast<ConstantSDNode>(Node->getOperand(1))) {
      if (!(ConstOp->hasOneUse()))
        break;
      // The imm must be in range [-4096,-2049] or [2048,4094].
      int64_t Imm = ConstOp->getSExtValue();
      if (!(-4096 <= Imm && Imm <= -2049) && !(2048 <= Imm && Imm <= 4094))
        break;
      // Break the imm to imm0+imm1.
      SDLoc DL(Node);
      EVT VT = Node->getValueType(0);
      const SDValue ImmOp0 = CurDAG->getTargetConstant(Imm - Imm / 2, DL, VT);
      const SDValue ImmOp1 = CurDAG->getTargetConstant(Imm / 2, DL, VT);
      auto *NodeAddi0 = CurDAG->getMachineNode(RISCV::ADDI, DL, VT,
                                               Node->getOperand(0), ImmOp0);
      auto *NodeAddi1 = CurDAG->getMachineNode(RISCV::ADDI, DL, VT,
                                               SDValue(NodeAddi0, 0), ImmOp1);
      ReplaceNode(Node, NodeAddi1);
      return;
    }
    break;
  }
  case ISD::Constant: {
    auto ConstNode = cast<ConstantSDNode>(Node);
    if (VT == XLenVT && ConstNode->isNullValue()) {
      SDValue New = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), SDLoc(Node),
                                           RISCV::X0, XLenVT);
      ReplaceNode(Node, New.getNode());
      return;
    }
    int64_t Imm = ConstNode->getSExtValue();
    if (XLenVT == MVT::i64) {
      ReplaceNode(Node, selectImm(CurDAG, SDLoc(Node), Imm, XLenVT));
      return;
    }
    break;
  }
  case ISD::FrameIndex: {
    SDValue Imm = CurDAG->getTargetConstant(0, DL, XLenVT);
    int FI = cast<FrameIndexSDNode>(Node)->getIndex();
    SDValue TFI = CurDAG->getTargetFrameIndex(FI, VT);
    ReplaceNode(Node, CurDAG->getMachineNode(RISCV::ADDI, DL, VT, TFI, Imm));
    return;
  }
  case ISD::SRL: {
    if (!Subtarget->is64Bit())
      break;
    SDNode *Op0 = Node->getOperand(0).getNode();
    uint64_t Mask;
    // Match (srl (and val, mask), imm) where the result would be a
    // zero-extended 32-bit integer. i.e. the mask is 0xffffffff or the result
    // is equivalent to this (SimplifyDemandedBits may have removed lower bits
    // from the mask that aren't necessary due to the right-shifting).
    if (isa<ConstantSDNode>(Node->getOperand(1)) && isConstantMask(Op0, Mask)) {
      uint64_t ShAmt = Node->getConstantOperandVal(1);

      if ((Mask | maskTrailingOnes<uint64_t>(ShAmt)) == 0xffffffff) {
        SDValue ShAmtVal =
            CurDAG->getTargetConstant(ShAmt, SDLoc(Node), XLenVT);
        CurDAG->SelectNodeTo(Node, RISCV::SRLIW, XLenVT, Op0->getOperand(0),
                             ShAmtVal);
        return;
      }
    }
    break;
  }
  case RISCVISD::READ_CYCLE_WIDE:
    assert(!Subtarget->is64Bit() && "READ_CYCLE_WIDE is only used on riscv32");

    ReplaceNode(Node, CurDAG->getMachineNode(RISCV::ReadCycleWide, DL, MVT::i32,
                                             MVT::i32, MVT::Other,
                                             Node->getOperand(0)));
    return;
  }

  // Select the default instruction.
  SelectCode(Node);
}

bool RISCVDAGToDAGISel::SelectInlineAsmMemoryOperand(
    const SDValue &Op, unsigned ConstraintID, std::vector<SDValue> &OutOps) {
  switch (ConstraintID) {
  case InlineAsm::Constraint_m:
    // We just support simple memory operands that have a single address
    // operand and need no special handling.
    OutOps.push_back(Op);
    return false;
  case InlineAsm::Constraint_A:
    OutOps.push_back(Op);
    return false;
  default:
    break;
  }

  return true;
}

bool RISCVDAGToDAGISel::SelectAddrFI(SDValue Addr, SDValue &Base) {
  if (auto FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), Subtarget->getXLenVT());
    return true;
  }
  return false;
}

// Check that it is a SLOI (Shift Left Ones Immediate). We first check that
// it is the right node tree:
//
//  (OR (SHL RS1, VC2), VC1)
//
// and then we check that VC1, the mask used to fill with ones, is compatible
// with VC2, the shamt:
//
//  VC1 == maskTrailingOnes<uint64_t>(VC2)

bool RISCVDAGToDAGISel::SelectSLOI(SDValue N, SDValue &RS1, SDValue &Shamt) {
  MVT XLenVT = Subtarget->getXLenVT();
  if (N.getOpcode() == ISD::OR) {
    SDValue Or = N;
    if (Or.getOperand(0).getOpcode() == ISD::SHL) {
      SDValue Shl = Or.getOperand(0);
      if (isa<ConstantSDNode>(Shl.getOperand(1)) &&
          isa<ConstantSDNode>(Or.getOperand(1))) {
        if (XLenVT == MVT::i64) {
          uint64_t VC1 = Or.getConstantOperandVal(1);
          uint64_t VC2 = Shl.getConstantOperandVal(1);
          if (VC1 == maskTrailingOnes<uint64_t>(VC2)) {
            RS1 = Shl.getOperand(0);
            Shamt = CurDAG->getTargetConstant(VC2, SDLoc(N),
                           Shl.getOperand(1).getValueType());
            return true;
          }
        }
        if (XLenVT == MVT::i32) {
          uint32_t VC1 = Or.getConstantOperandVal(1);
          uint32_t VC2 = Shl.getConstantOperandVal(1);
          if (VC1 == maskTrailingOnes<uint32_t>(VC2)) {
            RS1 = Shl.getOperand(0);
            Shamt = CurDAG->getTargetConstant(VC2, SDLoc(N),
                           Shl.getOperand(1).getValueType());
            return true;
          }
        }
      }
    }
  }
  return false;
}

// Check that it is a SROI (Shift Right Ones Immediate). We first check that
// it is the right node tree:
//
//  (OR (SRL RS1, VC2), VC1)
//
// and then we check that VC1, the mask used to fill with ones, is compatible
// with VC2, the shamt:
//
//  VC1 == maskLeadingOnes<uint64_t>(VC2)

bool RISCVDAGToDAGISel::SelectSROI(SDValue N, SDValue &RS1, SDValue &Shamt) {
  MVT XLenVT = Subtarget->getXLenVT();
  if (N.getOpcode() == ISD::OR) {
    SDValue Or = N;
    if (Or.getOperand(0).getOpcode() == ISD::SRL) {
      SDValue Srl = Or.getOperand(0);
      if (isa<ConstantSDNode>(Srl.getOperand(1)) &&
          isa<ConstantSDNode>(Or.getOperand(1))) {
        if (XLenVT == MVT::i64) {
          uint64_t VC1 = Or.getConstantOperandVal(1);
          uint64_t VC2 = Srl.getConstantOperandVal(1);
          if (VC1 == maskLeadingOnes<uint64_t>(VC2)) {
            RS1 = Srl.getOperand(0);
            Shamt = CurDAG->getTargetConstant(VC2, SDLoc(N),
                           Srl.getOperand(1).getValueType());
            return true;
          }
        }
        if (XLenVT == MVT::i32) {
          uint32_t VC1 = Or.getConstantOperandVal(1);
          uint32_t VC2 = Srl.getConstantOperandVal(1);
          if (VC1 == maskLeadingOnes<uint32_t>(VC2)) {
            RS1 = Srl.getOperand(0);
            Shamt = CurDAG->getTargetConstant(VC2, SDLoc(N),
                           Srl.getOperand(1).getValueType());
            return true;
          }
        }
      }
    }
  }
  return false;
}

// Check that it is a SLLIUW (Shift Logical Left Immediate Unsigned i32
// on RV64).
// SLLIUW is the same as SLLI except for the fact that it clears the bits
// XLEN-1:32 of the input RS1 before shifting.
// We first check that it is the right node tree:
//
//  (AND (SHL RS1, VC2), VC1)
//
// We check that VC2, the shamt is less than 32, otherwise the pattern is
// exactly the same as SLLI and we give priority to that.
// Eventually we check that that VC1, the mask used to clear the upper 32 bits
// of RS1, is correct:
//
//  VC1 == (0xFFFFFFFF << VC2)

bool RISCVDAGToDAGISel::SelectSLLIUW(SDValue N, SDValue &RS1, SDValue &Shamt) {
  if (N.getOpcode() == ISD::AND && Subtarget->getXLenVT() == MVT::i64) {
    SDValue And = N;
    if (And.getOperand(0).getOpcode() == ISD::SHL) {
      SDValue Shl = And.getOperand(0);
      if (isa<ConstantSDNode>(Shl.getOperand(1)) &&
          isa<ConstantSDNode>(And.getOperand(1))) {
        uint64_t VC1 = And.getConstantOperandVal(1);
        uint64_t VC2 = Shl.getConstantOperandVal(1);
        if (VC2 < 32 && VC1 == ((uint64_t)0xFFFFFFFF << VC2)) {
          RS1 = Shl.getOperand(0);
          Shamt = CurDAG->getTargetConstant(VC2, SDLoc(N),
                                            Shl.getOperand(1).getValueType());
          return true;
        }
      }
    }
  }
  return false;
}

// Check that it is a SLOIW (Shift Left Ones Immediate i32 on RV64).
// We first check that it is the right node tree:
//
//  (SIGN_EXTEND_INREG (OR (SHL RS1, VC2), VC1))
//
// and then we check that VC1, the mask used to fill with ones, is compatible
// with VC2, the shamt:
//
//  VC1 == maskTrailingOnes<uint32_t>(VC2)

bool RISCVDAGToDAGISel::SelectSLOIW(SDValue N, SDValue &RS1, SDValue &Shamt) {
  if (Subtarget->getXLenVT() == MVT::i64 &&
      N.getOpcode() == ISD::SIGN_EXTEND_INREG &&
      cast<VTSDNode>(N.getOperand(1))->getVT() == MVT::i32) {
    if (N.getOperand(0).getOpcode() == ISD::OR) {
      SDValue Or = N.getOperand(0);
      if (Or.getOperand(0).getOpcode() == ISD::SHL) {
        SDValue Shl = Or.getOperand(0);
        if (isa<ConstantSDNode>(Shl.getOperand(1)) &&
            isa<ConstantSDNode>(Or.getOperand(1))) {
          uint32_t VC1 = Or.getConstantOperandVal(1);
          uint32_t VC2 = Shl.getConstantOperandVal(1);
          if (VC1 == maskTrailingOnes<uint32_t>(VC2)) {
            RS1 = Shl.getOperand(0);
            Shamt = CurDAG->getTargetConstant(VC2, SDLoc(N),
                                              Shl.getOperand(1).getValueType());
            return true;
          }
        }
      }
    }
  }
  return false;
}

// Check that it is a SROIW (Shift Right Ones Immediate i32 on RV64).
// We first check that it is the right node tree:
//
//  (OR (SHL RS1, VC2), VC1)
//
// and then we check that VC1, the mask used to fill with ones, is compatible
// with VC2, the shamt:
//
//  VC1 == maskLeadingOnes<uint32_t>(VC2)

bool RISCVDAGToDAGISel::SelectSROIW(SDValue N, SDValue &RS1, SDValue &Shamt) {
  if (N.getOpcode() == ISD::OR && Subtarget->getXLenVT() == MVT::i64) {
    SDValue Or = N;
    if (Or.getOperand(0).getOpcode() == ISD::SRL) {
      SDValue Srl = Or.getOperand(0);
      if (isa<ConstantSDNode>(Srl.getOperand(1)) &&
          isa<ConstantSDNode>(Or.getOperand(1))) {
        uint32_t VC1 = Or.getConstantOperandVal(1);
        uint32_t VC2 = Srl.getConstantOperandVal(1);
        if (VC1 == maskLeadingOnes<uint32_t>(VC2)) {
          RS1 = Srl.getOperand(0);
          Shamt = CurDAG->getTargetConstant(VC2, SDLoc(N),
                                            Srl.getOperand(1).getValueType());
          return true;
        }
      }
    }
  }
  return false;
}

// Check that it is a RORIW (i32 Right Rotate Immediate on RV64).
// We first check that it is the right node tree:
//
//  (SIGN_EXTEND_INREG (OR (SHL RS1, VC2),
//                         (SRL (AND RS1, VC3), VC1)))
//
// Then we check that the constant operands respect these constraints:
//
// VC2 == 32 - VC1
// VC3 | maskTrailingOnes<uint64_t>(VC1) == 0xffffffff
//
// being VC1 the Shamt we need, VC2 the complementary of Shamt over 32
// and VC3 being 0xffffffff after accounting for SimplifyDemandedBits removing
// some bits due to the right shift.

bool RISCVDAGToDAGISel::SelectRORIW(SDValue N, SDValue &RS1, SDValue &Shamt) {
  if (N.getOpcode() == ISD::SIGN_EXTEND_INREG &&
      Subtarget->getXLenVT() == MVT::i64 &&
      cast<VTSDNode>(N.getOperand(1))->getVT() == MVT::i32) {
    if (N.getOperand(0).getOpcode() == ISD::OR) {
      SDValue Or = N.getOperand(0);
      SDValue Shl = Or.getOperand(0);
      SDValue Srl = Or.getOperand(1);

      // OR is commutable so canonicalize SHL to LHS.
      if (Srl.getOpcode() == ISD::SHL)
        std::swap(Shl, Srl);

      if (Shl.getOpcode() == ISD::SHL && Srl.getOpcode() == ISD::SRL) {
        if (Srl.getOperand(0).getOpcode() == ISD::AND) {
          SDValue And = Srl.getOperand(0);
          if (And.getOperand(0) == Shl.getOperand(0) &&
              isa<ConstantSDNode>(Srl.getOperand(1)) &&
              isa<ConstantSDNode>(Shl.getOperand(1)) &&
              isa<ConstantSDNode>(And.getOperand(1))) {
            uint64_t VC1 = Srl.getConstantOperandVal(1);
            uint64_t VC2 = Shl.getConstantOperandVal(1);
            uint64_t VC3 = And.getConstantOperandVal(1);
            // The mask needs to be 0xffffffff, but SimplifyDemandedBits may
            // have removed lower bits that aren't necessary due to the right
            // shift.
            if (VC2 == (32 - VC1) &&
                (VC3 | maskTrailingOnes<uint64_t>(VC1)) == 0xffffffff) {
              RS1 = Shl.getOperand(0);
              Shamt = CurDAG->getTargetConstant(VC1, SDLoc(N),
                                              Srl.getOperand(1).getValueType());
              return true;
            }
          }
        }
      }
    }
  }
  return false;
}

// Merge an ADDI into the offset of a load/store instruction where possible.
// (load (addi base, off1), off2) -> (load base, off1+off2)
// (store val, (addi base, off1), off2) -> (store val, base, off1+off2)
// This is possible when off1+off2 fits a 12-bit immediate.
void RISCVDAGToDAGISel::doPeepholeLoadStoreADDI() {
  SelectionDAG::allnodes_iterator Position(CurDAG->getRoot().getNode());
  ++Position;

  while (Position != CurDAG->allnodes_begin()) {
    SDNode *N = &*--Position;
    // Skip dead nodes and any non-machine opcodes.
    if (N->use_empty() || !N->isMachineOpcode())
      continue;

    int OffsetOpIdx;
    int BaseOpIdx;

    // Only attempt this optimisation for I-type loads and S-type stores.
    switch (N->getMachineOpcode()) {
    default:
      continue;
    case RISCV::LB:
    case RISCV::LH:
    case RISCV::LW:
    case RISCV::LBU:
    case RISCV::LHU:
    case RISCV::LWU:
    case RISCV::LD:
    case RISCV::FLW:
    case RISCV::FLD:
      BaseOpIdx = 0;
      OffsetOpIdx = 1;
      break;
    case RISCV::SB:
    case RISCV::SH:
    case RISCV::SW:
    case RISCV::SD:
    case RISCV::FSW:
    case RISCV::FSD:
      BaseOpIdx = 1;
      OffsetOpIdx = 2;
      break;
    }

    if (!isa<ConstantSDNode>(N->getOperand(OffsetOpIdx)))
      continue;

    SDValue Base = N->getOperand(BaseOpIdx);

    // If the base is an ADDI, we can merge it in to the load/store.
    if (!Base.isMachineOpcode() || Base.getMachineOpcode() != RISCV::ADDI)
      continue;

    SDValue ImmOperand = Base.getOperand(1);
    uint64_t Offset2 = N->getConstantOperandVal(OffsetOpIdx);

    if (auto Const = dyn_cast<ConstantSDNode>(ImmOperand)) {
      int64_t Offset1 = Const->getSExtValue();
      int64_t CombinedOffset = Offset1 + Offset2;
      if (!isInt<12>(CombinedOffset))
        continue;
      ImmOperand = CurDAG->getTargetConstant(CombinedOffset, SDLoc(ImmOperand),
                                             ImmOperand.getValueType());
    } else if (auto GA = dyn_cast<GlobalAddressSDNode>(ImmOperand)) {
      // If the off1 in (addi base, off1) is a global variable's address (its
      // low part, really), then we can rely on the alignment of that variable
      // to provide a margin of safety before off1 can overflow the 12 bits.
      // Check if off2 falls within that margin; if so off1+off2 can't overflow.
      const DataLayout &DL = CurDAG->getDataLayout();
      Align Alignment = GA->getGlobal()->getPointerAlignment(DL);
      if (Offset2 != 0 && Alignment <= Offset2)
        continue;
      int64_t Offset1 = GA->getOffset();
      int64_t CombinedOffset = Offset1 + Offset2;
      ImmOperand = CurDAG->getTargetGlobalAddress(
          GA->getGlobal(), SDLoc(ImmOperand), ImmOperand.getValueType(),
          CombinedOffset, GA->getTargetFlags());
    } else if (auto CP = dyn_cast<ConstantPoolSDNode>(ImmOperand)) {
      // Ditto.
      Align Alignment = CP->getAlign();
      if (Offset2 != 0 && Alignment <= Offset2)
        continue;
      int64_t Offset1 = CP->getOffset();
      int64_t CombinedOffset = Offset1 + Offset2;
      ImmOperand = CurDAG->getTargetConstantPool(
          CP->getConstVal(), ImmOperand.getValueType(), CP->getAlign(),
          CombinedOffset, CP->getTargetFlags());
    } else {
      continue;
    }

    LLVM_DEBUG(dbgs() << "Folding add-immediate into mem-op:\nBase:    ");
    LLVM_DEBUG(Base->dump(CurDAG));
    LLVM_DEBUG(dbgs() << "\nN: ");
    LLVM_DEBUG(N->dump(CurDAG));
    LLVM_DEBUG(dbgs() << "\n");

    // Modify the offset operand of the load/store.
    if (BaseOpIdx == 0) // Load
      CurDAG->UpdateNodeOperands(N, Base.getOperand(0), ImmOperand,
                                 N->getOperand(2));
    else // Store
      CurDAG->UpdateNodeOperands(N, N->getOperand(0), Base.getOperand(0),
                                 ImmOperand, N->getOperand(3));

    // The add-immediate may now be dead, in which case remove it.
    if (Base.getNode()->use_empty())
      CurDAG->RemoveDeadNode(Base.getNode());
  }
}

// This pass converts a legalized DAG into a RISCV-specific DAG, ready
// for instruction scheduling.
FunctionPass *llvm::createRISCVISelDag(RISCVTargetMachine &TM) {
  return new RISCVDAGToDAGISel(TM);
}
