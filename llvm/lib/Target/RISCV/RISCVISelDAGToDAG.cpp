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
#include "llvm/IR/IntrinsicsRISCV.h"
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
      SDValue New =
          CurDAG->getCopyFromReg(CurDAG->getEntryNode(), DL, RISCV::X0, XLenVT);
      ReplaceNode(Node, New.getNode());
      return;
    }
    int64_t Imm = ConstNode->getSExtValue();
    if (XLenVT == MVT::i64) {
      ReplaceNode(Node, selectImm(CurDAG, DL, Imm, XLenVT));
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
  case ISD::INTRINSIC_W_CHAIN: {
    unsigned IntNo = cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue();
    switch (IntNo) {
      // By default we do not custom select any intrinsic.
    default:
      break;

    case Intrinsic::riscv_vsetvli: {
      if (!Subtarget->hasStdExtV())
        break;

      assert(Node->getNumOperands() == 5);

      RISCVVSEW VSEW =
          static_cast<RISCVVSEW>(Node->getConstantOperandVal(3) & 0x7);
      RISCVVLMUL VLMul =
          static_cast<RISCVVLMUL>(Node->getConstantOperandVal(4) & 0x7);

      unsigned VTypeI = RISCVVType::encodeVTYPE(
          VLMul, VSEW, /*TailAgnostic*/ true, /*MaskAgnostic*/ false);
      SDValue VTypeIOp = CurDAG->getTargetConstant(VTypeI, DL, XLenVT);

      SDValue VLOperand = Node->getOperand(2);
      if (auto *C = dyn_cast<ConstantSDNode>(VLOperand)) {
        if (C->isNullValue()) {
          VLOperand = SDValue(
              CurDAG->getMachineNode(RISCV::ADDI, DL, XLenVT,
                                     CurDAG->getRegister(RISCV::X0, XLenVT),
                                     CurDAG->getTargetConstant(0, DL, XLenVT)),
              0);
        }
      }

      ReplaceNode(Node,
                  CurDAG->getMachineNode(RISCV::PseudoVSETVLI, DL, XLenVT,
                                         MVT::Other, VLOperand, VTypeIOp,
                                         /* Chain */ Node->getOperand(0)));
      return;
    }
    case Intrinsic::riscv_vsetvlimax: {
      if (!Subtarget->hasStdExtV())
        break;

      assert(Node->getNumOperands() == 4);

      RISCVVSEW VSEW =
          static_cast<RISCVVSEW>(Node->getConstantOperandVal(2) & 0x7);
      RISCVVLMUL VLMul =
          static_cast<RISCVVLMUL>(Node->getConstantOperandVal(3) & 0x7);

      unsigned VTypeI = RISCVVType::encodeVTYPE(
          VLMul, VSEW, /*TailAgnostic*/ true, /*MaskAgnostic*/ false);
      SDValue VTypeIOp = CurDAG->getTargetConstant(VTypeI, DL, XLenVT);

      SDValue VLOperand = CurDAG->getRegister(RISCV::X0, XLenVT);
      ReplaceNode(Node,
                  CurDAG->getMachineNode(RISCV::PseudoVSETVLI, DL, XLenVT,
                                         MVT::Other, VLOperand, VTypeIOp,
                                         /* Chain */ Node->getOperand(0)));
      return;
    }
    }
    break;
  }
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

// Match (srl (and val, mask), imm) where the result would be a
// zero-extended 32-bit integer. i.e. the mask is 0xffffffff or the result
// is equivalent to this (SimplifyDemandedBits may have removed lower bits
// from the mask that aren't necessary due to the right-shifting).
bool RISCVDAGToDAGISel::MatchSRLIW(SDNode *N) const {
  assert(N->getOpcode() == ISD::SRL);
  assert(N->getOperand(0).getOpcode() == ISD::AND);
  assert(isa<ConstantSDNode>(N->getOperand(1)));
  assert(isa<ConstantSDNode>(N->getOperand(0).getOperand(1)));

  // The IsRV64 predicate is checked after PatFrag predicates so we can get
  // here even on RV32.
  if (!Subtarget->is64Bit())
    return false;

  SDValue And = N->getOperand(0);
  uint64_t ShAmt = N->getConstantOperandVal(1);
  uint64_t Mask = And.getConstantOperandVal(1);
  return (Mask | maskTrailingOnes<uint64_t>(ShAmt)) == 0xffffffff;
}

// Check that it is a SLOI (Shift Left Ones Immediate). A PatFrag has already
// determined it has the right structure:
//
//  (OR (SHL RS1, VC2), VC1)
//
// Check that VC1, the mask used to fill with ones, is compatible
// with VC2, the shamt:
//
//  VC1 == maskTrailingOnes(VC2)
//
bool RISCVDAGToDAGISel::MatchSLOI(SDNode *N) const {
  assert(N->getOpcode() == ISD::OR);
  assert(N->getOperand(0).getOpcode() == ISD::SHL);
  assert(isa<ConstantSDNode>(N->getOperand(1)));
  assert(isa<ConstantSDNode>(N->getOperand(0).getOperand(1)));

  SDValue Shl = N->getOperand(0);
  if (Subtarget->is64Bit()) {
    uint64_t VC1 = N->getConstantOperandVal(1);
    uint64_t VC2 = Shl.getConstantOperandVal(1);
    return VC1 == maskTrailingOnes<uint64_t>(VC2);
  }

  uint32_t VC1 = N->getConstantOperandVal(1);
  uint32_t VC2 = Shl.getConstantOperandVal(1);
  return VC1 == maskTrailingOnes<uint32_t>(VC2);
}

// Check that it is a SROI (Shift Right Ones Immediate). A PatFrag has already
// determined it has the right structure:
//
//  (OR (SRL RS1, VC2), VC1)
//
// Check that VC1, the mask used to fill with ones, is compatible
// with VC2, the shamt:
//
//  VC1 == maskLeadingOnes(VC2)
//
bool RISCVDAGToDAGISel::MatchSROI(SDNode *N) const {
  assert(N->getOpcode() == ISD::OR);
  assert(N->getOperand(0).getOpcode() == ISD::SRL);
  assert(isa<ConstantSDNode>(N->getOperand(1)));
  assert(isa<ConstantSDNode>(N->getOperand(0).getOperand(1)));

  SDValue Srl = N->getOperand(0);
  if (Subtarget->is64Bit()) {
    uint64_t VC1 = N->getConstantOperandVal(1);
    uint64_t VC2 = Srl.getConstantOperandVal(1);
    return VC1 == maskLeadingOnes<uint64_t>(VC2);
  }

  uint32_t VC1 = N->getConstantOperandVal(1);
  uint32_t VC2 = Srl.getConstantOperandVal(1);
  return VC1 == maskLeadingOnes<uint32_t>(VC2);
}

// Check that it is a SROIW (Shift Right Ones Immediate i32 on RV64). A PatFrag
// has already determined it has the right structure:
//
//  (OR (SRL RS1, VC2), VC1)
//
// and then we check that VC1, the mask used to fill with ones, is compatible
// with VC2, the shamt:
//
//  VC2 < 32
//  VC1 == maskTrailingZeros<uint64_t>(32 - VC2)
//
bool RISCVDAGToDAGISel::MatchSROIW(SDNode *N) const {
  assert(N->getOpcode() == ISD::OR);
  assert(N->getOperand(0).getOpcode() == ISD::SRL);
  assert(isa<ConstantSDNode>(N->getOperand(1)));
  assert(isa<ConstantSDNode>(N->getOperand(0).getOperand(1)));

  // The IsRV64 predicate is checked after PatFrag predicates so we can get
  // here even on RV32.
  if (!Subtarget->is64Bit())
    return false;

  SDValue Srl = N->getOperand(0);
  uint64_t VC1 = N->getConstantOperandVal(1);
  uint64_t VC2 = Srl.getConstantOperandVal(1);

  // Immediate range should be enforced by uimm5 predicate.
  assert(VC2 < 32 && "Unexpected immediate");
  return VC1 == maskTrailingZeros<uint64_t>(32 - VC2);
}

// Check that it is a SLLIUW (Shift Logical Left Immediate Unsigned i32
// on RV64).
// SLLIUW is the same as SLLI except for the fact that it clears the bits
// XLEN-1:32 of the input RS1 before shifting.
// A PatFrag has already checked that it has the right structure:
//
//  (AND (SHL RS1, VC2), VC1)
//
// We check that VC2, the shamt is less than 32, otherwise the pattern is
// exactly the same as SLLI and we give priority to that.
// Eventually we check that VC1, the mask used to clear the upper 32 bits
// of RS1, is correct:
//
//  VC1 == (0xFFFFFFFF << VC2)
//
bool RISCVDAGToDAGISel::MatchSLLIUW(SDNode *N) const {
  assert(N->getOpcode() == ISD::AND);
  assert(N->getOperand(0).getOpcode() == ISD::SHL);
  assert(isa<ConstantSDNode>(N->getOperand(1)));
  assert(isa<ConstantSDNode>(N->getOperand(0).getOperand(1)));

  // The IsRV64 predicate is checked after PatFrag predicates so we can get
  // here even on RV32.
  if (!Subtarget->is64Bit())
    return false;

  SDValue Shl = N->getOperand(0);
  uint64_t VC1 = N->getConstantOperandVal(1);
  uint64_t VC2 = Shl.getConstantOperandVal(1);

  // Immediate range should be enforced by uimm5 predicate.
  assert(VC2 < 32 && "Unexpected immediate");
  return VC1 == ((uint64_t)0xFFFFFFFF << VC2);
}

bool RISCVDAGToDAGISel::selectVSplat(SDValue N, SDValue &SplatVal) {
  if (N.getOpcode() != ISD::SPLAT_VECTOR &&
      N.getOpcode() != RISCVISD::SPLAT_VECTOR_I64)
    return false;
  SplatVal = N.getOperand(0);
  return true;
}

bool RISCVDAGToDAGISel::selectVSplatSimm5(SDValue N, SDValue &SplatVal) {
  if ((N.getOpcode() != ISD::SPLAT_VECTOR &&
       N.getOpcode() != RISCVISD::SPLAT_VECTOR_I64) ||
      !isa<ConstantSDNode>(N.getOperand(0)))
    return false;

  int64_t SplatImm = cast<ConstantSDNode>(N.getOperand(0))->getSExtValue();

  // Both ISD::SPLAT_VECTOR and RISCVISD::SPLAT_VECTOR_I64 share semantics when
  // the operand type is wider than the resulting vector element type: an
  // implicit truncation first takes place. Therefore, perform a manual
  // truncation/sign-extension in order to ignore any truncated bits and catch
  // any zero-extended immediate.
  // For example, we wish to match (i8 -1) -> (XLenVT 255) as a simm5 by first
  // sign-extending to (XLenVT -1).
  auto XLenVT = Subtarget->getXLenVT();
  assert(XLenVT == N.getOperand(0).getSimpleValueType() &&
         "Unexpected splat operand type");
  auto EltVT = N.getValueType().getVectorElementType();
  if (EltVT.bitsLT(XLenVT)) {
    SplatImm = SignExtend64(SplatImm, EltVT.getSizeInBits());
  }

  if (!isInt<5>(SplatImm))
    return false;

  SplatVal = CurDAG->getTargetConstant(SplatImm, SDLoc(N), XLenVT);
  return true;
}

bool RISCVDAGToDAGISel::selectVSplatUimm5(SDValue N, SDValue &SplatVal) {
  if ((N.getOpcode() != ISD::SPLAT_VECTOR &&
       N.getOpcode() != RISCVISD::SPLAT_VECTOR_I64) ||
      !isa<ConstantSDNode>(N.getOperand(0)))
    return false;

  int64_t SplatImm = cast<ConstantSDNode>(N.getOperand(0))->getSExtValue();

  if (!isUInt<5>(SplatImm))
    return false;

  SplatVal =
      CurDAG->getTargetConstant(SplatImm, SDLoc(N), Subtarget->getXLenVT());

  return true;
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
    case RISCV::FLH:
    case RISCV::FLW:
    case RISCV::FLD:
      BaseOpIdx = 0;
      OffsetOpIdx = 1;
      break;
    case RISCV::SB:
    case RISCV::SH:
    case RISCV::SW:
    case RISCV::SD:
    case RISCV::FSH:
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
