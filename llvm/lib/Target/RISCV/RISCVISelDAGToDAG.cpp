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
#include "MCTargetDesc/RISCVMatInt.h"
#include "RISCVISelLowering.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-isel"

namespace llvm {
namespace RISCV {
#define GET_RISCVVSSEGTable_IMPL
#define GET_RISCVVLSEGTable_IMPL
#define GET_RISCVVLXSEGTable_IMPL
#define GET_RISCVVSXSEGTable_IMPL
#define GET_RISCVVLETable_IMPL
#define GET_RISCVVSETable_IMPL
#define GET_RISCVVLXTable_IMPL
#define GET_RISCVVSXTable_IMPL
#include "RISCVGenSearchableTables.inc"
} // namespace RISCV
} // namespace llvm

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

static SDValue createTupleImpl(SelectionDAG &CurDAG, ArrayRef<SDValue> Regs,
                               unsigned RegClassID, unsigned SubReg0) {
  assert(Regs.size() >= 2 && Regs.size() <= 8);

  SDLoc DL(Regs[0]);
  SmallVector<SDValue, 8> Ops;

  Ops.push_back(CurDAG.getTargetConstant(RegClassID, DL, MVT::i32));

  for (unsigned I = 0; I < Regs.size(); ++I) {
    Ops.push_back(Regs[I]);
    Ops.push_back(CurDAG.getTargetConstant(SubReg0 + I, DL, MVT::i32));
  }
  SDNode *N =
      CurDAG.getMachineNode(TargetOpcode::REG_SEQUENCE, DL, MVT::Untyped, Ops);
  return SDValue(N, 0);
}

static SDValue createM1Tuple(SelectionDAG &CurDAG, ArrayRef<SDValue> Regs,
                             unsigned NF) {
  static const unsigned RegClassIDs[] = {
      RISCV::VRN2M1RegClassID, RISCV::VRN3M1RegClassID, RISCV::VRN4M1RegClassID,
      RISCV::VRN5M1RegClassID, RISCV::VRN6M1RegClassID, RISCV::VRN7M1RegClassID,
      RISCV::VRN8M1RegClassID};

  return createTupleImpl(CurDAG, Regs, RegClassIDs[NF - 2], RISCV::sub_vrm1_0);
}

static SDValue createM2Tuple(SelectionDAG &CurDAG, ArrayRef<SDValue> Regs,
                             unsigned NF) {
  static const unsigned RegClassIDs[] = {RISCV::VRN2M2RegClassID,
                                         RISCV::VRN3M2RegClassID,
                                         RISCV::VRN4M2RegClassID};

  return createTupleImpl(CurDAG, Regs, RegClassIDs[NF - 2], RISCV::sub_vrm2_0);
}

static SDValue createM4Tuple(SelectionDAG &CurDAG, ArrayRef<SDValue> Regs,
                             unsigned NF) {
  return createTupleImpl(CurDAG, Regs, RISCV::VRN2M4RegClassID,
                         RISCV::sub_vrm4_0);
}

static SDValue createTuple(SelectionDAG &CurDAG, ArrayRef<SDValue> Regs,
                           unsigned NF, RISCVVLMUL LMUL) {
  switch (LMUL) {
  default:
    llvm_unreachable("Invalid LMUL.");
  case RISCVVLMUL::LMUL_F8:
  case RISCVVLMUL::LMUL_F4:
  case RISCVVLMUL::LMUL_F2:
  case RISCVVLMUL::LMUL_1:
    return createM1Tuple(CurDAG, Regs, NF);
  case RISCVVLMUL::LMUL_2:
    return createM2Tuple(CurDAG, Regs, NF);
  case RISCVVLMUL::LMUL_4:
    return createM4Tuple(CurDAG, Regs, NF);
  }
}

void RISCVDAGToDAGISel::selectVLSEG(SDNode *Node, bool IsMasked,
                                    bool IsStrided) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumValues() - 1;
  MVT VT = Node->getSimpleValueType(0);
  unsigned ScalarSize = VT.getScalarSizeInBits();
  MVT XLenVT = Subtarget->getXLenVT();
  RISCVVLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
  SDValue SEW = CurDAG->getTargetConstant(ScalarSize, DL, XLenVT);
  unsigned CurOp = 2;
  SmallVector<SDValue, 7> Operands;
  if (IsMasked) {
    SmallVector<SDValue, 8> Regs(Node->op_begin() + CurOp,
                                 Node->op_begin() + CurOp + NF);
    SDValue MaskedOff = createTuple(*CurDAG, Regs, NF, LMUL);
    Operands.push_back(MaskedOff);
    CurOp += NF;
  }
  SDValue Base;
  SelectBaseAddr(Node->getOperand(CurOp++), Base);
  Operands.push_back(Base); // Base pointer.
  if (IsStrided)
    Operands.push_back(Node->getOperand(CurOp++)); // Stride.
  if (IsMasked)
    Operands.push_back(Node->getOperand(CurOp++)); // Mask.
  SDValue VL;
  selectVLOp(Node->getOperand(CurOp++), VL);
  Operands.push_back(VL);
  Operands.push_back(SEW);
  Operands.push_back(Node->getOperand(0)); // Chain.
  const RISCV::VLSEGPseudo *P =
      RISCV::getVLSEGPseudo(NF, IsMasked, IsStrided, /*FF*/ false, ScalarSize,
                            static_cast<unsigned>(LMUL));
  MachineSDNode *Load =
      CurDAG->getMachineNode(P->Pseudo, DL, MVT::Untyped, MVT::Other, Operands);

  if (auto *MemOp = dyn_cast<MemSDNode>(Node))
    CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

  SDValue SuperReg = SDValue(Load, 0);
  for (unsigned I = 0; I < NF; ++I) {
    unsigned SubRegIdx = RISCVTargetLowering::getSubregIndexByMVT(VT, I);
    ReplaceUses(SDValue(Node, I),
                CurDAG->getTargetExtractSubreg(SubRegIdx, DL, VT, SuperReg));
  }

  ReplaceUses(SDValue(Node, NF), SDValue(Load, 1));
  CurDAG->RemoveDeadNode(Node);
}

void RISCVDAGToDAGISel::selectVLSEGFF(SDNode *Node, bool IsMasked) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumValues() - 2; // Do not count VL and Chain.
  MVT VT = Node->getSimpleValueType(0);
  MVT XLenVT = Subtarget->getXLenVT();
  unsigned ScalarSize = VT.getScalarSizeInBits();
  RISCVVLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
  SDValue SEW = CurDAG->getTargetConstant(ScalarSize, DL, XLenVT);

  unsigned CurOp = 2;
  SmallVector<SDValue, 7> Operands;
  if (IsMasked) {
    SmallVector<SDValue, 8> Regs(Node->op_begin() + CurOp,
                                 Node->op_begin() + CurOp + NF);
    SDValue MaskedOff = createTuple(*CurDAG, Regs, NF, LMUL);
    Operands.push_back(MaskedOff);
    CurOp += NF;
  }
  SDValue Base;
  SelectBaseAddr(Node->getOperand(CurOp++), Base);
  Operands.push_back(Base); // Base pointer.
  if (IsMasked)
    Operands.push_back(Node->getOperand(CurOp++)); // Mask.
  SDValue VL;
  selectVLOp(Node->getOperand(CurOp++), VL);
  Operands.push_back(VL);
  Operands.push_back(SEW);
  Operands.push_back(Node->getOperand(0)); // Chain.
  const RISCV::VLSEGPseudo *P =
      RISCV::getVLSEGPseudo(NF, IsMasked, /*Strided*/ false, /*FF*/ true,
                            ScalarSize, static_cast<unsigned>(LMUL));
  MachineSDNode *Load = CurDAG->getMachineNode(P->Pseudo, DL, MVT::Untyped,
                                               MVT::Other, MVT::Glue, Operands);
  SDNode *ReadVL = CurDAG->getMachineNode(RISCV::PseudoReadVL, DL, XLenVT,
                                          /*Glue*/ SDValue(Load, 2));

  if (auto *MemOp = dyn_cast<MemSDNode>(Node))
    CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

  SDValue SuperReg = SDValue(Load, 0);
  for (unsigned I = 0; I < NF; ++I) {
    unsigned SubRegIdx = RISCVTargetLowering::getSubregIndexByMVT(VT, I);
    ReplaceUses(SDValue(Node, I),
                CurDAG->getTargetExtractSubreg(SubRegIdx, DL, VT, SuperReg));
  }

  ReplaceUses(SDValue(Node, NF), SDValue(ReadVL, 0));   // VL
  ReplaceUses(SDValue(Node, NF + 1), SDValue(Load, 1)); // Chain
  CurDAG->RemoveDeadNode(Node);
}

void RISCVDAGToDAGISel::selectVLXSEG(SDNode *Node, bool IsMasked,
                                     bool IsOrdered) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumValues() - 1;
  MVT VT = Node->getSimpleValueType(0);
  unsigned ScalarSize = VT.getScalarSizeInBits();
  MVT XLenVT = Subtarget->getXLenVT();
  RISCVVLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
  SDValue SEW = CurDAG->getTargetConstant(ScalarSize, DL, XLenVT);
  unsigned CurOp = 2;
  SmallVector<SDValue, 7> Operands;
  if (IsMasked) {
    SmallVector<SDValue, 8> Regs(Node->op_begin() + CurOp,
                                 Node->op_begin() + CurOp + NF);
    SDValue MaskedOff = createTuple(*CurDAG, Regs, NF, LMUL);
    Operands.push_back(MaskedOff);
    CurOp += NF;
  }
  SDValue Base;
  SelectBaseAddr(Node->getOperand(CurOp++), Base);
  Operands.push_back(Base); // Base pointer.
  Operands.push_back(Node->getOperand(CurOp++)); // Index.
  MVT IndexVT = Operands.back()->getSimpleValueType(0);
  if (IsMasked)
    Operands.push_back(Node->getOperand(CurOp++)); // Mask.
  SDValue VL;
  selectVLOp(Node->getOperand(CurOp++), VL);
  Operands.push_back(VL);
  Operands.push_back(SEW);
  Operands.push_back(Node->getOperand(0)); // Chain.

  assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
         "Element count mismatch");

  RISCVVLMUL IndexLMUL = RISCVTargetLowering::getLMUL(IndexVT);
  unsigned IndexScalarSize = IndexVT.getScalarSizeInBits();
  const RISCV::VLXSEGPseudo *P = RISCV::getVLXSEGPseudo(
      NF, IsMasked, IsOrdered, IndexScalarSize, static_cast<unsigned>(LMUL),
      static_cast<unsigned>(IndexLMUL));
  MachineSDNode *Load =
      CurDAG->getMachineNode(P->Pseudo, DL, MVT::Untyped, MVT::Other, Operands);

  if (auto *MemOp = dyn_cast<MemSDNode>(Node))
    CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

  SDValue SuperReg = SDValue(Load, 0);
  for (unsigned I = 0; I < NF; ++I) {
    unsigned SubRegIdx = RISCVTargetLowering::getSubregIndexByMVT(VT, I);
    ReplaceUses(SDValue(Node, I),
                CurDAG->getTargetExtractSubreg(SubRegIdx, DL, VT, SuperReg));
  }

  ReplaceUses(SDValue(Node, NF), SDValue(Load, 1));
  CurDAG->RemoveDeadNode(Node);
}

void RISCVDAGToDAGISel::selectVSSEG(SDNode *Node, bool IsMasked,
                                    bool IsStrided) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumOperands() - 4;
  if (IsStrided)
    NF--;
  if (IsMasked)
    NF--;
  MVT VT = Node->getOperand(2)->getSimpleValueType(0);
  unsigned ScalarSize = VT.getScalarSizeInBits();
  MVT XLenVT = Subtarget->getXLenVT();
  RISCVVLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
  SDValue SEW = CurDAG->getTargetConstant(ScalarSize, DL, XLenVT);
  SmallVector<SDValue, 8> Regs(Node->op_begin() + 2, Node->op_begin() + 2 + NF);
  SDValue StoreVal = createTuple(*CurDAG, Regs, NF, LMUL);
  SmallVector<SDValue, 7> Operands;
  Operands.push_back(StoreVal);
  unsigned CurOp = 2 + NF;
  SDValue Base;
  SelectBaseAddr(Node->getOperand(CurOp++), Base);
  Operands.push_back(Base); // Base pointer.
  if (IsStrided)
    Operands.push_back(Node->getOperand(CurOp++)); // Stride.
  if (IsMasked)
    Operands.push_back(Node->getOperand(CurOp++)); // Mask.
  SDValue VL;
  selectVLOp(Node->getOperand(CurOp++), VL);
  Operands.push_back(VL);
  Operands.push_back(SEW);
  Operands.push_back(Node->getOperand(0)); // Chain.
  const RISCV::VSSEGPseudo *P = RISCV::getVSSEGPseudo(
      NF, IsMasked, IsStrided, ScalarSize, static_cast<unsigned>(LMUL));
  MachineSDNode *Store =
      CurDAG->getMachineNode(P->Pseudo, DL, Node->getValueType(0), Operands);

  if (auto *MemOp = dyn_cast<MemSDNode>(Node))
    CurDAG->setNodeMemRefs(Store, {MemOp->getMemOperand()});

  ReplaceNode(Node, Store);
}

void RISCVDAGToDAGISel::selectVSXSEG(SDNode *Node, bool IsMasked,
                                     bool IsOrdered) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumOperands() - 5;
  if (IsMasked)
    --NF;
  MVT VT = Node->getOperand(2)->getSimpleValueType(0);
  unsigned ScalarSize = VT.getScalarSizeInBits();
  MVT XLenVT = Subtarget->getXLenVT();
  RISCVVLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
  SDValue SEW = CurDAG->getTargetConstant(ScalarSize, DL, XLenVT);
  SmallVector<SDValue, 7> Operands;
  SmallVector<SDValue, 8> Regs(Node->op_begin() + 2, Node->op_begin() + 2 + NF);
  SDValue StoreVal = createTuple(*CurDAG, Regs, NF, LMUL);
  Operands.push_back(StoreVal);
  unsigned CurOp = 2 + NF;
  SDValue Base;
  SelectBaseAddr(Node->getOperand(CurOp++), Base);
  Operands.push_back(Base); // Base pointer.
  Operands.push_back(Node->getOperand(CurOp++)); // Index.
  MVT IndexVT = Operands.back()->getSimpleValueType(0);
  if (IsMasked)
    Operands.push_back(Node->getOperand(CurOp++)); // Mask.
  SDValue VL;
  selectVLOp(Node->getOperand(CurOp++), VL);
  Operands.push_back(VL);
  Operands.push_back(SEW);
  Operands.push_back(Node->getOperand(0)); // Chain.

  assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
         "Element count mismatch");

  RISCVVLMUL IndexLMUL = RISCVTargetLowering::getLMUL(IndexVT);
  unsigned IndexScalarSize = IndexVT.getScalarSizeInBits();
  const RISCV::VSXSEGPseudo *P = RISCV::getVSXSEGPseudo(
      NF, IsMasked, IsOrdered, IndexScalarSize, static_cast<unsigned>(LMUL),
      static_cast<unsigned>(IndexLMUL));
  MachineSDNode *Store =
      CurDAG->getMachineNode(P->Pseudo, DL, Node->getValueType(0), Operands);

  if (auto *MemOp = dyn_cast<MemSDNode>(Node))
    CurDAG->setNodeMemRefs(Store, {MemOp->getMemOperand()});

  ReplaceNode(Node, Store);
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
  MVT VT = Node->getSimpleValueType(0);

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
    auto *ConstNode = cast<ConstantSDNode>(Node);
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
  case ISD::SRL: {
    // Optimize (srl (and X, 0xffff), C) -> (srli (slli X, 16), 16 + C).
    // Taking into account that the 0xffff may have had lower bits unset by
    // SimplifyDemandedBits. This avoids materializing the 0xffff immediate.
    // This pattern occurs when type legalizing i16 right shifts.
    // FIXME: This could be extended to other AND masks.
    auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
    if (N1C) {
      uint64_t ShAmt = N1C->getZExtValue();
      SDValue N0 = Node->getOperand(0);
      if (ShAmt < 16 && N0.getOpcode() == ISD::AND && N0.hasOneUse() &&
          isa<ConstantSDNode>(N0.getOperand(1))) {
        uint64_t Mask = N0.getConstantOperandVal(1);
        Mask |= maskTrailingOnes<uint64_t>(ShAmt);
        if (Mask == 0xffff) {
          unsigned SLLOpc = Subtarget->is64Bit() ? RISCV::SLLIW : RISCV::SLLI;
          unsigned SRLOpc = Subtarget->is64Bit() ? RISCV::SRLIW : RISCV::SRLI;
          SDNode *SLLI =
              CurDAG->getMachineNode(SLLOpc, DL, VT, N0->getOperand(0),
                                     CurDAG->getTargetConstant(16, DL, VT));
          SDNode *SRLI = CurDAG->getMachineNode(
              SRLOpc, DL, VT, SDValue(SLLI, 0),
              CurDAG->getTargetConstant(16 + ShAmt, DL, VT));
          ReplaceNode(Node, SRLI);
          return;
        }
      }
    }

    break;
  }
  case ISD::INTRINSIC_W_CHAIN: {
    unsigned IntNo = cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue();
    switch (IntNo) {
      // By default we do not custom select any intrinsic.
    default:
      break;

    case Intrinsic::riscv_vsetvli:
    case Intrinsic::riscv_vsetvlimax: {
      if (!Subtarget->hasStdExtV())
        break;

      bool VLMax = IntNo == Intrinsic::riscv_vsetvlimax;
      unsigned Offset = VLMax ? 2 : 3;

      assert(Node->getNumOperands() == Offset + 2 &&
             "Unexpected number of operands");

      RISCVVSEW VSEW =
          static_cast<RISCVVSEW>(Node->getConstantOperandVal(Offset) & 0x7);
      RISCVVLMUL VLMul = static_cast<RISCVVLMUL>(
          Node->getConstantOperandVal(Offset + 1) & 0x7);

      unsigned VTypeI = RISCVVType::encodeVTYPE(
          VLMul, VSEW, /*TailAgnostic*/ true, /*MaskAgnostic*/ false);
      SDValue VTypeIOp = CurDAG->getTargetConstant(VTypeI, DL, XLenVT);

      SDValue VLOperand;
      if (VLMax) {
        VLOperand = CurDAG->getRegister(RISCV::X0, XLenVT);
      } else {
        VLOperand = Node->getOperand(2);

        if (auto *C = dyn_cast<ConstantSDNode>(VLOperand)) {
          uint64_t AVL = C->getZExtValue();
          if (isUInt<5>(AVL)) {
            SDValue VLImm = CurDAG->getTargetConstant(AVL, DL, XLenVT);
            ReplaceNode(
                Node, CurDAG->getMachineNode(RISCV::PseudoVSETIVLI, DL, XLenVT,
                                             MVT::Other, VLImm, VTypeIOp,
                                             /* Chain */ Node->getOperand(0)));
            return;
          }
        }
      }

      ReplaceNode(Node,
                  CurDAG->getMachineNode(RISCV::PseudoVSETVLI, DL, XLenVT,
                                         MVT::Other, VLOperand, VTypeIOp,
                                         /* Chain */ Node->getOperand(0)));
      return;
    }
    case Intrinsic::riscv_vlseg2:
    case Intrinsic::riscv_vlseg3:
    case Intrinsic::riscv_vlseg4:
    case Intrinsic::riscv_vlseg5:
    case Intrinsic::riscv_vlseg6:
    case Intrinsic::riscv_vlseg7:
    case Intrinsic::riscv_vlseg8: {
      selectVLSEG(Node, /*IsMasked*/ false, /*IsStrided*/ false);
      return;
    }
    case Intrinsic::riscv_vlseg2_mask:
    case Intrinsic::riscv_vlseg3_mask:
    case Intrinsic::riscv_vlseg4_mask:
    case Intrinsic::riscv_vlseg5_mask:
    case Intrinsic::riscv_vlseg6_mask:
    case Intrinsic::riscv_vlseg7_mask:
    case Intrinsic::riscv_vlseg8_mask: {
      selectVLSEG(Node, /*IsMasked*/ true, /*IsStrided*/ false);
      return;
    }
    case Intrinsic::riscv_vlsseg2:
    case Intrinsic::riscv_vlsseg3:
    case Intrinsic::riscv_vlsseg4:
    case Intrinsic::riscv_vlsseg5:
    case Intrinsic::riscv_vlsseg6:
    case Intrinsic::riscv_vlsseg7:
    case Intrinsic::riscv_vlsseg8: {
      selectVLSEG(Node, /*IsMasked*/ false, /*IsStrided*/ true);
      return;
    }
    case Intrinsic::riscv_vlsseg2_mask:
    case Intrinsic::riscv_vlsseg3_mask:
    case Intrinsic::riscv_vlsseg4_mask:
    case Intrinsic::riscv_vlsseg5_mask:
    case Intrinsic::riscv_vlsseg6_mask:
    case Intrinsic::riscv_vlsseg7_mask:
    case Intrinsic::riscv_vlsseg8_mask: {
      selectVLSEG(Node, /*IsMasked*/ true, /*IsStrided*/ true);
      return;
    }
    case Intrinsic::riscv_vloxseg2:
    case Intrinsic::riscv_vloxseg3:
    case Intrinsic::riscv_vloxseg4:
    case Intrinsic::riscv_vloxseg5:
    case Intrinsic::riscv_vloxseg6:
    case Intrinsic::riscv_vloxseg7:
    case Intrinsic::riscv_vloxseg8:
      selectVLXSEG(Node, /*IsMasked*/ false, /*IsOrdered*/ true);
      return;
    case Intrinsic::riscv_vluxseg2:
    case Intrinsic::riscv_vluxseg3:
    case Intrinsic::riscv_vluxseg4:
    case Intrinsic::riscv_vluxseg5:
    case Intrinsic::riscv_vluxseg6:
    case Intrinsic::riscv_vluxseg7:
    case Intrinsic::riscv_vluxseg8:
      selectVLXSEG(Node, /*IsMasked*/ false, /*IsOrdered*/ false);
      return;
    case Intrinsic::riscv_vloxseg2_mask:
    case Intrinsic::riscv_vloxseg3_mask:
    case Intrinsic::riscv_vloxseg4_mask:
    case Intrinsic::riscv_vloxseg5_mask:
    case Intrinsic::riscv_vloxseg6_mask:
    case Intrinsic::riscv_vloxseg7_mask:
    case Intrinsic::riscv_vloxseg8_mask:
      selectVLXSEG(Node, /*IsMasked*/ true, /*IsOrdered*/ true);
      return;
    case Intrinsic::riscv_vluxseg2_mask:
    case Intrinsic::riscv_vluxseg3_mask:
    case Intrinsic::riscv_vluxseg4_mask:
    case Intrinsic::riscv_vluxseg5_mask:
    case Intrinsic::riscv_vluxseg6_mask:
    case Intrinsic::riscv_vluxseg7_mask:
    case Intrinsic::riscv_vluxseg8_mask:
      selectVLXSEG(Node, /*IsMasked*/ true, /*IsOrdered*/ false);
      return;
    case Intrinsic::riscv_vlseg8ff:
    case Intrinsic::riscv_vlseg7ff:
    case Intrinsic::riscv_vlseg6ff:
    case Intrinsic::riscv_vlseg5ff:
    case Intrinsic::riscv_vlseg4ff:
    case Intrinsic::riscv_vlseg3ff:
    case Intrinsic::riscv_vlseg2ff: {
      selectVLSEGFF(Node, /*IsMasked*/ false);
      return;
    }
    case Intrinsic::riscv_vlseg8ff_mask:
    case Intrinsic::riscv_vlseg7ff_mask:
    case Intrinsic::riscv_vlseg6ff_mask:
    case Intrinsic::riscv_vlseg5ff_mask:
    case Intrinsic::riscv_vlseg4ff_mask:
    case Intrinsic::riscv_vlseg3ff_mask:
    case Intrinsic::riscv_vlseg2ff_mask: {
      selectVLSEGFF(Node, /*IsMasked*/ true);
      return;
    }
    case Intrinsic::riscv_vloxei:
    case Intrinsic::riscv_vloxei_mask:
    case Intrinsic::riscv_vluxei:
    case Intrinsic::riscv_vluxei_mask: {
      bool IsMasked = IntNo == Intrinsic::riscv_vloxei_mask ||
                      IntNo == Intrinsic::riscv_vluxei_mask;
      bool IsOrdered = IntNo == Intrinsic::riscv_vloxei ||
                       IntNo == Intrinsic::riscv_vloxei_mask;

      MVT VT = Node->getSimpleValueType(0);
      unsigned ScalarSize = VT.getScalarSizeInBits();
      MVT XLenVT = Subtarget->getXLenVT();
      SDValue SEW = CurDAG->getTargetConstant(ScalarSize, DL, XLenVT);

      unsigned CurOp = 2;
      SmallVector<SDValue, 7> Operands;
      if (IsMasked)
        Operands.push_back(Node->getOperand(CurOp++));
      SDValue Base;
      SelectBaseAddr(Node->getOperand(CurOp++), Base);
      Operands.push_back(Base); // Base pointer.
      Operands.push_back(Node->getOperand(CurOp++)); // Index.
      MVT IndexVT = Operands.back()->getSimpleValueType(0);
      if (IsMasked)
        Operands.push_back(Node->getOperand(CurOp++)); // Mask.
      SDValue VL;
      selectVLOp(Node->getOperand(CurOp++), VL);
      Operands.push_back(VL);
      Operands.push_back(SEW);
      Operands.push_back(Node->getOperand(0)); // Chain.

      assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
             "Element count mismatch");

      RISCVVLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
      RISCVVLMUL IndexLMUL = RISCVTargetLowering::getLMUL(IndexVT);
      unsigned IndexScalarSize = IndexVT.getScalarSizeInBits();
      const RISCV::VLX_VSXPseudo *P = RISCV::getVLXPseudo(
          IsMasked, IsOrdered, IndexScalarSize, static_cast<unsigned>(LMUL),
          static_cast<unsigned>(IndexLMUL));
      MachineSDNode *Load =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);

      if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

      ReplaceNode(Node, Load);
      return;
    }
    case Intrinsic::riscv_vle1:
    case Intrinsic::riscv_vle:
    case Intrinsic::riscv_vle_mask:
    case Intrinsic::riscv_vlse:
    case Intrinsic::riscv_vlse_mask: {
      bool IsMasked = IntNo == Intrinsic::riscv_vle_mask ||
                      IntNo == Intrinsic::riscv_vlse_mask;
      bool IsStrided =
          IntNo == Intrinsic::riscv_vlse || IntNo == Intrinsic::riscv_vlse_mask;

      MVT VT = Node->getSimpleValueType(0);
      unsigned ScalarSize = VT.getScalarSizeInBits();
      MVT XLenVT = Subtarget->getXLenVT();
      // VLE1 uses an SEW of 8.
      unsigned SEWImm = (IntNo == Intrinsic::riscv_vle1) ? 8 : ScalarSize;
      SDValue SEW = CurDAG->getTargetConstant(SEWImm, DL, XLenVT);

      unsigned CurOp = 2;
      SmallVector<SDValue, 7> Operands;
      if (IsMasked)
        Operands.push_back(Node->getOperand(CurOp++));
      SDValue Base;
      SelectBaseAddr(Node->getOperand(CurOp++), Base);
      Operands.push_back(Base); // Base pointer.
      if (IsStrided)
        Operands.push_back(Node->getOperand(CurOp++)); // Stride.
      if (IsMasked)
        Operands.push_back(Node->getOperand(CurOp++)); // Mask.
      SDValue VL;
      selectVLOp(Node->getOperand(CurOp++), VL);
      Operands.push_back(VL);
      Operands.push_back(SEW);
      Operands.push_back(Node->getOperand(0)); // Chain.

      RISCVVLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
      const RISCV::VLEPseudo *P =
          RISCV::getVLEPseudo(IsMasked, IsStrided, /*FF*/ false, ScalarSize,
                              static_cast<unsigned>(LMUL));
      MachineSDNode *Load =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);

      if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

      ReplaceNode(Node, Load);
      return;
    }
    case Intrinsic::riscv_vleff:
    case Intrinsic::riscv_vleff_mask: {
      bool IsMasked = IntNo == Intrinsic::riscv_vleff_mask;

      MVT VT = Node->getSimpleValueType(0);
      unsigned ScalarSize = VT.getScalarSizeInBits();
      MVT XLenVT = Subtarget->getXLenVT();
      SDValue SEW = CurDAG->getTargetConstant(ScalarSize, DL, XLenVT);

      unsigned CurOp = 2;
      SmallVector<SDValue, 7> Operands;
      if (IsMasked)
        Operands.push_back(Node->getOperand(CurOp++));
      SDValue Base;
      SelectBaseAddr(Node->getOperand(CurOp++), Base);
      Operands.push_back(Base); // Base pointer.
      if (IsMasked)
        Operands.push_back(Node->getOperand(CurOp++)); // Mask.
      SDValue VL;
      selectVLOp(Node->getOperand(CurOp++), VL);
      Operands.push_back(VL);
      Operands.push_back(SEW);
      Operands.push_back(Node->getOperand(0)); // Chain.

      RISCVVLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
      const RISCV::VLEPseudo *P =
          RISCV::getVLEPseudo(IsMasked, /*Strided*/ false, /*FF*/ true,
                              ScalarSize, static_cast<unsigned>(LMUL));
      MachineSDNode *Load =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getValueType(0),
                                 MVT::Other, MVT::Glue, Operands);
      SDNode *ReadVL = CurDAG->getMachineNode(RISCV::PseudoReadVL, DL, XLenVT,
                                              /*Glue*/ SDValue(Load, 2));

      if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

      ReplaceUses(SDValue(Node, 0), SDValue(Load, 0));
      ReplaceUses(SDValue(Node, 1), SDValue(ReadVL, 0)); // VL
      ReplaceUses(SDValue(Node, 2), SDValue(Load, 1));   // Chain
      CurDAG->RemoveDeadNode(Node);
      return;
    }
    }
    break;
  }
  case ISD::INTRINSIC_VOID: {
    unsigned IntNo = cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue();
    switch (IntNo) {
    case Intrinsic::riscv_vsseg2:
    case Intrinsic::riscv_vsseg3:
    case Intrinsic::riscv_vsseg4:
    case Intrinsic::riscv_vsseg5:
    case Intrinsic::riscv_vsseg6:
    case Intrinsic::riscv_vsseg7:
    case Intrinsic::riscv_vsseg8: {
      selectVSSEG(Node, /*IsMasked*/ false, /*IsStrided*/ false);
      return;
    }
    case Intrinsic::riscv_vsseg2_mask:
    case Intrinsic::riscv_vsseg3_mask:
    case Intrinsic::riscv_vsseg4_mask:
    case Intrinsic::riscv_vsseg5_mask:
    case Intrinsic::riscv_vsseg6_mask:
    case Intrinsic::riscv_vsseg7_mask:
    case Intrinsic::riscv_vsseg8_mask: {
      selectVSSEG(Node, /*IsMasked*/ true, /*IsStrided*/ false);
      return;
    }
    case Intrinsic::riscv_vssseg2:
    case Intrinsic::riscv_vssseg3:
    case Intrinsic::riscv_vssseg4:
    case Intrinsic::riscv_vssseg5:
    case Intrinsic::riscv_vssseg6:
    case Intrinsic::riscv_vssseg7:
    case Intrinsic::riscv_vssseg8: {
      selectVSSEG(Node, /*IsMasked*/ false, /*IsStrided*/ true);
      return;
    }
    case Intrinsic::riscv_vssseg2_mask:
    case Intrinsic::riscv_vssseg3_mask:
    case Intrinsic::riscv_vssseg4_mask:
    case Intrinsic::riscv_vssseg5_mask:
    case Intrinsic::riscv_vssseg6_mask:
    case Intrinsic::riscv_vssseg7_mask:
    case Intrinsic::riscv_vssseg8_mask: {
      selectVSSEG(Node, /*IsMasked*/ true, /*IsStrided*/ true);
      return;
    }
    case Intrinsic::riscv_vsoxseg2:
    case Intrinsic::riscv_vsoxseg3:
    case Intrinsic::riscv_vsoxseg4:
    case Intrinsic::riscv_vsoxseg5:
    case Intrinsic::riscv_vsoxseg6:
    case Intrinsic::riscv_vsoxseg7:
    case Intrinsic::riscv_vsoxseg8:
      selectVSXSEG(Node, /*IsMasked*/ false, /*IsOrdered*/ true);
      return;
    case Intrinsic::riscv_vsuxseg2:
    case Intrinsic::riscv_vsuxseg3:
    case Intrinsic::riscv_vsuxseg4:
    case Intrinsic::riscv_vsuxseg5:
    case Intrinsic::riscv_vsuxseg6:
    case Intrinsic::riscv_vsuxseg7:
    case Intrinsic::riscv_vsuxseg8:
      selectVSXSEG(Node, /*IsMasked*/ false, /*IsOrdered*/ false);
      return;
    case Intrinsic::riscv_vsoxseg2_mask:
    case Intrinsic::riscv_vsoxseg3_mask:
    case Intrinsic::riscv_vsoxseg4_mask:
    case Intrinsic::riscv_vsoxseg5_mask:
    case Intrinsic::riscv_vsoxseg6_mask:
    case Intrinsic::riscv_vsoxseg7_mask:
    case Intrinsic::riscv_vsoxseg8_mask:
      selectVSXSEG(Node, /*IsMasked*/ true, /*IsOrdered*/ true);
      return;
    case Intrinsic::riscv_vsuxseg2_mask:
    case Intrinsic::riscv_vsuxseg3_mask:
    case Intrinsic::riscv_vsuxseg4_mask:
    case Intrinsic::riscv_vsuxseg5_mask:
    case Intrinsic::riscv_vsuxseg6_mask:
    case Intrinsic::riscv_vsuxseg7_mask:
    case Intrinsic::riscv_vsuxseg8_mask:
      selectVSXSEG(Node, /*IsMasked*/ true, /*IsOrdered*/ false);
      return;
    case Intrinsic::riscv_vsoxei:
    case Intrinsic::riscv_vsoxei_mask:
    case Intrinsic::riscv_vsuxei:
    case Intrinsic::riscv_vsuxei_mask: {
      bool IsMasked = IntNo == Intrinsic::riscv_vsoxei_mask ||
                      IntNo == Intrinsic::riscv_vsuxei_mask;
      bool IsOrdered = IntNo == Intrinsic::riscv_vsoxei ||
                       IntNo == Intrinsic::riscv_vsoxei_mask;

      MVT VT = Node->getOperand(2)->getSimpleValueType(0);
      unsigned ScalarSize = VT.getScalarSizeInBits();
      MVT XLenVT = Subtarget->getXLenVT();
      SDValue SEW = CurDAG->getTargetConstant(ScalarSize, DL, XLenVT);

      unsigned CurOp = 2;
      SmallVector<SDValue, 6> Operands;
      Operands.push_back(Node->getOperand(CurOp++)); // Store value.
      SDValue Base;
      SelectBaseAddr(Node->getOperand(CurOp++), Base);
      Operands.push_back(Base); // Base pointer.
      Operands.push_back(Node->getOperand(CurOp++)); // Index.
      MVT IndexVT = Operands.back()->getSimpleValueType(0);
      if (IsMasked)
        Operands.push_back(Node->getOperand(CurOp++)); // Mask.
      SDValue VL;
      selectVLOp(Node->getOperand(CurOp++), VL);
      Operands.push_back(VL);
      Operands.push_back(SEW);
      Operands.push_back(Node->getOperand(0)); // Chain.

      assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
             "Element count mismatch");

      RISCVVLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
      RISCVVLMUL IndexLMUL = RISCVTargetLowering::getLMUL(IndexVT);
      unsigned IndexScalarSize = IndexVT.getScalarSizeInBits();
      const RISCV::VLX_VSXPseudo *P = RISCV::getVSXPseudo(
          IsMasked, IsOrdered, IndexScalarSize, static_cast<unsigned>(LMUL),
          static_cast<unsigned>(IndexLMUL));
      MachineSDNode *Store =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);

      if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        CurDAG->setNodeMemRefs(Store, {MemOp->getMemOperand()});

      ReplaceNode(Node, Store);
      return;
    }
    case Intrinsic::riscv_vse1:
    case Intrinsic::riscv_vse:
    case Intrinsic::riscv_vse_mask:
    case Intrinsic::riscv_vsse:
    case Intrinsic::riscv_vsse_mask: {
      bool IsMasked = IntNo == Intrinsic::riscv_vse_mask ||
                      IntNo == Intrinsic::riscv_vsse_mask;
      bool IsStrided =
          IntNo == Intrinsic::riscv_vsse || IntNo == Intrinsic::riscv_vsse_mask;

      MVT VT = Node->getOperand(2)->getSimpleValueType(0);
      unsigned ScalarSize = VT.getScalarSizeInBits();
      MVT XLenVT = Subtarget->getXLenVT();
      // VSE1 uses an SEW of 8.
      unsigned SEWImm = (IntNo == Intrinsic::riscv_vse1) ? 8 : ScalarSize;
      SDValue SEW = CurDAG->getTargetConstant(SEWImm, DL, XLenVT);

      unsigned CurOp = 2;
      SmallVector<SDValue, 6> Operands;
      Operands.push_back(Node->getOperand(CurOp++)); // Store value.
      SDValue Base;
      SelectBaseAddr(Node->getOperand(CurOp++), Base);
      Operands.push_back(Base); // Base pointer.
      if (IsStrided)
        Operands.push_back(Node->getOperand(CurOp++)); // Stride.
      if (IsMasked)
        Operands.push_back(Node->getOperand(CurOp++)); // Mask.
      SDValue VL;
      selectVLOp(Node->getOperand(CurOp++), VL);
      Operands.push_back(VL);
      Operands.push_back(SEW);
      Operands.push_back(Node->getOperand(0)); // Chain.

      RISCVVLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
      const RISCV::VSEPseudo *P = RISCV::getVSEPseudo(
          IsMasked, IsStrided, ScalarSize, static_cast<unsigned>(LMUL));
      MachineSDNode *Store =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);

      if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        CurDAG->setNodeMemRefs(Store, {MemOp->getMemOperand()});

      ReplaceNode(Node, Store);
      return;
    }
    }
    break;
  }
  case ISD::BITCAST: {
    MVT SrcVT = Node->getOperand(0).getSimpleValueType();
    // Just drop bitcasts between vectors if both are fixed or both are
    // scalable.
    if ((VT.isScalableVector() && SrcVT.isScalableVector()) ||
        (VT.isFixedLengthVector() && SrcVT.isFixedLengthVector())) {
      ReplaceUses(SDValue(Node, 0), Node->getOperand(0));
      CurDAG->RemoveDeadNode(Node);
      return;
    }
    break;
  }
  case ISD::INSERT_SUBVECTOR: {
    SDValue V = Node->getOperand(0);
    SDValue SubV = Node->getOperand(1);
    SDLoc DL(SubV);
    auto Idx = Node->getConstantOperandVal(2);
    MVT SubVecVT = SubV.getSimpleValueType();

    MVT SubVecContainerVT = SubVecVT;
    // Establish the correct scalable-vector types for any fixed-length type.
    if (SubVecVT.isFixedLengthVector())
      SubVecContainerVT = RISCVTargetLowering::getContainerForFixedLengthVector(
          *CurDAG, SubVecVT, *Subtarget);
    if (VT.isFixedLengthVector())
      VT = RISCVTargetLowering::getContainerForFixedLengthVector(*CurDAG, VT,
                                                                 *Subtarget);

    const auto *TRI = Subtarget->getRegisterInfo();
    unsigned SubRegIdx;
    std::tie(SubRegIdx, Idx) =
        RISCVTargetLowering::decomposeSubvectorInsertExtractToSubRegs(
            VT, SubVecContainerVT, Idx, TRI);

    // If the Idx hasn't been completely eliminated then this is a subvector
    // insert which doesn't naturally align to a vector register. These must
    // be handled using instructions to manipulate the vector registers.
    if (Idx != 0)
      break;

    RISCVVLMUL SubVecLMUL = RISCVTargetLowering::getLMUL(SubVecVT);
    bool IsSubVecPartReg = SubVecLMUL == RISCVVLMUL::LMUL_F2 ||
                           SubVecLMUL == RISCVVLMUL::LMUL_F4 ||
                           SubVecLMUL == RISCVVLMUL::LMUL_F8;
    (void)IsSubVecPartReg; // Silence unused variable warning without asserts.
    assert((!IsSubVecPartReg || V.isUndef()) &&
           "Expecting lowering to have created legal INSERT_SUBVECTORs when "
           "the subvector is smaller than a full-sized register");

    // If we haven't set a SubRegIdx, then we must be going between
    // equally-sized LMUL groups (e.g. VR -> VR). This can be done as a copy.
    if (SubRegIdx == RISCV::NoSubRegister) {
      unsigned InRegClassID = RISCVTargetLowering::getRegClassIDForVecVT(VT);
      assert(RISCVTargetLowering::getRegClassIDForVecVT(SubVecContainerVT) ==
                 InRegClassID &&
             "Unexpected subvector extraction");
      SDValue RC = CurDAG->getTargetConstant(InRegClassID, DL, XLenVT);
      SDNode *NewNode = CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS,
                                               DL, VT, SubV, RC);
      return ReplaceNode(Node, NewNode);
    }

    SDNode *NewNode = CurDAG->getMachineNode(
        TargetOpcode::INSERT_SUBREG, DL, VT, V, SubV,
        CurDAG->getTargetConstant(SubRegIdx, DL, XLenVT));
    return ReplaceNode(Node, NewNode);
  }
  case ISD::EXTRACT_SUBVECTOR: {
    SDValue V = Node->getOperand(0);
    auto Idx = Node->getConstantOperandVal(1);
    MVT InVT = V.getSimpleValueType();
    SDLoc DL(V);

    MVT SubVecContainerVT = VT;
    // Establish the correct scalable-vector types for any fixed-length type.
    if (VT.isFixedLengthVector())
      SubVecContainerVT = RISCVTargetLowering::getContainerForFixedLengthVector(
          *CurDAG, VT, *Subtarget);
    if (InVT.isFixedLengthVector())
      InVT = RISCVTargetLowering::getContainerForFixedLengthVector(
          *CurDAG, InVT, *Subtarget);

    const auto *TRI = Subtarget->getRegisterInfo();
    unsigned SubRegIdx;
    std::tie(SubRegIdx, Idx) =
        RISCVTargetLowering::decomposeSubvectorInsertExtractToSubRegs(
            InVT, SubVecContainerVT, Idx, TRI);

    // If the Idx hasn't been completely eliminated then this is a subvector
    // extract which doesn't naturally align to a vector register. These must
    // be handled using instructions to manipulate the vector registers.
    if (Idx != 0)
      break;

    // If we haven't set a SubRegIdx, then we must be going between
    // equally-sized LMUL types (e.g. VR -> VR). This can be done as a copy.
    if (SubRegIdx == RISCV::NoSubRegister) {
      unsigned InRegClassID = RISCVTargetLowering::getRegClassIDForVecVT(InVT);
      assert(RISCVTargetLowering::getRegClassIDForVecVT(SubVecContainerVT) ==
                 InRegClassID &&
             "Unexpected subvector extraction");
      SDValue RC = CurDAG->getTargetConstant(InRegClassID, DL, XLenVT);
      SDNode *NewNode =
          CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS, DL, VT, V, RC);
      return ReplaceNode(Node, NewNode);
    }

    SDNode *NewNode = CurDAG->getMachineNode(
        TargetOpcode::EXTRACT_SUBREG, DL, VT, V,
        CurDAG->getTargetConstant(SubRegIdx, DL, XLenVT));
    return ReplaceNode(Node, NewNode);
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
  if (auto *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), Subtarget->getXLenVT());
    return true;
  }
  return false;
}

bool RISCVDAGToDAGISel::SelectBaseAddr(SDValue Addr, SDValue &Base) {
  // If this is FrameIndex, select it directly. Otherwise just let it get
  // selected to a register independently.
  if (auto *FIN = dyn_cast<FrameIndexSDNode>(Addr))
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), Subtarget->getXLenVT());
  else
    Base = Addr;
  return true;
}

bool RISCVDAGToDAGISel::selectShiftMask(SDValue N, unsigned ShiftWidth,
                                        SDValue &ShAmt) {
  // Shift instructions on RISCV only read the lower 5 or 6 bits of the shift
  // amount. If there is an AND on the shift amount, we can bypass it if it
  // doesn't affect any of those bits.
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
  }

  ShAmt = N;
  return true;
}

bool RISCVDAGToDAGISel::selectSExti32(SDValue N, SDValue &Val) {
  if (N.getOpcode() == ISD::SIGN_EXTEND_INREG &&
      cast<VTSDNode>(N.getOperand(1))->getVT() == MVT::i32) {
    Val = N.getOperand(0);
    return true;
  }
  // FIXME: Should we just call computeNumSignBits here?
  if (N.getOpcode() == ISD::AssertSext &&
      cast<VTSDNode>(N->getOperand(1))->getVT().bitsLE(MVT::i32)) {
    Val = N;
    return true;
  }
  if (N.getOpcode() == ISD::AssertZext &&
      cast<VTSDNode>(N->getOperand(1))->getVT().bitsLT(MVT::i32)) {
    Val = N;
    return true;
  }

  return false;
}

bool RISCVDAGToDAGISel::selectZExti32(SDValue N, SDValue &Val) {
  if (N.getOpcode() == ISD::AND) {
    auto *C = dyn_cast<ConstantSDNode>(N.getOperand(1));
    if (C && CheckAndMask(N.getOperand(0), C, UINT64_C(0xFFFFFFFF))) {
      Val = N.getOperand(0);
      return true;
    }
  }
  // FIXME: Should we just call computeKnownBits here?
  if (N.getOpcode() == ISD::AssertZext &&
      cast<VTSDNode>(N->getOperand(1))->getVT().bitsLE(MVT::i32)) {
    Val = N;
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
  return (VC1 >> VC2) == UINT64_C(0xFFFFFFFF);
}

// X0 has special meaning for vsetvl/vsetvli.
//  rd | rs1 |   AVL value | Effect on vl
//--------------------------------------------------------------
// !X0 |  X0 |       VLMAX | Set vl to VLMAX
//  X0 |  X0 | Value in vl | Keep current vl, just change vtype.
bool RISCVDAGToDAGISel::selectVLOp(SDValue N, SDValue &VL) {
  // If the VL value is a constant 0, manually select it to an ADDI with 0
  // immediate to prevent the default selection path from matching it to X0.
  auto *C = dyn_cast<ConstantSDNode>(N);
  if (C && C->isNullValue())
    VL = SDValue(selectImm(CurDAG, SDLoc(N), 0, Subtarget->getXLenVT()), 0);
  else
    VL = N;

  return true;
}

bool RISCVDAGToDAGISel::selectVSplat(SDValue N, SDValue &SplatVal) {
  if (N.getOpcode() != ISD::SPLAT_VECTOR &&
      N.getOpcode() != RISCVISD::SPLAT_VECTOR_I64 &&
      N.getOpcode() != RISCVISD::VMV_V_X_VL)
    return false;
  SplatVal = N.getOperand(0);
  return true;
}

bool RISCVDAGToDAGISel::selectVSplatSimm5(SDValue N, SDValue &SplatVal) {
  if ((N.getOpcode() != ISD::SPLAT_VECTOR &&
       N.getOpcode() != RISCVISD::SPLAT_VECTOR_I64 &&
       N.getOpcode() != RISCVISD::VMV_V_X_VL) ||
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
  MVT XLenVT = Subtarget->getXLenVT();
  assert(XLenVT == N.getOperand(0).getSimpleValueType() &&
         "Unexpected splat operand type");
  MVT EltVT = N.getSimpleValueType().getVectorElementType();
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
       N.getOpcode() != RISCVISD::SPLAT_VECTOR_I64 &&
       N.getOpcode() != RISCVISD::VMV_V_X_VL) ||
      !isa<ConstantSDNode>(N.getOperand(0)))
    return false;

  int64_t SplatImm = cast<ConstantSDNode>(N.getOperand(0))->getSExtValue();

  if (!isUInt<5>(SplatImm))
    return false;

  SplatVal =
      CurDAG->getTargetConstant(SplatImm, SDLoc(N), Subtarget->getXLenVT());

  return true;
}

bool RISCVDAGToDAGISel::selectRVVSimm5(SDValue N, unsigned Width,
                                       SDValue &Imm) {
  if (auto *C = dyn_cast<ConstantSDNode>(N)) {
    int64_t ImmVal = SignExtend64(C->getSExtValue(), Width);

    if (!isInt<5>(ImmVal))
      return false;

    Imm = CurDAG->getTargetConstant(ImmVal, SDLoc(N), Subtarget->getXLenVT());
    return true;
  }

  return false;
}

bool RISCVDAGToDAGISel::selectRVVUimm5(SDValue N, unsigned Width,
                                       SDValue &Imm) {
  if (auto *C = dyn_cast<ConstantSDNode>(N)) {
    int64_t ImmVal = C->getSExtValue();

    if (!isUInt<5>(ImmVal))
      return false;

    Imm = CurDAG->getTargetConstant(ImmVal, SDLoc(N), Subtarget->getXLenVT());
    return true;
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

    if (auto *Const = dyn_cast<ConstantSDNode>(ImmOperand)) {
      int64_t Offset1 = Const->getSExtValue();
      int64_t CombinedOffset = Offset1 + Offset2;
      if (!isInt<12>(CombinedOffset))
        continue;
      ImmOperand = CurDAG->getTargetConstant(CombinedOffset, SDLoc(ImmOperand),
                                             ImmOperand.getValueType());
    } else if (auto *GA = dyn_cast<GlobalAddressSDNode>(ImmOperand)) {
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
    } else if (auto *CP = dyn_cast<ConstantPoolSDNode>(ImmOperand)) {
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
