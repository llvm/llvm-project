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

static RISCVVLMUL getLMUL(EVT VT) {
  switch (VT.getSizeInBits().getKnownMinValue() / 8) {
  default:
    llvm_unreachable("Invalid LMUL.");
  case 1:
    return RISCVVLMUL::LMUL_F8;
  case 2:
    return RISCVVLMUL::LMUL_F4;
  case 4:
    return RISCVVLMUL::LMUL_F2;
  case 8:
    return RISCVVLMUL::LMUL_1;
  case 16:
    return RISCVVLMUL::LMUL_2;
  case 32:
    return RISCVVLMUL::LMUL_4;
  case 64:
    return RISCVVLMUL::LMUL_8;
  }
}

static unsigned getSubregIndexByEVT(EVT VT, unsigned Index) {
  RISCVVLMUL LMUL = getLMUL(VT);
  if (LMUL == RISCVVLMUL::LMUL_F8 || LMUL == RISCVVLMUL::LMUL_F4 ||
      LMUL == RISCVVLMUL::LMUL_F2 || LMUL == RISCVVLMUL::LMUL_1) {
    static_assert(RISCV::sub_vrm1_7 == RISCV::sub_vrm1_0 + 7,
                  "Unexpected subreg numbering");
    return RISCV::sub_vrm1_0 + Index;
  } else if (LMUL == RISCVVLMUL::LMUL_2) {
    static_assert(RISCV::sub_vrm2_3 == RISCV::sub_vrm2_0 + 3,
                  "Unexpected subreg numbering");
    return RISCV::sub_vrm2_0 + Index;
  } else if (LMUL == RISCVVLMUL::LMUL_4) {
    static_assert(RISCV::sub_vrm4_1 == RISCV::sub_vrm4_0 + 1,
                  "Unexpected subreg numbering");
    return RISCV::sub_vrm4_0 + Index;
  }
  llvm_unreachable("Invalid vector type.");
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

void RISCVDAGToDAGISel::selectVLSEG(SDNode *Node, unsigned IntNo,
                                    bool IsStrided) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumValues() - 1;
  EVT VT = Node->getValueType(0);
  unsigned ScalarSize = VT.getScalarSizeInBits();
  MVT XLenVT = Subtarget->getXLenVT();
  RISCVVLMUL LMUL = getLMUL(VT);
  SDValue SEW = CurDAG->getTargetConstant(ScalarSize, DL, XLenVT);
  SmallVector<SDValue, 5> Operands;
  Operands.push_back(Node->getOperand(2)); // Base pointer.
  if (IsStrided) {
    Operands.push_back(Node->getOperand(3)); // Stride.
    Operands.push_back(Node->getOperand(4)); // VL.
  } else {
    Operands.push_back(Node->getOperand(3)); // VL.
  }
  Operands.push_back(SEW);
  Operands.push_back(Node->getOperand(0)); // Chain.
  const RISCVZvlssegTable::RISCVZvlsseg *P = RISCVZvlssegTable::getPseudo(
      IntNo, ScalarSize, static_cast<unsigned>(LMUL),
      static_cast<unsigned>(RISCVVLMUL::LMUL_1));
  SDNode *Load =
      CurDAG->getMachineNode(P->Pseudo, DL, MVT::Untyped, MVT::Other, Operands);
  SDValue SuperReg = SDValue(Load, 0);
  for (unsigned I = 0; I < NF; ++I)
    ReplaceUses(SDValue(Node, I),
                CurDAG->getTargetExtractSubreg(getSubregIndexByEVT(VT, I), DL,
                                               VT, SuperReg));

  ReplaceUses(SDValue(Node, NF), SDValue(Load, 1));
  CurDAG->RemoveDeadNode(Node);
}

void RISCVDAGToDAGISel::selectVLSEGMask(SDNode *Node, unsigned IntNo,
                                        bool IsStrided) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumValues() - 1;
  EVT VT = Node->getValueType(0);
  unsigned ScalarSize = VT.getScalarSizeInBits();
  MVT XLenVT = Subtarget->getXLenVT();
  RISCVVLMUL LMUL = getLMUL(VT);
  SDValue SEW = CurDAG->getTargetConstant(ScalarSize, DL, XLenVT);
  SmallVector<SDValue, 8> Regs(Node->op_begin() + 2, Node->op_begin() + 2 + NF);
  SDValue MaskedOff = createTuple(*CurDAG, Regs, NF, LMUL);
  SmallVector<SDValue, 7> Operands;
  Operands.push_back(MaskedOff);
  Operands.push_back(Node->getOperand(NF + 2)); // Base pointer.
  if (IsStrided) {
    Operands.push_back(Node->getOperand(NF + 3)); // Stride.
    Operands.push_back(Node->getOperand(NF + 4)); // Mask.
    Operands.push_back(Node->getOperand(NF + 5)); // VL.
  } else {
    Operands.push_back(Node->getOperand(NF + 3)); // Mask.
    Operands.push_back(Node->getOperand(NF + 4)); // VL.
  }
  Operands.push_back(SEW);
  Operands.push_back(Node->getOperand(0)); /// Chain.
  const RISCVZvlssegTable::RISCVZvlsseg *P = RISCVZvlssegTable::getPseudo(
      IntNo, ScalarSize, static_cast<unsigned>(LMUL),
      static_cast<unsigned>(RISCVVLMUL::LMUL_1));
  SDNode *Load =
      CurDAG->getMachineNode(P->Pseudo, DL, MVT::Untyped, MVT::Other, Operands);
  SDValue SuperReg = SDValue(Load, 0);
  for (unsigned I = 0; I < NF; ++I)
    ReplaceUses(SDValue(Node, I),
                CurDAG->getTargetExtractSubreg(getSubregIndexByEVT(VT, I), DL,
                                               VT, SuperReg));

  ReplaceUses(SDValue(Node, NF), SDValue(Load, 1));
  CurDAG->RemoveDeadNode(Node);
}

void RISCVDAGToDAGISel::selectVLSEGFF(SDNode *Node) {
  SDLoc DL(Node);
  unsigned IntNo = cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue();
  unsigned NF = Node->getNumValues() - 2; // Do not count Chain and Glue.
  EVT VT = Node->getValueType(0);
  unsigned ScalarSize = VT.getScalarSizeInBits();
  MVT XLenVT = Subtarget->getXLenVT();
  RISCVVLMUL LMUL = getLMUL(VT);
  SDValue SEW = CurDAG->getTargetConstant(ScalarSize, DL, XLenVT);
  SmallVector<SDValue, 5> Operands;
  Operands.push_back(Node->getOperand(2)); // Base pointer.
  Operands.push_back(Node->getOperand(3)); // VL.
  Operands.push_back(SEW);
  Operands.push_back(Node->getOperand(0)); // Chain.
  const RISCVZvlssegTable::RISCVZvlsseg *P = RISCVZvlssegTable::getPseudo(
      IntNo, ScalarSize, static_cast<unsigned>(LMUL),
      static_cast<unsigned>(RISCVVLMUL::LMUL_1));
  SDNode *Load = CurDAG->getMachineNode(P->Pseudo, DL, MVT::Untyped, MVT::Other,
                                        MVT::Glue, Operands);
  SDValue SuperReg = SDValue(Load, 0);
  for (unsigned I = 0; I < NF; ++I)
    ReplaceUses(SDValue(Node, I),
                CurDAG->getTargetExtractSubreg(getSubregIndexByEVT(VT, I), DL,
                                               VT, SuperReg));

  ReplaceUses(SDValue(Node, NF), SDValue(Load, 1));     // Chain.
  ReplaceUses(SDValue(Node, NF + 1), SDValue(Load, 2)); // Glue.
  CurDAG->RemoveDeadNode(Node);
}

void RISCVDAGToDAGISel::selectVLSEGFFMask(SDNode *Node) {
  SDLoc DL(Node);
  unsigned IntNo = cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue();
  unsigned NF = Node->getNumValues() - 2; // Do not count Chain and Glue.
  EVT VT = Node->getValueType(0);
  unsigned ScalarSize = VT.getScalarSizeInBits();
  MVT XLenVT = Subtarget->getXLenVT();
  RISCVVLMUL LMUL = getLMUL(VT);
  SDValue SEW = CurDAG->getTargetConstant(ScalarSize, DL, XLenVT);
  SmallVector<SDValue, 8> Regs(Node->op_begin() + 2, Node->op_begin() + 2 + NF);
  SDValue MaskedOff = createTuple(*CurDAG, Regs, NF, LMUL);
  SmallVector<SDValue, 7> Operands;
  Operands.push_back(MaskedOff);
  Operands.push_back(Node->getOperand(NF + 2)); // Base pointer.
  Operands.push_back(Node->getOperand(NF + 3)); // Mask.
  Operands.push_back(Node->getOperand(NF + 4)); // VL.
  Operands.push_back(SEW);
  Operands.push_back(Node->getOperand(0)); /// Chain.
  const RISCVZvlssegTable::RISCVZvlsseg *P = RISCVZvlssegTable::getPseudo(
      IntNo, ScalarSize, static_cast<unsigned>(LMUL),
      static_cast<unsigned>(RISCVVLMUL::LMUL_1));
  SDNode *Load = CurDAG->getMachineNode(P->Pseudo, DL, MVT::Untyped, MVT::Other,
                                        MVT::Glue, Operands);
  SDValue SuperReg = SDValue(Load, 0);
  for (unsigned I = 0; I < NF; ++I)
    ReplaceUses(SDValue(Node, I),
                CurDAG->getTargetExtractSubreg(getSubregIndexByEVT(VT, I), DL,
                                               VT, SuperReg));

  ReplaceUses(SDValue(Node, NF), SDValue(Load, 1));     // Chain.
  ReplaceUses(SDValue(Node, NF + 1), SDValue(Load, 2)); // Glue.
  CurDAG->RemoveDeadNode(Node);
}

void RISCVDAGToDAGISel::selectVLXSEG(SDNode *Node, unsigned IntNo) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumValues() - 1;
  EVT VT = Node->getValueType(0);
  unsigned ScalarSize = VT.getScalarSizeInBits();
  MVT XLenVT = Subtarget->getXLenVT();
  RISCVVLMUL LMUL = getLMUL(VT);
  SDValue SEW = CurDAG->getTargetConstant(ScalarSize, DL, XLenVT);
  SDValue Operands[] = {
      Node->getOperand(2),     // Base pointer.
      Node->getOperand(3),     // Index.
      Node->getOperand(4),     // VL.
      SEW, Node->getOperand(0) // Chain.
  };

  EVT IndexVT = Node->getOperand(3)->getValueType(0);
  RISCVVLMUL IndexLMUL = getLMUL(IndexVT);
  unsigned IndexScalarSize = IndexVT.getScalarSizeInBits();
  const RISCVZvlssegTable::RISCVZvlsseg *P = RISCVZvlssegTable::getPseudo(
      IntNo, IndexScalarSize, static_cast<unsigned>(LMUL),
      static_cast<unsigned>(IndexLMUL));
  SDNode *Load =
      CurDAG->getMachineNode(P->Pseudo, DL, MVT::Untyped, MVT::Other, Operands);
  SDValue SuperReg = SDValue(Load, 0);
  for (unsigned I = 0; I < NF; ++I)
    ReplaceUses(SDValue(Node, I),
                CurDAG->getTargetExtractSubreg(getSubregIndexByEVT(VT, I), DL,
                                               VT, SuperReg));

  ReplaceUses(SDValue(Node, NF), SDValue(Load, 1));
  CurDAG->RemoveDeadNode(Node);
}

void RISCVDAGToDAGISel::selectVLXSEGMask(SDNode *Node, unsigned IntNo) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumValues() - 1;
  EVT VT = Node->getValueType(0);
  unsigned ScalarSize = VT.getScalarSizeInBits();
  MVT XLenVT = Subtarget->getXLenVT();
  RISCVVLMUL LMUL = getLMUL(VT);
  SDValue SEW = CurDAG->getTargetConstant(ScalarSize, DL, XLenVT);
  SmallVector<SDValue, 8> Regs(Node->op_begin() + 2, Node->op_begin() + 2 + NF);
  SDValue MaskedOff = createTuple(*CurDAG, Regs, NF, LMUL);
  SDValue Operands[] = {
      MaskedOff,
      Node->getOperand(NF + 2), // Base pointer.
      Node->getOperand(NF + 3), // Index.
      Node->getOperand(NF + 4), // Mask.
      Node->getOperand(NF + 5), // VL.
      SEW,
      Node->getOperand(0) // Chain.
  };

  EVT IndexVT = Node->getOperand(NF + 3)->getValueType(0);
  RISCVVLMUL IndexLMUL = getLMUL(IndexVT);
  unsigned IndexScalarSize = IndexVT.getScalarSizeInBits();
  const RISCVZvlssegTable::RISCVZvlsseg *P = RISCVZvlssegTable::getPseudo(
      IntNo, IndexScalarSize, static_cast<unsigned>(LMUL),
      static_cast<unsigned>(IndexLMUL));
  SDNode *Load =
      CurDAG->getMachineNode(P->Pseudo, DL, MVT::Untyped, MVT::Other, Operands);
  SDValue SuperReg = SDValue(Load, 0);
  for (unsigned I = 0; I < NF; ++I)
    ReplaceUses(SDValue(Node, I),
                CurDAG->getTargetExtractSubreg(getSubregIndexByEVT(VT, I), DL,
                                               VT, SuperReg));

  ReplaceUses(SDValue(Node, NF), SDValue(Load, 1));
  CurDAG->RemoveDeadNode(Node);
}

void RISCVDAGToDAGISel::selectVSSEG(SDNode *Node, unsigned IntNo,
                                    bool IsStrided) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumOperands() - 4;
  if (IsStrided)
    NF--;
  EVT VT = Node->getOperand(2)->getValueType(0);
  unsigned ScalarSize = VT.getScalarSizeInBits();
  MVT XLenVT = Subtarget->getXLenVT();
  RISCVVLMUL LMUL = getLMUL(VT);
  SDValue SEW = CurDAG->getTargetConstant(ScalarSize, DL, XLenVT);
  SmallVector<SDValue, 8> Regs(Node->op_begin() + 2, Node->op_begin() + 2 + NF);
  SDValue StoreVal = createTuple(*CurDAG, Regs, NF, LMUL);
  SmallVector<SDValue, 6> Operands;
  Operands.push_back(StoreVal);
  Operands.push_back(Node->getOperand(2 + NF)); // Base pointer.
  if (IsStrided) {
    Operands.push_back(Node->getOperand(3 + NF)); // Stride.
    Operands.push_back(Node->getOperand(4 + NF)); // VL.
  } else {
    Operands.push_back(Node->getOperand(3 + NF)); // VL.
  }
  Operands.push_back(SEW);
  Operands.push_back(Node->getOperand(0)); // Chain.
  const RISCVZvlssegTable::RISCVZvlsseg *P = RISCVZvlssegTable::getPseudo(
      IntNo, ScalarSize, static_cast<unsigned>(LMUL),
      static_cast<unsigned>(RISCVVLMUL::LMUL_1));
  SDNode *Store =
      CurDAG->getMachineNode(P->Pseudo, DL, Node->getValueType(0), Operands);
  ReplaceNode(Node, Store);
}

void RISCVDAGToDAGISel::selectVSSEGMask(SDNode *Node, unsigned IntNo,
                                        bool IsStrided) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumOperands() - 5;
  if (IsStrided)
    NF--;
  EVT VT = Node->getOperand(2)->getValueType(0);
  unsigned ScalarSize = VT.getScalarSizeInBits();
  MVT XLenVT = Subtarget->getXLenVT();
  RISCVVLMUL LMUL = getLMUL(VT);
  SDValue SEW = CurDAG->getTargetConstant(ScalarSize, DL, XLenVT);
  SmallVector<SDValue, 8> Regs(Node->op_begin() + 2, Node->op_begin() + 2 + NF);
  SDValue StoreVal = createTuple(*CurDAG, Regs, NF, LMUL);
  SmallVector<SDValue, 7> Operands;
  Operands.push_back(StoreVal);
  Operands.push_back(Node->getOperand(2 + NF)); // Base pointer.
  if (IsStrided) {
    Operands.push_back(Node->getOperand(3 + NF)); // Stride.
    Operands.push_back(Node->getOperand(4 + NF)); // Mask.
    Operands.push_back(Node->getOperand(5 + NF)); // VL.
  } else {
    Operands.push_back(Node->getOperand(3 + NF)); // Mask.
    Operands.push_back(Node->getOperand(4 + NF)); // VL.
  }
  Operands.push_back(SEW);
  Operands.push_back(Node->getOperand(0)); // Chain.
  const RISCVZvlssegTable::RISCVZvlsseg *P = RISCVZvlssegTable::getPseudo(
      IntNo, ScalarSize, static_cast<unsigned>(LMUL),
      static_cast<unsigned>(RISCVVLMUL::LMUL_1));
  SDNode *Store =
      CurDAG->getMachineNode(P->Pseudo, DL, Node->getValueType(0), Operands);
  ReplaceNode(Node, Store);
}

void RISCVDAGToDAGISel::selectVSXSEG(SDNode *Node, unsigned IntNo) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumOperands() - 5;
  EVT VT = Node->getOperand(2)->getValueType(0);
  unsigned ScalarSize = VT.getScalarSizeInBits();
  MVT XLenVT = Subtarget->getXLenVT();
  RISCVVLMUL LMUL = getLMUL(VT);
  SDValue SEW = CurDAG->getTargetConstant(ScalarSize, DL, XLenVT);
  SmallVector<SDValue, 8> Regs(Node->op_begin() + 2, Node->op_begin() + 2 + NF);
  SDValue StoreVal = createTuple(*CurDAG, Regs, NF, LMUL);
  SDValue Operands[] = {
      StoreVal,
      Node->getOperand(2 + NF), // Base pointer.
      Node->getOperand(3 + NF), // Index.
      Node->getOperand(4 + NF), // VL.
      SEW,
      Node->getOperand(0) // Chain.
  };

  EVT IndexVT = Node->getOperand(3 + NF)->getValueType(0);
  RISCVVLMUL IndexLMUL = getLMUL(IndexVT);
  unsigned IndexScalarSize = IndexVT.getScalarSizeInBits();
  const RISCVZvlssegTable::RISCVZvlsseg *P = RISCVZvlssegTable::getPseudo(
      IntNo, IndexScalarSize, static_cast<unsigned>(LMUL),
      static_cast<unsigned>(IndexLMUL));
  SDNode *Store =
      CurDAG->getMachineNode(P->Pseudo, DL, Node->getValueType(0), Operands);
  ReplaceNode(Node, Store);
}

void RISCVDAGToDAGISel::selectVSXSEGMask(SDNode *Node, unsigned IntNo) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumOperands() - 6;
  EVT VT = Node->getOperand(2)->getValueType(0);
  unsigned ScalarSize = VT.getScalarSizeInBits();
  MVT XLenVT = Subtarget->getXLenVT();
  RISCVVLMUL LMUL = getLMUL(VT);
  SDValue SEW = CurDAG->getTargetConstant(ScalarSize, DL, XLenVT);
  SmallVector<SDValue, 8> Regs(Node->op_begin() + 2, Node->op_begin() + 2 + NF);
  SDValue StoreVal = createTuple(*CurDAG, Regs, NF, LMUL);
  SDValue Operands[] = {
      StoreVal,
      Node->getOperand(2 + NF), // Base pointer.
      Node->getOperand(3 + NF), // Index.
      Node->getOperand(4 + NF), // Mask.
      Node->getOperand(5 + NF), // VL.
      SEW,
      Node->getOperand(0) // Chain.
  };

  EVT IndexVT = Node->getOperand(3 + NF)->getValueType(0);
  RISCVVLMUL IndexLMUL = getLMUL(IndexVT);
  unsigned IndexScalarSize = IndexVT.getScalarSizeInBits();
  const RISCVZvlssegTable::RISCVZvlsseg *P = RISCVZvlssegTable::getPseudo(
      IntNo, IndexScalarSize, static_cast<unsigned>(LMUL),
      static_cast<unsigned>(IndexLMUL));
  SDNode *Store =
      CurDAG->getMachineNode(P->Pseudo, DL, Node->getValueType(0), Operands);
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
        uint64_t AVL = C->getZExtValue();
        if (isUInt<5>(AVL)) {
          SDValue VLImm = CurDAG->getTargetConstant(AVL, DL, XLenVT);
          ReplaceNode(Node,
                      CurDAG->getMachineNode(RISCV::PseudoVSETIVLI, DL, XLenVT,
                                             MVT::Other, VLImm, VTypeIOp,
                                             /* Chain */ Node->getOperand(0)));
          return;
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
    case Intrinsic::riscv_vlseg2:
    case Intrinsic::riscv_vlseg3:
    case Intrinsic::riscv_vlseg4:
    case Intrinsic::riscv_vlseg5:
    case Intrinsic::riscv_vlseg6:
    case Intrinsic::riscv_vlseg7:
    case Intrinsic::riscv_vlseg8: {
      selectVLSEG(Node, IntNo, /*IsStrided=*/false);
      return;
    }
    case Intrinsic::riscv_vlseg2_mask:
    case Intrinsic::riscv_vlseg3_mask:
    case Intrinsic::riscv_vlseg4_mask:
    case Intrinsic::riscv_vlseg5_mask:
    case Intrinsic::riscv_vlseg6_mask:
    case Intrinsic::riscv_vlseg7_mask:
    case Intrinsic::riscv_vlseg8_mask: {
      selectVLSEGMask(Node, IntNo, /*IsStrided=*/false);
      return;
    }
    case Intrinsic::riscv_vlsseg2:
    case Intrinsic::riscv_vlsseg3:
    case Intrinsic::riscv_vlsseg4:
    case Intrinsic::riscv_vlsseg5:
    case Intrinsic::riscv_vlsseg6:
    case Intrinsic::riscv_vlsseg7:
    case Intrinsic::riscv_vlsseg8: {
      selectVLSEG(Node, IntNo, /*IsStrided=*/true);
      return;
    }
    case Intrinsic::riscv_vlsseg2_mask:
    case Intrinsic::riscv_vlsseg3_mask:
    case Intrinsic::riscv_vlsseg4_mask:
    case Intrinsic::riscv_vlsseg5_mask:
    case Intrinsic::riscv_vlsseg6_mask:
    case Intrinsic::riscv_vlsseg7_mask:
    case Intrinsic::riscv_vlsseg8_mask: {
      selectVLSEGMask(Node, IntNo, /*IsStrided=*/true);
      return;
    }
    case Intrinsic::riscv_vloxseg2:
    case Intrinsic::riscv_vloxseg3:
    case Intrinsic::riscv_vloxseg4:
    case Intrinsic::riscv_vloxseg5:
    case Intrinsic::riscv_vloxseg6:
    case Intrinsic::riscv_vloxseg7:
    case Intrinsic::riscv_vloxseg8:
    case Intrinsic::riscv_vluxseg2:
    case Intrinsic::riscv_vluxseg3:
    case Intrinsic::riscv_vluxseg4:
    case Intrinsic::riscv_vluxseg5:
    case Intrinsic::riscv_vluxseg6:
    case Intrinsic::riscv_vluxseg7:
    case Intrinsic::riscv_vluxseg8: {
      selectVLXSEG(Node, IntNo);
      return;
    }
    case Intrinsic::riscv_vloxseg2_mask:
    case Intrinsic::riscv_vloxseg3_mask:
    case Intrinsic::riscv_vloxseg4_mask:
    case Intrinsic::riscv_vloxseg5_mask:
    case Intrinsic::riscv_vloxseg6_mask:
    case Intrinsic::riscv_vloxseg7_mask:
    case Intrinsic::riscv_vloxseg8_mask:
    case Intrinsic::riscv_vluxseg2_mask:
    case Intrinsic::riscv_vluxseg3_mask:
    case Intrinsic::riscv_vluxseg4_mask:
    case Intrinsic::riscv_vluxseg5_mask:
    case Intrinsic::riscv_vluxseg6_mask:
    case Intrinsic::riscv_vluxseg7_mask:
    case Intrinsic::riscv_vluxseg8_mask: {
      selectVLXSEGMask(Node, IntNo);
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
      selectVSSEG(Node, IntNo, /*IsStrided=*/false);
      return;
    }
    case Intrinsic::riscv_vsseg2_mask:
    case Intrinsic::riscv_vsseg3_mask:
    case Intrinsic::riscv_vsseg4_mask:
    case Intrinsic::riscv_vsseg5_mask:
    case Intrinsic::riscv_vsseg6_mask:
    case Intrinsic::riscv_vsseg7_mask:
    case Intrinsic::riscv_vsseg8_mask: {
      selectVSSEGMask(Node, IntNo, /*IsStrided=*/false);
      return;
    }
    case Intrinsic::riscv_vssseg2:
    case Intrinsic::riscv_vssseg3:
    case Intrinsic::riscv_vssseg4:
    case Intrinsic::riscv_vssseg5:
    case Intrinsic::riscv_vssseg6:
    case Intrinsic::riscv_vssseg7:
    case Intrinsic::riscv_vssseg8: {
      selectVSSEG(Node, IntNo, /*IsStrided=*/true);
      return;
    }
    case Intrinsic::riscv_vssseg2_mask:
    case Intrinsic::riscv_vssseg3_mask:
    case Intrinsic::riscv_vssseg4_mask:
    case Intrinsic::riscv_vssseg5_mask:
    case Intrinsic::riscv_vssseg6_mask:
    case Intrinsic::riscv_vssseg7_mask:
    case Intrinsic::riscv_vssseg8_mask: {
      selectVSSEGMask(Node, IntNo, /*IsStrided=*/true);
      return;
    }
    case Intrinsic::riscv_vsoxseg2:
    case Intrinsic::riscv_vsoxseg3:
    case Intrinsic::riscv_vsoxseg4:
    case Intrinsic::riscv_vsoxseg5:
    case Intrinsic::riscv_vsoxseg6:
    case Intrinsic::riscv_vsoxseg7:
    case Intrinsic::riscv_vsoxseg8:
    case Intrinsic::riscv_vsuxseg2:
    case Intrinsic::riscv_vsuxseg3:
    case Intrinsic::riscv_vsuxseg4:
    case Intrinsic::riscv_vsuxseg5:
    case Intrinsic::riscv_vsuxseg6:
    case Intrinsic::riscv_vsuxseg7:
    case Intrinsic::riscv_vsuxseg8: {
      selectVSXSEG(Node, IntNo);
      return;
    }
    case Intrinsic::riscv_vsoxseg2_mask:
    case Intrinsic::riscv_vsoxseg3_mask:
    case Intrinsic::riscv_vsoxseg4_mask:
    case Intrinsic::riscv_vsoxseg5_mask:
    case Intrinsic::riscv_vsoxseg6_mask:
    case Intrinsic::riscv_vsoxseg7_mask:
    case Intrinsic::riscv_vsoxseg8_mask:
    case Intrinsic::riscv_vsuxseg2_mask:
    case Intrinsic::riscv_vsuxseg3_mask:
    case Intrinsic::riscv_vsuxseg4_mask:
    case Intrinsic::riscv_vsuxseg5_mask:
    case Intrinsic::riscv_vsuxseg6_mask:
    case Intrinsic::riscv_vsuxseg7_mask:
    case Intrinsic::riscv_vsuxseg8_mask: {
      selectVSXSEGMask(Node, IntNo);
      return;
    }
    }
    break;
  }
  case RISCVISD::VLSEGFF: {
    selectVLSEGFF(Node);
    return;
  }
  case RISCVISD::VLSEGFF_MASK: {
    selectVLSEGFFMask(Node);
    return;
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
