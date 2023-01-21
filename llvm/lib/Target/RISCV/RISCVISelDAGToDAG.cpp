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
#include "RISCVMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "riscv-isel"
#define PASS_NAME "RISCV DAG->DAG Pattern Instruction Selection"

namespace llvm::RISCV {
#define GET_RISCVVSSEGTable_IMPL
#define GET_RISCVVLSEGTable_IMPL
#define GET_RISCVVLXSEGTable_IMPL
#define GET_RISCVVSXSEGTable_IMPL
#define GET_RISCVVLETable_IMPL
#define GET_RISCVVSETable_IMPL
#define GET_RISCVVLXTable_IMPL
#define GET_RISCVVSXTable_IMPL
#define GET_RISCVMaskedPseudosTable_IMPL
#include "RISCVGenSearchableTables.inc"
} // namespace llvm::RISCV

static unsigned getLastNonGlueOrChainOpIdx(const SDNode *Node) {
  assert(Node->getNumOperands() > 0 && "Node with no operands");
  unsigned LastOpIdx = Node->getNumOperands() - 1;
  if (Node->getOperand(LastOpIdx).getValueType() == MVT::Glue)
    --LastOpIdx;
  if (Node->getOperand(LastOpIdx).getValueType() == MVT::Other)
    --LastOpIdx;
  return LastOpIdx;
}

static unsigned getVecPolicyOpIdx(const SDNode *Node, const MCInstrDesc &MCID) {
  assert(RISCVII::hasVecPolicyOp(MCID.TSFlags));
  (void)MCID;
  return getLastNonGlueOrChainOpIdx(Node);
}

void RISCVDAGToDAGISel::PreprocessISelDAG() {
  SelectionDAG::allnodes_iterator Position = CurDAG->allnodes_end();

  bool MadeChange = false;
  while (Position != CurDAG->allnodes_begin()) {
    SDNode *N = &*--Position;
    if (N->use_empty())
      continue;

    SDValue Result;
    switch (N->getOpcode()) {
    case ISD::SPLAT_VECTOR: {
      // Convert integer SPLAT_VECTOR to VMV_V_X_VL and floating-point
      // SPLAT_VECTOR to VFMV_V_F_VL to reduce isel burden.
      MVT VT = N->getSimpleValueType(0);
      unsigned Opc =
          VT.isInteger() ? RISCVISD::VMV_V_X_VL : RISCVISD::VFMV_V_F_VL;
      SDLoc DL(N);
      SDValue VL = CurDAG->getRegister(RISCV::X0, Subtarget->getXLenVT());
      Result = CurDAG->getNode(Opc, DL, VT, CurDAG->getUNDEF(VT),
                               N->getOperand(0), VL);
      break;
    }
    case RISCVISD::SPLAT_VECTOR_SPLIT_I64_VL: {
      // Lower SPLAT_VECTOR_SPLIT_I64 to two scalar stores and a stride 0 vector
      // load. Done after lowering and combining so that we have a chance to
      // optimize this to VMV_V_X_VL when the upper bits aren't needed.
      assert(N->getNumOperands() == 4 && "Unexpected number of operands");
      MVT VT = N->getSimpleValueType(0);
      SDValue Passthru = N->getOperand(0);
      SDValue Lo = N->getOperand(1);
      SDValue Hi = N->getOperand(2);
      SDValue VL = N->getOperand(3);
      assert(VT.getVectorElementType() == MVT::i64 && VT.isScalableVector() &&
             Lo.getValueType() == MVT::i32 && Hi.getValueType() == MVT::i32 &&
             "Unexpected VTs!");
      MachineFunction &MF = CurDAG->getMachineFunction();
      RISCVMachineFunctionInfo *FuncInfo =
          MF.getInfo<RISCVMachineFunctionInfo>();
      SDLoc DL(N);

      // We use the same frame index we use for moving two i32s into 64-bit FPR.
      // This is an analogous operation.
      int FI = FuncInfo->getMoveF64FrameIndex(MF);
      MachinePointerInfo MPI = MachinePointerInfo::getFixedStack(MF, FI);
      const TargetLowering &TLI = CurDAG->getTargetLoweringInfo();
      SDValue StackSlot =
          CurDAG->getFrameIndex(FI, TLI.getPointerTy(CurDAG->getDataLayout()));

      SDValue Chain = CurDAG->getEntryNode();
      Lo = CurDAG->getStore(Chain, DL, Lo, StackSlot, MPI, Align(8));

      SDValue OffsetSlot =
          CurDAG->getMemBasePlusOffset(StackSlot, TypeSize::Fixed(4), DL);
      Hi = CurDAG->getStore(Chain, DL, Hi, OffsetSlot, MPI.getWithOffset(4),
                            Align(8));

      Chain = CurDAG->getNode(ISD::TokenFactor, DL, MVT::Other, Lo, Hi);

      SDVTList VTs = CurDAG->getVTList({VT, MVT::Other});
      SDValue IntID =
          CurDAG->getTargetConstant(Intrinsic::riscv_vlse, DL, MVT::i64);
      SDValue Ops[] = {Chain,
                       IntID,
                       Passthru,
                       StackSlot,
                       CurDAG->getRegister(RISCV::X0, MVT::i64),
                       VL};

      Result = CurDAG->getMemIntrinsicNode(ISD::INTRINSIC_W_CHAIN, DL, VTs, Ops,
                                           MVT::i64, MPI, Align(8),
                                           MachineMemOperand::MOLoad);
      break;
    }
    }

    if (Result) {
      LLVM_DEBUG(dbgs() << "RISCV DAG preprocessing replacing:\nOld:    ");
      LLVM_DEBUG(N->dump(CurDAG));
      LLVM_DEBUG(dbgs() << "\nNew: ");
      LLVM_DEBUG(Result->dump(CurDAG));
      LLVM_DEBUG(dbgs() << "\n");

      CurDAG->ReplaceAllUsesOfValueWith(SDValue(N, 0), Result);
      MadeChange = true;
    }
  }

  if (MadeChange)
    CurDAG->RemoveDeadNodes();
}

void RISCVDAGToDAGISel::PostprocessISelDAG() {
  HandleSDNode Dummy(CurDAG->getRoot());
  SelectionDAG::allnodes_iterator Position = CurDAG->allnodes_end();

  bool MadeChange = false;
  while (Position != CurDAG->allnodes_begin()) {
    SDNode *N = &*--Position;
    // Skip dead nodes and any non-machine opcodes.
    if (N->use_empty() || !N->isMachineOpcode())
      continue;

    MadeChange |= doPeepholeSExtW(N);
    MadeChange |= doPeepholeMaskedRVV(N);
  }

  CurDAG->setRoot(Dummy.getValue());

  MadeChange |= doPeepholeMergeVVMFold();

  if (MadeChange)
    CurDAG->RemoveDeadNodes();
}

static SDNode *selectImmSeq(SelectionDAG *CurDAG, const SDLoc &DL, const MVT VT,
                            RISCVMatInt::InstSeq &Seq) {
  SDNode *Result = nullptr;
  SDValue SrcReg = CurDAG->getRegister(RISCV::X0, VT);
  for (RISCVMatInt::Inst &Inst : Seq) {
    SDValue SDImm = CurDAG->getTargetConstant(Inst.getImm(), DL, VT);
    switch (Inst.getOpndKind()) {
    case RISCVMatInt::Imm:
      Result = CurDAG->getMachineNode(Inst.getOpcode(), DL, VT, SDImm);
      break;
    case RISCVMatInt::RegX0:
      Result = CurDAG->getMachineNode(Inst.getOpcode(), DL, VT, SrcReg,
                                      CurDAG->getRegister(RISCV::X0, VT));
      break;
    case RISCVMatInt::RegReg:
      Result = CurDAG->getMachineNode(Inst.getOpcode(), DL, VT, SrcReg, SrcReg);
      break;
    case RISCVMatInt::RegImm:
      Result = CurDAG->getMachineNode(Inst.getOpcode(), DL, VT, SrcReg, SDImm);
      break;
    }

    // Only the first instruction has X0 as its source.
    SrcReg = SDValue(Result, 0);
  }

  return Result;
}

static SDNode *selectImm(SelectionDAG *CurDAG, const SDLoc &DL, const MVT VT,
                         int64_t Imm, const RISCVSubtarget &Subtarget) {
  RISCVMatInt::InstSeq Seq =
      RISCVMatInt::generateInstSeq(Imm, Subtarget.getFeatureBits());

  return selectImmSeq(CurDAG, DL, VT, Seq);
}

static SDValue createTuple(SelectionDAG &CurDAG, ArrayRef<SDValue> Regs,
                           unsigned NF, RISCVII::VLMUL LMUL) {
  static const unsigned M1TupleRegClassIDs[] = {
      RISCV::VRN2M1RegClassID, RISCV::VRN3M1RegClassID, RISCV::VRN4M1RegClassID,
      RISCV::VRN5M1RegClassID, RISCV::VRN6M1RegClassID, RISCV::VRN7M1RegClassID,
      RISCV::VRN8M1RegClassID};
  static const unsigned M2TupleRegClassIDs[] = {RISCV::VRN2M2RegClassID,
                                                RISCV::VRN3M2RegClassID,
                                                RISCV::VRN4M2RegClassID};

  assert(Regs.size() >= 2 && Regs.size() <= 8);

  unsigned RegClassID;
  unsigned SubReg0;
  switch (LMUL) {
  default:
    llvm_unreachable("Invalid LMUL.");
  case RISCVII::VLMUL::LMUL_F8:
  case RISCVII::VLMUL::LMUL_F4:
  case RISCVII::VLMUL::LMUL_F2:
  case RISCVII::VLMUL::LMUL_1:
    static_assert(RISCV::sub_vrm1_7 == RISCV::sub_vrm1_0 + 7,
                  "Unexpected subreg numbering");
    SubReg0 = RISCV::sub_vrm1_0;
    RegClassID = M1TupleRegClassIDs[NF - 2];
    break;
  case RISCVII::VLMUL::LMUL_2:
    static_assert(RISCV::sub_vrm2_3 == RISCV::sub_vrm2_0 + 3,
                  "Unexpected subreg numbering");
    SubReg0 = RISCV::sub_vrm2_0;
    RegClassID = M2TupleRegClassIDs[NF - 2];
    break;
  case RISCVII::VLMUL::LMUL_4:
    static_assert(RISCV::sub_vrm4_1 == RISCV::sub_vrm4_0 + 1,
                  "Unexpected subreg numbering");
    SubReg0 = RISCV::sub_vrm4_0;
    RegClassID = RISCV::VRN2M4RegClassID;
    break;
  }

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

void RISCVDAGToDAGISel::addVectorLoadStoreOperands(
    SDNode *Node, unsigned Log2SEW, const SDLoc &DL, unsigned CurOp,
    bool IsMasked, bool IsStridedOrIndexed, SmallVectorImpl<SDValue> &Operands,
    bool IsLoad, MVT *IndexVT) {
  SDValue Chain = Node->getOperand(0);
  SDValue Glue;

  Operands.push_back(Node->getOperand(CurOp++)); // Base pointer.

  if (IsStridedOrIndexed) {
    Operands.push_back(Node->getOperand(CurOp++)); // Index.
    if (IndexVT)
      *IndexVT = Operands.back()->getSimpleValueType(0);
  }

  if (IsMasked) {
    // Mask needs to be copied to V0.
    SDValue Mask = Node->getOperand(CurOp++);
    Chain = CurDAG->getCopyToReg(Chain, DL, RISCV::V0, Mask, SDValue());
    Glue = Chain.getValue(1);
    Operands.push_back(CurDAG->getRegister(RISCV::V0, Mask.getValueType()));
  }
  SDValue VL;
  selectVLOp(Node->getOperand(CurOp++), VL);
  Operands.push_back(VL);

  MVT XLenVT = Subtarget->getXLenVT();
  SDValue SEWOp = CurDAG->getTargetConstant(Log2SEW, DL, XLenVT);
  Operands.push_back(SEWOp);

  // Masked load has the tail policy argument.
  if (IsMasked && IsLoad) {
    // Policy must be a constant.
    uint64_t Policy = Node->getConstantOperandVal(CurOp++);
    SDValue PolicyOp = CurDAG->getTargetConstant(Policy, DL, XLenVT);
    Operands.push_back(PolicyOp);
  }

  Operands.push_back(Chain); // Chain.
  if (Glue)
    Operands.push_back(Glue);
}

static bool isAllUndef(ArrayRef<SDValue> Values) {
  return llvm::all_of(Values, [](SDValue V) { return V->isUndef(); });
}

void RISCVDAGToDAGISel::selectVLSEG(SDNode *Node, bool IsMasked,
                                    bool IsStrided) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumValues() - 1;
  MVT VT = Node->getSimpleValueType(0);
  unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());
  RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);

  unsigned CurOp = 2;
  SmallVector<SDValue, 8> Operands;

  SmallVector<SDValue, 8> Regs(Node->op_begin() + CurOp,
                               Node->op_begin() + CurOp + NF);
  bool IsTU = IsMasked || !isAllUndef(Regs);
  if (IsTU) {
    SDValue Merge = createTuple(*CurDAG, Regs, NF, LMUL);
    Operands.push_back(Merge);
  }
  CurOp += NF;

  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked, IsStrided,
                             Operands, /*IsLoad=*/true);

  const RISCV::VLSEGPseudo *P =
      RISCV::getVLSEGPseudo(NF, IsMasked, IsTU, IsStrided, /*FF*/ false, Log2SEW,
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
  unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());
  RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);

  unsigned CurOp = 2;
  SmallVector<SDValue, 7> Operands;

  SmallVector<SDValue, 8> Regs(Node->op_begin() + CurOp,
                               Node->op_begin() + CurOp + NF);
  bool IsTU = IsMasked || !isAllUndef(Regs);
  if (IsTU) {
    SDValue MaskedOff = createTuple(*CurDAG, Regs, NF, LMUL);
    Operands.push_back(MaskedOff);
  }
  CurOp += NF;

  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                             /*IsStridedOrIndexed*/ false, Operands,
                             /*IsLoad=*/true);

  const RISCV::VLSEGPseudo *P =
      RISCV::getVLSEGPseudo(NF, IsMasked, IsTU, /*Strided*/ false, /*FF*/ true,
                            Log2SEW, static_cast<unsigned>(LMUL));
  MachineSDNode *Load = CurDAG->getMachineNode(P->Pseudo, DL, MVT::Untyped,
                                               XLenVT, MVT::Other, Operands);

  if (auto *MemOp = dyn_cast<MemSDNode>(Node))
    CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

  SDValue SuperReg = SDValue(Load, 0);
  for (unsigned I = 0; I < NF; ++I) {
    unsigned SubRegIdx = RISCVTargetLowering::getSubregIndexByMVT(VT, I);
    ReplaceUses(SDValue(Node, I),
                CurDAG->getTargetExtractSubreg(SubRegIdx, DL, VT, SuperReg));
  }

  ReplaceUses(SDValue(Node, NF), SDValue(Load, 1));     // VL
  ReplaceUses(SDValue(Node, NF + 1), SDValue(Load, 2)); // Chain
  CurDAG->RemoveDeadNode(Node);
}

void RISCVDAGToDAGISel::selectVLXSEG(SDNode *Node, bool IsMasked,
                                     bool IsOrdered) {
  SDLoc DL(Node);
  unsigned NF = Node->getNumValues() - 1;
  MVT VT = Node->getSimpleValueType(0);
  unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());
  RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);

  unsigned CurOp = 2;
  SmallVector<SDValue, 8> Operands;

  SmallVector<SDValue, 8> Regs(Node->op_begin() + CurOp,
                               Node->op_begin() + CurOp + NF);
  bool IsTU = IsMasked || !isAllUndef(Regs);
  if (IsTU) {
    SDValue MaskedOff = createTuple(*CurDAG, Regs, NF, LMUL);
    Operands.push_back(MaskedOff);
  }
  CurOp += NF;

  MVT IndexVT;
  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                             /*IsStridedOrIndexed*/ true, Operands,
                             /*IsLoad=*/true, &IndexVT);

  assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
         "Element count mismatch");

  RISCVII::VLMUL IndexLMUL = RISCVTargetLowering::getLMUL(IndexVT);
  unsigned IndexLog2EEW = Log2_32(IndexVT.getScalarSizeInBits());
  if (IndexLog2EEW == 6 && !Subtarget->is64Bit()) {
    report_fatal_error("The V extension does not support EEW=64 for index "
                       "values when XLEN=32");
  }
  const RISCV::VLXSEGPseudo *P = RISCV::getVLXSEGPseudo(
      NF, IsMasked, IsTU, IsOrdered, IndexLog2EEW, static_cast<unsigned>(LMUL),
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
  unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());
  RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
  SmallVector<SDValue, 8> Regs(Node->op_begin() + 2, Node->op_begin() + 2 + NF);
  SDValue StoreVal = createTuple(*CurDAG, Regs, NF, LMUL);

  SmallVector<SDValue, 8> Operands;
  Operands.push_back(StoreVal);
  unsigned CurOp = 2 + NF;

  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked, IsStrided,
                             Operands);

  const RISCV::VSSEGPseudo *P = RISCV::getVSSEGPseudo(
      NF, IsMasked, IsStrided, Log2SEW, static_cast<unsigned>(LMUL));
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
  unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());
  RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
  SmallVector<SDValue, 8> Regs(Node->op_begin() + 2, Node->op_begin() + 2 + NF);
  SDValue StoreVal = createTuple(*CurDAG, Regs, NF, LMUL);

  SmallVector<SDValue, 8> Operands;
  Operands.push_back(StoreVal);
  unsigned CurOp = 2 + NF;

  MVT IndexVT;
  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                             /*IsStridedOrIndexed*/ true, Operands,
                             /*IsLoad=*/false, &IndexVT);

  assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
         "Element count mismatch");

  RISCVII::VLMUL IndexLMUL = RISCVTargetLowering::getLMUL(IndexVT);
  unsigned IndexLog2EEW = Log2_32(IndexVT.getScalarSizeInBits());
  if (IndexLog2EEW == 6 && !Subtarget->is64Bit()) {
    report_fatal_error("The V extension does not support EEW=64 for index "
                       "values when XLEN=32");
  }
  const RISCV::VSXSEGPseudo *P = RISCV::getVSXSEGPseudo(
      NF, IsMasked, IsOrdered, IndexLog2EEW, static_cast<unsigned>(LMUL),
      static_cast<unsigned>(IndexLMUL));
  MachineSDNode *Store =
      CurDAG->getMachineNode(P->Pseudo, DL, Node->getValueType(0), Operands);

  if (auto *MemOp = dyn_cast<MemSDNode>(Node))
    CurDAG->setNodeMemRefs(Store, {MemOp->getMemOperand()});

  ReplaceNode(Node, Store);
}

void RISCVDAGToDAGISel::selectVSETVLI(SDNode *Node) {
  if (!Subtarget->hasVInstructions())
    return;

  assert((Node->getOpcode() == ISD::INTRINSIC_W_CHAIN ||
          Node->getOpcode() == ISD::INTRINSIC_WO_CHAIN) &&
         "Unexpected opcode");

  SDLoc DL(Node);
  MVT XLenVT = Subtarget->getXLenVT();

  bool HasChain = Node->getOpcode() == ISD::INTRINSIC_W_CHAIN;
  unsigned IntNoOffset = HasChain ? 1 : 0;
  unsigned IntNo = Node->getConstantOperandVal(IntNoOffset);

  assert((IntNo == Intrinsic::riscv_vsetvli ||
          IntNo == Intrinsic::riscv_vsetvlimax ||
          IntNo == Intrinsic::riscv_vsetvli_opt ||
          IntNo == Intrinsic::riscv_vsetvlimax_opt) &&
         "Unexpected vsetvli intrinsic");

  bool VLMax = IntNo == Intrinsic::riscv_vsetvlimax ||
               IntNo == Intrinsic::riscv_vsetvlimax_opt;
  unsigned Offset = IntNoOffset + (VLMax ? 1 : 2);

  assert(Node->getNumOperands() == Offset + 2 &&
         "Unexpected number of operands");

  unsigned SEW =
      RISCVVType::decodeVSEW(Node->getConstantOperandVal(Offset) & 0x7);
  RISCVII::VLMUL VLMul = static_cast<RISCVII::VLMUL>(
      Node->getConstantOperandVal(Offset + 1) & 0x7);

  unsigned VTypeI = RISCVVType::encodeVTYPE(VLMul, SEW, /*TailAgnostic*/ true,
                                            /*MaskAgnostic*/ false);
  SDValue VTypeIOp = CurDAG->getTargetConstant(VTypeI, DL, XLenVT);

  SmallVector<EVT, 2> VTs = {XLenVT};
  if (HasChain)
    VTs.push_back(MVT::Other);

  SDValue VLOperand;
  unsigned Opcode = RISCV::PseudoVSETVLI;
  if (VLMax) {
    VLOperand = CurDAG->getRegister(RISCV::X0, XLenVT);
    Opcode = RISCV::PseudoVSETVLIX0;
  } else {
    VLOperand = Node->getOperand(IntNoOffset + 1);

    if (auto *C = dyn_cast<ConstantSDNode>(VLOperand)) {
      uint64_t AVL = C->getZExtValue();
      if (isUInt<5>(AVL)) {
        SDValue VLImm = CurDAG->getTargetConstant(AVL, DL, XLenVT);
        SmallVector<SDValue, 3> Ops = {VLImm, VTypeIOp};
        if (HasChain)
          Ops.push_back(Node->getOperand(0));
        ReplaceNode(
            Node, CurDAG->getMachineNode(RISCV::PseudoVSETIVLI, DL, VTs, Ops));
        return;
      }
    }
  }

  SmallVector<SDValue, 3> Ops = {VLOperand, VTypeIOp};
  if (HasChain)
    Ops.push_back(Node->getOperand(0));

  ReplaceNode(Node, CurDAG->getMachineNode(Opcode, DL, VTs, Ops));
}

bool RISCVDAGToDAGISel::tryShrinkShlLogicImm(SDNode *Node) {
  MVT VT = Node->getSimpleValueType(0);
  unsigned Opcode = Node->getOpcode();
  assert((Opcode == ISD::AND || Opcode == ISD::OR || Opcode == ISD::XOR) &&
         "Unexpected opcode");
  SDLoc DL(Node);

  // For operations of the form (x << C1) op C2, check if we can use
  // ANDI/ORI/XORI by transforming it into (x op (C2>>C1)) << C1.
  SDValue N0 = Node->getOperand(0);
  SDValue N1 = Node->getOperand(1);

  ConstantSDNode *Cst = dyn_cast<ConstantSDNode>(N1);
  if (!Cst)
    return false;

  int64_t Val = Cst->getSExtValue();

  // Check if immediate can already use ANDI/ORI/XORI.
  if (isInt<12>(Val))
    return false;

  SDValue Shift = N0;

  // If Val is simm32 and we have a sext_inreg from i32, then the binop
  // produces at least 33 sign bits. We can peek through the sext_inreg and use
  // a SLLIW at the end.
  bool SignExt = false;
  if (isInt<32>(Val) && N0.getOpcode() == ISD::SIGN_EXTEND_INREG &&
      N0.hasOneUse() && cast<VTSDNode>(N0.getOperand(1))->getVT() == MVT::i32) {
    SignExt = true;
    Shift = N0.getOperand(0);
  }

  if (Shift.getOpcode() != ISD::SHL || !Shift.hasOneUse())
    return false;

  ConstantSDNode *ShlCst = dyn_cast<ConstantSDNode>(Shift.getOperand(1));
  if (!ShlCst)
    return false;

  uint64_t ShAmt = ShlCst->getZExtValue();

  // Make sure that we don't change the operation by removing bits.
  // This only matters for OR and XOR, AND is unaffected.
  uint64_t RemovedBitsMask = maskTrailingOnes<uint64_t>(ShAmt);
  if (Opcode != ISD::AND && (Val & RemovedBitsMask) != 0)
    return false;

  int64_t ShiftedVal = Val >> ShAmt;
  if (!isInt<12>(ShiftedVal))
    return false;

  // If we peeked through a sext_inreg, make sure the shift is valid for SLLIW.
  if (SignExt && ShAmt >= 32)
    return false;

  // Ok, we can reorder to get a smaller immediate.
  unsigned BinOpc;
  switch (Opcode) {
  default: llvm_unreachable("Unexpected opcode");
  case ISD::AND: BinOpc = RISCV::ANDI; break;
  case ISD::OR:  BinOpc = RISCV::ORI;  break;
  case ISD::XOR: BinOpc = RISCV::XORI; break;
  }

  unsigned ShOpc = SignExt ? RISCV::SLLIW : RISCV::SLLI;

  SDNode *BinOp =
      CurDAG->getMachineNode(BinOpc, DL, VT, Shift.getOperand(0),
                             CurDAG->getTargetConstant(ShiftedVal, DL, VT));
  SDNode *SLLI =
      CurDAG->getMachineNode(ShOpc, DL, VT, SDValue(BinOp, 0),
                             CurDAG->getTargetConstant(ShAmt, DL, VT));
  ReplaceNode(Node, SLLI);
  return true;
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
  case ISD::Constant: {
    auto *ConstNode = cast<ConstantSDNode>(Node);
    if (VT == XLenVT && ConstNode->isZero()) {
      SDValue New =
          CurDAG->getCopyFromReg(CurDAG->getEntryNode(), DL, RISCV::X0, XLenVT);
      ReplaceNode(Node, New.getNode());
      return;
    }
    int64_t Imm = ConstNode->getSExtValue();
    // If the upper XLen-16 bits are not used, try to convert this to a simm12
    // by sign extending bit 15.
    if (isUInt<16>(Imm) && isInt<12>(SignExtend64<16>(Imm)) &&
        hasAllHUsers(Node))
      Imm = SignExtend64<16>(Imm);
    // If the upper 32-bits are not used try to convert this into a simm32 by
    // sign extending bit 32.
    if (!isInt<32>(Imm) && isUInt<32>(Imm) && hasAllWUsers(Node))
      Imm = SignExtend64<32>(Imm);

    ReplaceNode(Node, selectImm(CurDAG, DL, VT, Imm, *Subtarget));
    return;
  }
  case ISD::SHL: {
    auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
    if (!N1C)
      break;
    SDValue N0 = Node->getOperand(0);
    if (N0.getOpcode() != ISD::AND || !N0.hasOneUse() ||
        !isa<ConstantSDNode>(N0.getOperand(1)))
      break;
    unsigned ShAmt = N1C->getZExtValue();
    uint64_t Mask = N0.getConstantOperandVal(1);

    // Optimize (shl (and X, C2), C) -> (slli (srliw X, C3), C3+C) where C2 has
    // 32 leading zeros and C3 trailing zeros.
    if (ShAmt <= 32 && isShiftedMask_64(Mask)) {
      unsigned XLen = Subtarget->getXLen();
      unsigned LeadingZeros = XLen - llvm::bit_width(Mask);
      unsigned TrailingZeros = countTrailingZeros(Mask);
      if (TrailingZeros > 0 && LeadingZeros == 32) {
        SDNode *SRLIW = CurDAG->getMachineNode(
            RISCV::SRLIW, DL, VT, N0->getOperand(0),
            CurDAG->getTargetConstant(TrailingZeros, DL, VT));
        SDNode *SLLI = CurDAG->getMachineNode(
            RISCV::SLLI, DL, VT, SDValue(SRLIW, 0),
            CurDAG->getTargetConstant(TrailingZeros + ShAmt, DL, VT));
        ReplaceNode(Node, SLLI);
        return;
      }
    }
    break;
  }
  case ISD::SRL: {
    auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
    if (!N1C)
      break;
    SDValue N0 = Node->getOperand(0);
    if (N0.getOpcode() != ISD::AND || !isa<ConstantSDNode>(N0.getOperand(1)))
      break;
    unsigned ShAmt = N1C->getZExtValue();
    uint64_t Mask = N0.getConstantOperandVal(1);

    // Optimize (srl (and X, C2), C) -> (slli (srliw X, C3), C3-C) where C2 has
    // 32 leading zeros and C3 trailing zeros.
    if (isShiftedMask_64(Mask) && N0.hasOneUse()) {
      unsigned XLen = Subtarget->getXLen();
      unsigned LeadingZeros = XLen - llvm::bit_width(Mask);
      unsigned TrailingZeros = countTrailingZeros(Mask);
      if (LeadingZeros == 32 && TrailingZeros > ShAmt) {
        SDNode *SRLIW = CurDAG->getMachineNode(
            RISCV::SRLIW, DL, VT, N0->getOperand(0),
            CurDAG->getTargetConstant(TrailingZeros, DL, VT));
        SDNode *SLLI = CurDAG->getMachineNode(
            RISCV::SLLI, DL, VT, SDValue(SRLIW, 0),
            CurDAG->getTargetConstant(TrailingZeros - ShAmt, DL, VT));
        ReplaceNode(Node, SLLI);
        return;
      }
    }

    // Optimize (srl (and X, C2), C) ->
    //          (srli (slli X, (XLen-C3), (XLen-C3) + C)
    // Where C2 is a mask with C3 trailing ones.
    // Taking into account that the C2 may have had lower bits unset by
    // SimplifyDemandedBits. This avoids materializing the C2 immediate.
    // This pattern occurs when type legalizing right shifts for types with
    // less than XLen bits.
    Mask |= maskTrailingOnes<uint64_t>(ShAmt);
    if (!isMask_64(Mask))
      break;
    unsigned TrailingOnes = countTrailingOnes(Mask);
    if (ShAmt >= TrailingOnes)
      break;
    // If the mask has 32 trailing ones, use SRLIW.
    if (TrailingOnes == 32) {
      SDNode *SRLIW =
          CurDAG->getMachineNode(RISCV::SRLIW, DL, VT, N0->getOperand(0),
                                 CurDAG->getTargetConstant(ShAmt, DL, VT));
      ReplaceNode(Node, SRLIW);
      return;
    }

    // Only do the remaining transforms if the shift has one use.
    if (!N0.hasOneUse())
      break;

    // If C2 is (1 << ShAmt) use bexti if possible.
    if (Subtarget->hasStdExtZbs() && ShAmt + 1 == TrailingOnes) {
      SDNode *BEXTI =
          CurDAG->getMachineNode(RISCV::BEXTI, DL, VT, N0->getOperand(0),
                                 CurDAG->getTargetConstant(ShAmt, DL, VT));
      ReplaceNode(Node, BEXTI);
      return;
    }
    unsigned LShAmt = Subtarget->getXLen() - TrailingOnes;
    SDNode *SLLI =
        CurDAG->getMachineNode(RISCV::SLLI, DL, VT, N0->getOperand(0),
                               CurDAG->getTargetConstant(LShAmt, DL, VT));
    SDNode *SRLI = CurDAG->getMachineNode(
        RISCV::SRLI, DL, VT, SDValue(SLLI, 0),
        CurDAG->getTargetConstant(LShAmt + ShAmt, DL, VT));
    ReplaceNode(Node, SRLI);
    return;
  }
  case ISD::SRA: {
    // Optimize (sra (sext_inreg X, i16), C) ->
    //          (srai (slli X, (XLen-16), (XLen-16) + C)
    // And      (sra (sext_inreg X, i8), C) ->
    //          (srai (slli X, (XLen-8), (XLen-8) + C)
    // This can occur when Zbb is enabled, which makes sext_inreg i16/i8 legal.
    // This transform matches the code we get without Zbb. The shifts are more
    // compressible, and this can help expose CSE opportunities in the sdiv by
    // constant optimization.
    auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
    if (!N1C)
      break;
    SDValue N0 = Node->getOperand(0);
    if (N0.getOpcode() != ISD::SIGN_EXTEND_INREG || !N0.hasOneUse())
      break;
    unsigned ShAmt = N1C->getZExtValue();
    unsigned ExtSize =
        cast<VTSDNode>(N0.getOperand(1))->getVT().getSizeInBits();
    // ExtSize of 32 should use sraiw via tablegen pattern.
    if (ExtSize >= 32 || ShAmt >= ExtSize)
      break;
    unsigned LShAmt = Subtarget->getXLen() - ExtSize;
    SDNode *SLLI =
        CurDAG->getMachineNode(RISCV::SLLI, DL, VT, N0->getOperand(0),
                               CurDAG->getTargetConstant(LShAmt, DL, VT));
    SDNode *SRAI = CurDAG->getMachineNode(
        RISCV::SRAI, DL, VT, SDValue(SLLI, 0),
        CurDAG->getTargetConstant(LShAmt + ShAmt, DL, VT));
    ReplaceNode(Node, SRAI);
    return;
  }
  case ISD::OR:
  case ISD::XOR:
    if (tryShrinkShlLogicImm(Node))
      return;

    break;
  case ISD::AND: {
    auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
    if (!N1C)
      break;

    SDValue N0 = Node->getOperand(0);

    bool LeftShift = N0.getOpcode() == ISD::SHL;
    if (LeftShift || N0.getOpcode() == ISD::SRL) {
      auto *C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
      if (!C)
        break;
      unsigned C2 = C->getZExtValue();
      unsigned XLen = Subtarget->getXLen();
      assert((C2 > 0 && C2 < XLen) && "Unexpected shift amount!");

      uint64_t C1 = N1C->getZExtValue();

      // Keep track of whether this is a c.andi. If we can't use c.andi, the
      // shift pair might offer more compression opportunities.
      // TODO: We could check for C extension here, but we don't have many lit
      // tests with the C extension enabled so not checking gets better
      // coverage.
      // TODO: What if ANDI faster than shift?
      bool IsCANDI = isInt<6>(N1C->getSExtValue());

      // Clear irrelevant bits in the mask.
      if (LeftShift)
        C1 &= maskTrailingZeros<uint64_t>(C2);
      else
        C1 &= maskTrailingOnes<uint64_t>(XLen - C2);

      // Some transforms should only be done if the shift has a single use or
      // the AND would become (srli (slli X, 32), 32)
      bool OneUseOrZExtW = N0.hasOneUse() || C1 == UINT64_C(0xFFFFFFFF);

      SDValue X = N0.getOperand(0);

      // Turn (and (srl x, c2) c1) -> (srli (slli x, c3-c2), c3) if c1 is a mask
      // with c3 leading zeros.
      if (!LeftShift && isMask_64(C1)) {
        unsigned Leading = XLen - llvm::bit_width(C1);
        if (C2 < Leading) {
          // If the number of leading zeros is C2+32 this can be SRLIW.
          if (C2 + 32 == Leading) {
            SDNode *SRLIW = CurDAG->getMachineNode(
                RISCV::SRLIW, DL, VT, X, CurDAG->getTargetConstant(C2, DL, VT));
            ReplaceNode(Node, SRLIW);
            return;
          }

          // (and (srl (sexti32 Y), c2), c1) -> (srliw (sraiw Y, 31), c3 - 32)
          // if c1 is a mask with c3 leading zeros and c2 >= 32 and c3-c2==1.
          //
          // This pattern occurs when (i32 (srl (sra 31), c3 - 32)) is type
          // legalized and goes through DAG combine.
          if (C2 >= 32 && (Leading - C2) == 1 && N0.hasOneUse() &&
              X.getOpcode() == ISD::SIGN_EXTEND_INREG &&
              cast<VTSDNode>(X.getOperand(1))->getVT() == MVT::i32) {
            SDNode *SRAIW =
                CurDAG->getMachineNode(RISCV::SRAIW, DL, VT, X.getOperand(0),
                                       CurDAG->getTargetConstant(31, DL, VT));
            SDNode *SRLIW = CurDAG->getMachineNode(
                RISCV::SRLIW, DL, VT, SDValue(SRAIW, 0),
                CurDAG->getTargetConstant(Leading - 32, DL, VT));
            ReplaceNode(Node, SRLIW);
            return;
          }

          // (srli (slli x, c3-c2), c3).
          // Skip if we could use (zext.w (sraiw X, C2)).
          bool Skip = Subtarget->hasStdExtZba() && Leading == 32 &&
                      X.getOpcode() == ISD::SIGN_EXTEND_INREG &&
                      cast<VTSDNode>(X.getOperand(1))->getVT() == MVT::i32;
          // Also Skip if we can use bexti.
          Skip |= Subtarget->hasStdExtZbs() && Leading == XLen - 1;
          if (OneUseOrZExtW && !Skip) {
            SDNode *SLLI = CurDAG->getMachineNode(
                RISCV::SLLI, DL, VT, X,
                CurDAG->getTargetConstant(Leading - C2, DL, VT));
            SDNode *SRLI = CurDAG->getMachineNode(
                RISCV::SRLI, DL, VT, SDValue(SLLI, 0),
                CurDAG->getTargetConstant(Leading, DL, VT));
            ReplaceNode(Node, SRLI);
            return;
          }
        }
      }

      // Turn (and (shl x, c2), c1) -> (srli (slli c2+c3), c3) if c1 is a mask
      // shifted by c2 bits with c3 leading zeros.
      if (LeftShift && isShiftedMask_64(C1)) {
        unsigned Leading = XLen - llvm::bit_width(C1);

        if (C2 + Leading < XLen &&
            C1 == (maskTrailingOnes<uint64_t>(XLen - (C2 + Leading)) << C2)) {
          // Use slli.uw when possible.
          if ((XLen - (C2 + Leading)) == 32 && Subtarget->hasStdExtZba()) {
            SDNode *SLLI_UW =
                CurDAG->getMachineNode(RISCV::SLLI_UW, DL, VT, X,
                                       CurDAG->getTargetConstant(C2, DL, VT));
            ReplaceNode(Node, SLLI_UW);
            return;
          }

          // (srli (slli c2+c3), c3)
          if (OneUseOrZExtW && !IsCANDI) {
            SDNode *SLLI = CurDAG->getMachineNode(
                RISCV::SLLI, DL, VT, X,
                CurDAG->getTargetConstant(C2 + Leading, DL, VT));
            SDNode *SRLI = CurDAG->getMachineNode(
                RISCV::SRLI, DL, VT, SDValue(SLLI, 0),
                CurDAG->getTargetConstant(Leading, DL, VT));
            ReplaceNode(Node, SRLI);
            return;
          }
        }
      }

      // Turn (and (shr x, c2), c1) -> (slli (srli x, c2+c3), c3) if c1 is a
      // shifted mask with c2 leading zeros and c3 trailing zeros.
      if (!LeftShift && isShiftedMask_64(C1)) {
        unsigned Leading = XLen - llvm::bit_width(C1);
        unsigned Trailing = countTrailingZeros(C1);
        if (Leading == C2 && C2 + Trailing < XLen && OneUseOrZExtW &&
            !IsCANDI) {
          unsigned SrliOpc = RISCV::SRLI;
          // If the input is zexti32 we should use SRLIW.
          if (X.getOpcode() == ISD::AND &&
              isa<ConstantSDNode>(X.getOperand(1)) &&
              X.getConstantOperandVal(1) == UINT64_C(0xFFFFFFFF)) {
            SrliOpc = RISCV::SRLIW;
            X = X.getOperand(0);
          }
          SDNode *SRLI = CurDAG->getMachineNode(
              SrliOpc, DL, VT, X,
              CurDAG->getTargetConstant(C2 + Trailing, DL, VT));
          SDNode *SLLI = CurDAG->getMachineNode(
              RISCV::SLLI, DL, VT, SDValue(SRLI, 0),
              CurDAG->getTargetConstant(Trailing, DL, VT));
          ReplaceNode(Node, SLLI);
          return;
        }
        // If the leading zero count is C2+32, we can use SRLIW instead of SRLI.
        if (Leading > 32 && (Leading - 32) == C2 && C2 + Trailing < 32 &&
            OneUseOrZExtW && !IsCANDI) {
          SDNode *SRLIW = CurDAG->getMachineNode(
              RISCV::SRLIW, DL, VT, X,
              CurDAG->getTargetConstant(C2 + Trailing, DL, VT));
          SDNode *SLLI = CurDAG->getMachineNode(
              RISCV::SLLI, DL, VT, SDValue(SRLIW, 0),
              CurDAG->getTargetConstant(Trailing, DL, VT));
          ReplaceNode(Node, SLLI);
          return;
        }
      }

      // Turn (and (shl x, c2), c1) -> (slli (srli x, c3-c2), c3) if c1 is a
      // shifted mask with no leading zeros and c3 trailing zeros.
      if (LeftShift && isShiftedMask_64(C1)) {
        unsigned Leading = XLen - llvm::bit_width(C1);
        unsigned Trailing = countTrailingZeros(C1);
        if (Leading == 0 && C2 < Trailing && OneUseOrZExtW && !IsCANDI) {
          SDNode *SRLI = CurDAG->getMachineNode(
              RISCV::SRLI, DL, VT, X,
              CurDAG->getTargetConstant(Trailing - C2, DL, VT));
          SDNode *SLLI = CurDAG->getMachineNode(
              RISCV::SLLI, DL, VT, SDValue(SRLI, 0),
              CurDAG->getTargetConstant(Trailing, DL, VT));
          ReplaceNode(Node, SLLI);
          return;
        }
        // If we have (32-C2) leading zeros, we can use SRLIW instead of SRLI.
        if (C2 < Trailing && Leading + C2 == 32 && OneUseOrZExtW && !IsCANDI) {
          SDNode *SRLIW = CurDAG->getMachineNode(
              RISCV::SRLIW, DL, VT, X,
              CurDAG->getTargetConstant(Trailing - C2, DL, VT));
          SDNode *SLLI = CurDAG->getMachineNode(
              RISCV::SLLI, DL, VT, SDValue(SRLIW, 0),
              CurDAG->getTargetConstant(Trailing, DL, VT));
          ReplaceNode(Node, SLLI);
          return;
        }
      }
    }

    if (tryShrinkShlLogicImm(Node))
      return;

    break;
  }
  case ISD::MUL: {
    // Special case for calculating (mul (and X, C2), C1) where the full product
    // fits in XLen bits. We can shift X left by the number of leading zeros in
    // C2 and shift C1 left by XLen-lzcnt(C2). This will ensure the final
    // product has XLen trailing zeros, putting it in the output of MULHU. This
    // can avoid materializing a constant in a register for C2.

    // RHS should be a constant.
    auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
    if (!N1C || !N1C->hasOneUse())
      break;

    // LHS should be an AND with constant.
    SDValue N0 = Node->getOperand(0);
    if (N0.getOpcode() != ISD::AND || !isa<ConstantSDNode>(N0.getOperand(1)))
      break;

    uint64_t C2 = cast<ConstantSDNode>(N0.getOperand(1))->getZExtValue();

    // Constant should be a mask.
    if (!isMask_64(C2))
      break;

    // If this can be an ANDI, ZEXT.H or ZEXT.W, don't do this if the ANDI/ZEXT
    // has multiple users or the constant is a simm12. This prevents inserting
    // a shift and still have uses of the AND/ZEXT. Shifting a simm12 will
    // likely make it more costly to materialize. Otherwise, using a SLLI
    // might allow it to be compressed.
    bool IsANDIOrZExt =
        isInt<12>(C2) ||
        (C2 == UINT64_C(0xFFFF) && Subtarget->hasStdExtZbb()) ||
        (C2 == UINT64_C(0xFFFFFFFF) && Subtarget->hasStdExtZba());
    if (IsANDIOrZExt && (isInt<12>(N1C->getSExtValue()) || !N0.hasOneUse()))
      break;

    // We need to shift left the AND input and C1 by a total of XLen bits.

    // How far left do we need to shift the AND input?
    unsigned XLen = Subtarget->getXLen();
    unsigned LeadingZeros = XLen - llvm::bit_width(C2);

    // The constant gets shifted by the remaining amount unless that would
    // shift bits out.
    uint64_t C1 = N1C->getZExtValue();
    unsigned ConstantShift = XLen - LeadingZeros;
    if (ConstantShift > (XLen - llvm::bit_width(C1)))
      break;

    uint64_t ShiftedC1 = C1 << ConstantShift;
    // If this RV32, we need to sign extend the constant.
    if (XLen == 32)
      ShiftedC1 = SignExtend64<32>(ShiftedC1);

    // Create (mulhu (slli X, lzcnt(C2)), C1 << (XLen - lzcnt(C2))).
    SDNode *Imm = selectImm(CurDAG, DL, VT, ShiftedC1, *Subtarget);
    SDNode *SLLI =
        CurDAG->getMachineNode(RISCV::SLLI, DL, VT, N0.getOperand(0),
                               CurDAG->getTargetConstant(LeadingZeros, DL, VT));
    SDNode *MULHU = CurDAG->getMachineNode(RISCV::MULHU, DL, VT,
                                           SDValue(SLLI, 0), SDValue(Imm, 0));
    ReplaceNode(Node, MULHU);
    return;
  }
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IntNo = Node->getConstantOperandVal(0);
    switch (IntNo) {
      // By default we do not custom select any intrinsic.
    default:
      break;
    case Intrinsic::riscv_vmsgeu:
    case Intrinsic::riscv_vmsge: {
      SDValue Src1 = Node->getOperand(1);
      SDValue Src2 = Node->getOperand(2);
      bool IsUnsigned = IntNo == Intrinsic::riscv_vmsgeu;
      bool IsCmpUnsignedZero = false;
      // Only custom select scalar second operand.
      if (Src2.getValueType() != XLenVT)
        break;
      // Small constants are handled with patterns.
      if (auto *C = dyn_cast<ConstantSDNode>(Src2)) {
        int64_t CVal = C->getSExtValue();
        if (CVal >= -15 && CVal <= 16) {
          if (!IsUnsigned || CVal != 0)
            break;
          IsCmpUnsignedZero = true;
        }
      }
      MVT Src1VT = Src1.getSimpleValueType();
      unsigned VMSLTOpcode, VMNANDOpcode, VMSetOpcode;
      switch (RISCVTargetLowering::getLMUL(Src1VT)) {
      default:
        llvm_unreachable("Unexpected LMUL!");
#define CASE_VMSLT_VMNAND_VMSET_OPCODES(lmulenum, suffix, suffix_b)            \
  case RISCVII::VLMUL::lmulenum:                                               \
    VMSLTOpcode = IsUnsigned ? RISCV::PseudoVMSLTU_VX_##suffix                 \
                             : RISCV::PseudoVMSLT_VX_##suffix;                 \
    VMNANDOpcode = RISCV::PseudoVMNAND_MM_##suffix;                            \
    VMSetOpcode = RISCV::PseudoVMSET_M_##suffix_b;                             \
    break;
        CASE_VMSLT_VMNAND_VMSET_OPCODES(LMUL_F8, MF8, B1)
        CASE_VMSLT_VMNAND_VMSET_OPCODES(LMUL_F4, MF4, B2)
        CASE_VMSLT_VMNAND_VMSET_OPCODES(LMUL_F2, MF2, B4)
        CASE_VMSLT_VMNAND_VMSET_OPCODES(LMUL_1, M1, B8)
        CASE_VMSLT_VMNAND_VMSET_OPCODES(LMUL_2, M2, B16)
        CASE_VMSLT_VMNAND_VMSET_OPCODES(LMUL_4, M4, B32)
        CASE_VMSLT_VMNAND_VMSET_OPCODES(LMUL_8, M8, B64)
#undef CASE_VMSLT_VMNAND_VMSET_OPCODES
      }
      SDValue SEW = CurDAG->getTargetConstant(
          Log2_32(Src1VT.getScalarSizeInBits()), DL, XLenVT);
      SDValue VL;
      selectVLOp(Node->getOperand(3), VL);

      // If vmsgeu with 0 immediate, expand it to vmset.
      if (IsCmpUnsignedZero) {
        ReplaceNode(Node, CurDAG->getMachineNode(VMSetOpcode, DL, VT, VL, SEW));
        return;
      }

      // Expand to
      // vmslt{u}.vx vd, va, x; vmnand.mm vd, vd, vd
      SDValue Cmp = SDValue(
          CurDAG->getMachineNode(VMSLTOpcode, DL, VT, {Src1, Src2, VL, SEW}),
          0);
      ReplaceNode(Node, CurDAG->getMachineNode(VMNANDOpcode, DL, VT,
                                               {Cmp, Cmp, VL, SEW}));
      return;
    }
    case Intrinsic::riscv_vmsgeu_mask:
    case Intrinsic::riscv_vmsge_mask: {
      SDValue Src1 = Node->getOperand(2);
      SDValue Src2 = Node->getOperand(3);
      bool IsUnsigned = IntNo == Intrinsic::riscv_vmsgeu_mask;
      bool IsCmpUnsignedZero = false;
      // Only custom select scalar second operand.
      if (Src2.getValueType() != XLenVT)
        break;
      // Small constants are handled with patterns.
      if (auto *C = dyn_cast<ConstantSDNode>(Src2)) {
        int64_t CVal = C->getSExtValue();
        if (CVal >= -15 && CVal <= 16) {
          if (!IsUnsigned || CVal != 0)
            break;
          IsCmpUnsignedZero = true;
        }
      }
      MVT Src1VT = Src1.getSimpleValueType();
      unsigned VMSLTOpcode, VMSLTMaskOpcode, VMXOROpcode, VMANDNOpcode,
          VMOROpcode;
      switch (RISCVTargetLowering::getLMUL(Src1VT)) {
      default:
        llvm_unreachable("Unexpected LMUL!");
#define CASE_VMSLT_OPCODES(lmulenum, suffix, suffix_b)                         \
  case RISCVII::VLMUL::lmulenum:                                               \
    VMSLTOpcode = IsUnsigned ? RISCV::PseudoVMSLTU_VX_##suffix                 \
                             : RISCV::PseudoVMSLT_VX_##suffix;                 \
    VMSLTMaskOpcode = IsUnsigned ? RISCV::PseudoVMSLTU_VX_##suffix##_MASK      \
                                 : RISCV::PseudoVMSLT_VX_##suffix##_MASK;      \
    break;
        CASE_VMSLT_OPCODES(LMUL_F8, MF8, B1)
        CASE_VMSLT_OPCODES(LMUL_F4, MF4, B2)
        CASE_VMSLT_OPCODES(LMUL_F2, MF2, B4)
        CASE_VMSLT_OPCODES(LMUL_1, M1, B8)
        CASE_VMSLT_OPCODES(LMUL_2, M2, B16)
        CASE_VMSLT_OPCODES(LMUL_4, M4, B32)
        CASE_VMSLT_OPCODES(LMUL_8, M8, B64)
#undef CASE_VMSLT_OPCODES
      }
      // Mask operations use the LMUL from the mask type.
      switch (RISCVTargetLowering::getLMUL(VT)) {
      default:
        llvm_unreachable("Unexpected LMUL!");
#define CASE_VMXOR_VMANDN_VMOR_OPCODES(lmulenum, suffix)                       \
  case RISCVII::VLMUL::lmulenum:                                               \
    VMXOROpcode = RISCV::PseudoVMXOR_MM_##suffix;                              \
    VMANDNOpcode = RISCV::PseudoVMANDN_MM_##suffix;                            \
    VMOROpcode = RISCV::PseudoVMOR_MM_##suffix;                                \
    break;
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_F8, MF8)
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_F4, MF4)
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_F2, MF2)
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_1, M1)
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_2, M2)
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_4, M4)
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_8, M8)
#undef CASE_VMXOR_VMANDN_VMOR_OPCODES
      }
      SDValue SEW = CurDAG->getTargetConstant(
          Log2_32(Src1VT.getScalarSizeInBits()), DL, XLenVT);
      SDValue MaskSEW = CurDAG->getTargetConstant(0, DL, XLenVT);
      SDValue VL;
      selectVLOp(Node->getOperand(5), VL);
      SDValue MaskedOff = Node->getOperand(1);
      SDValue Mask = Node->getOperand(4);

      // If vmsgeu_mask with 0 immediate, expand it to vmor mask, maskedoff.
      if (IsCmpUnsignedZero) {
        // We don't need vmor if the MaskedOff and the Mask are the same
        // value.
        if (Mask == MaskedOff) {
          ReplaceUses(Node, Mask.getNode());
          return;
        }
        ReplaceNode(Node,
                    CurDAG->getMachineNode(VMOROpcode, DL, VT,
                                           {Mask, MaskedOff, VL, MaskSEW}));
        return;
      }

      // If the MaskedOff value and the Mask are the same value use
      // vmslt{u}.vx vt, va, x;  vmandn.mm vd, vd, vt
      // This avoids needing to copy v0 to vd before starting the next sequence.
      if (Mask == MaskedOff) {
        SDValue Cmp = SDValue(
            CurDAG->getMachineNode(VMSLTOpcode, DL, VT, {Src1, Src2, VL, SEW}),
            0);
        ReplaceNode(Node, CurDAG->getMachineNode(VMANDNOpcode, DL, VT,
                                                 {Mask, Cmp, VL, MaskSEW}));
        return;
      }

      // Mask needs to be copied to V0.
      SDValue Chain = CurDAG->getCopyToReg(CurDAG->getEntryNode(), DL,
                                           RISCV::V0, Mask, SDValue());
      SDValue Glue = Chain.getValue(1);
      SDValue V0 = CurDAG->getRegister(RISCV::V0, VT);

      // Otherwise use
      // vmslt{u}.vx vd, va, x, v0.t; vmxor.mm vd, vd, v0
      // The result is mask undisturbed.
      // We use the same instructions to emulate mask agnostic behavior, because
      // the agnostic result can be either undisturbed or all 1.
      SDValue Cmp = SDValue(
          CurDAG->getMachineNode(VMSLTMaskOpcode, DL, VT,
                                 {MaskedOff, Src1, Src2, V0, VL, SEW, Glue}),
          0);
      // vmxor.mm vd, vd, v0 is used to update active value.
      ReplaceNode(Node, CurDAG->getMachineNode(VMXOROpcode, DL, VT,
                                               {Cmp, Mask, VL, MaskSEW}));
      return;
    }
    case Intrinsic::riscv_vsetvli_opt:
    case Intrinsic::riscv_vsetvlimax_opt:
      return selectVSETVLI(Node);
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
    case Intrinsic::riscv_vsetvlimax:
      return selectVSETVLI(Node);
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
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      unsigned CurOp = 2;
      // Masked intrinsic only have TU version pseduo instructions.
      bool IsTU = IsMasked || !Node->getOperand(CurOp).isUndef();
      SmallVector<SDValue, 8> Operands;
      if (IsTU)
        Operands.push_back(Node->getOperand(CurOp++));
      else
        // Skip the undef passthru operand for nomask TA version pseudo
        CurOp++;

      MVT IndexVT;
      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                                 /*IsStridedOrIndexed*/ true, Operands,
                                 /*IsLoad=*/true, &IndexVT);

      assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
             "Element count mismatch");

      RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
      RISCVII::VLMUL IndexLMUL = RISCVTargetLowering::getLMUL(IndexVT);
      unsigned IndexLog2EEW = Log2_32(IndexVT.getScalarSizeInBits());
      if (IndexLog2EEW == 6 && !Subtarget->is64Bit()) {
        report_fatal_error("The V extension does not support EEW=64 for index "
                           "values when XLEN=32");
      }
      const RISCV::VLX_VSXPseudo *P = RISCV::getVLXPseudo(
          IsMasked, IsTU, IsOrdered, IndexLog2EEW, static_cast<unsigned>(LMUL),
          static_cast<unsigned>(IndexLMUL));
      MachineSDNode *Load =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);

      if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

      ReplaceNode(Node, Load);
      return;
    }
    case Intrinsic::riscv_vlm:
    case Intrinsic::riscv_vle:
    case Intrinsic::riscv_vle_mask:
    case Intrinsic::riscv_vlse:
    case Intrinsic::riscv_vlse_mask: {
      bool IsMasked = IntNo == Intrinsic::riscv_vle_mask ||
                      IntNo == Intrinsic::riscv_vlse_mask;
      bool IsStrided =
          IntNo == Intrinsic::riscv_vlse || IntNo == Intrinsic::riscv_vlse_mask;

      MVT VT = Node->getSimpleValueType(0);
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      unsigned CurOp = 2;
      // The riscv_vlm intrinsic are always tail agnostic and no passthru operand.
      bool HasPassthruOperand = IntNo != Intrinsic::riscv_vlm;
      // Masked intrinsic only have TU version pseduo instructions.
      bool IsTU = HasPassthruOperand &&
                  (IsMasked || !Node->getOperand(CurOp).isUndef());
      SmallVector<SDValue, 8> Operands;
      if (IsTU)
        Operands.push_back(Node->getOperand(CurOp++));
      else if (HasPassthruOperand)
        // Skip the undef passthru operand for nomask TA version pseudo
        CurOp++;

      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked, IsStrided,
                                 Operands, /*IsLoad=*/true);

      RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
      const RISCV::VLEPseudo *P =
          RISCV::getVLEPseudo(IsMasked, IsTU, IsStrided, /*FF*/ false, Log2SEW,
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
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      unsigned CurOp = 2;
      // Masked intrinsic only have TU version pseduo instructions.
      bool IsTU = IsMasked || !Node->getOperand(CurOp).isUndef();
      SmallVector<SDValue, 7> Operands;
      if (IsTU)
        Operands.push_back(Node->getOperand(CurOp++));
      else
        // Skip the undef passthru operand for nomask TA version pseudo
        CurOp++;

      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                                 /*IsStridedOrIndexed*/ false, Operands,
                                 /*IsLoad=*/true);

      RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
      const RISCV::VLEPseudo *P =
          RISCV::getVLEPseudo(IsMasked, IsTU, /*Strided*/ false, /*FF*/ true,
                              Log2SEW, static_cast<unsigned>(LMUL));
      MachineSDNode *Load = CurDAG->getMachineNode(
          P->Pseudo, DL, Node->getVTList(), Operands);
      if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

      ReplaceNode(Node, Load);
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
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      unsigned CurOp = 2;
      SmallVector<SDValue, 8> Operands;
      Operands.push_back(Node->getOperand(CurOp++)); // Store value.

      MVT IndexVT;
      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                                 /*IsStridedOrIndexed*/ true, Operands,
                                 /*IsLoad=*/false, &IndexVT);

      assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
             "Element count mismatch");

      RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
      RISCVII::VLMUL IndexLMUL = RISCVTargetLowering::getLMUL(IndexVT);
      unsigned IndexLog2EEW = Log2_32(IndexVT.getScalarSizeInBits());
      if (IndexLog2EEW == 6 && !Subtarget->is64Bit()) {
        report_fatal_error("The V extension does not support EEW=64 for index "
                           "values when XLEN=32");
      }
      const RISCV::VLX_VSXPseudo *P = RISCV::getVSXPseudo(
          IsMasked, /*TU*/ false, IsOrdered, IndexLog2EEW,
          static_cast<unsigned>(LMUL), static_cast<unsigned>(IndexLMUL));
      MachineSDNode *Store =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);

      if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        CurDAG->setNodeMemRefs(Store, {MemOp->getMemOperand()});

      ReplaceNode(Node, Store);
      return;
    }
    case Intrinsic::riscv_vsm:
    case Intrinsic::riscv_vse:
    case Intrinsic::riscv_vse_mask:
    case Intrinsic::riscv_vsse:
    case Intrinsic::riscv_vsse_mask: {
      bool IsMasked = IntNo == Intrinsic::riscv_vse_mask ||
                      IntNo == Intrinsic::riscv_vsse_mask;
      bool IsStrided =
          IntNo == Intrinsic::riscv_vsse || IntNo == Intrinsic::riscv_vsse_mask;

      MVT VT = Node->getOperand(2)->getSimpleValueType(0);
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      unsigned CurOp = 2;
      SmallVector<SDValue, 8> Operands;
      Operands.push_back(Node->getOperand(CurOp++)); // Store value.

      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked, IsStrided,
                                 Operands);

      RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
      const RISCV::VSEPseudo *P = RISCV::getVSEPseudo(
          IsMasked, IsStrided, Log2SEW, static_cast<unsigned>(LMUL));
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

    const RISCVTargetLowering &TLI = *Subtarget->getTargetLowering();
    MVT SubVecContainerVT = SubVecVT;
    // Establish the correct scalable-vector types for any fixed-length type.
    if (SubVecVT.isFixedLengthVector())
      SubVecContainerVT = TLI.getContainerForFixedLengthVector(SubVecVT);
    if (VT.isFixedLengthVector())
      VT = TLI.getContainerForFixedLengthVector(VT);

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

    RISCVII::VLMUL SubVecLMUL = RISCVTargetLowering::getLMUL(SubVecContainerVT);
    bool IsSubVecPartReg = SubVecLMUL == RISCVII::VLMUL::LMUL_F2 ||
                           SubVecLMUL == RISCVII::VLMUL::LMUL_F4 ||
                           SubVecLMUL == RISCVII::VLMUL::LMUL_F8;
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
      ReplaceNode(Node, NewNode);
      return;
    }

    SDValue Insert = CurDAG->getTargetInsertSubreg(SubRegIdx, DL, VT, V, SubV);
    ReplaceNode(Node, Insert.getNode());
    return;
  }
  case ISD::EXTRACT_SUBVECTOR: {
    SDValue V = Node->getOperand(0);
    auto Idx = Node->getConstantOperandVal(1);
    MVT InVT = V.getSimpleValueType();
    SDLoc DL(V);

    const RISCVTargetLowering &TLI = *Subtarget->getTargetLowering();
    MVT SubVecContainerVT = VT;
    // Establish the correct scalable-vector types for any fixed-length type.
    if (VT.isFixedLengthVector())
      SubVecContainerVT = TLI.getContainerForFixedLengthVector(VT);
    if (InVT.isFixedLengthVector())
      InVT = TLI.getContainerForFixedLengthVector(InVT);

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
      ReplaceNode(Node, NewNode);
      return;
    }

    SDValue Extract = CurDAG->getTargetExtractSubreg(SubRegIdx, DL, VT, V);
    ReplaceNode(Node, Extract.getNode());
    return;
  }
  case RISCVISD::VMV_S_X_VL:
  case RISCVISD::VFMV_S_F_VL:
  case RISCVISD::VMV_V_X_VL:
  case RISCVISD::VFMV_V_F_VL: {
    // Only if we have optimized zero-stride vector load.
    if (!Subtarget->hasOptimizedZeroStrideLoad())
      break;

    // Try to match splat of a scalar load to a strided load with stride of x0.
    bool IsScalarMove = Node->getOpcode() == RISCVISD::VMV_S_X_VL ||
                        Node->getOpcode() == RISCVISD::VFMV_S_F_VL;
    if (!Node->getOperand(0).isUndef())
      break;
    SDValue Src = Node->getOperand(1);
    auto *Ld = dyn_cast<LoadSDNode>(Src);
    if (!Ld)
      break;
    EVT MemVT = Ld->getMemoryVT();
    // The memory VT should be the same size as the element type.
    if (MemVT.getStoreSize() != VT.getVectorElementType().getStoreSize())
      break;
    if (!IsProfitableToFold(Src, Node, Node) ||
        !IsLegalToFold(Src, Node, Node, TM.getOptLevel()))
      break;

    SDValue VL;
    if (IsScalarMove) {
      // We could deal with more VL if we update the VSETVLI insert pass to
      // avoid introducing more VSETVLI.
      if (!isOneConstant(Node->getOperand(2)))
        break;
      selectVLOp(Node->getOperand(2), VL);
    } else
      selectVLOp(Node->getOperand(2), VL);

    unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());
    SDValue SEW = CurDAG->getTargetConstant(Log2SEW, DL, XLenVT);

    SDValue Operands[] = {Ld->getBasePtr(),
                          CurDAG->getRegister(RISCV::X0, XLenVT), VL, SEW,
                          Ld->getChain()};

    RISCVII::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
    const RISCV::VLEPseudo *P = RISCV::getVLEPseudo(
        /*IsMasked*/ false, /*IsTU*/ false, /*IsStrided*/ true, /*FF*/ false,
        Log2SEW, static_cast<unsigned>(LMUL));
    MachineSDNode *Load =
        CurDAG->getMachineNode(P->Pseudo, DL, {VT, MVT::Other}, Operands);
    // Update the chain.
    ReplaceUses(Src.getValue(1), SDValue(Load, 1));
    // Record the mem-refs
    CurDAG->setNodeMemRefs(Load, {Ld->getMemOperand()});
    // Replace the splat with the vlse.
    ReplaceNode(Node, Load);
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

bool RISCVDAGToDAGISel::SelectAddrFrameIndex(SDValue Addr, SDValue &Base,
                                             SDValue &Offset) {
  if (auto *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), Subtarget->getXLenVT());
    Offset = CurDAG->getTargetConstant(0, SDLoc(Addr), Subtarget->getXLenVT());
    return true;
  }

  return false;
}

// Select a frame index and an optional immediate offset from an ADD or OR.
bool RISCVDAGToDAGISel::SelectFrameAddrRegImm(SDValue Addr, SDValue &Base,
                                              SDValue &Offset) {
  if (SelectAddrFrameIndex(Addr, Base, Offset))
    return true;

  if (!CurDAG->isBaseWithConstantOffset(Addr))
    return false;

  if (auto *FIN = dyn_cast<FrameIndexSDNode>(Addr.getOperand(0))) {
    int64_t CVal = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    if (isInt<12>(CVal)) {
      Base = CurDAG->getTargetFrameIndex(FIN->getIndex(),
                                         Subtarget->getXLenVT());
      Offset = CurDAG->getTargetConstant(CVal, SDLoc(Addr),
                                         Subtarget->getXLenVT());
      return true;
    }
  }

  return false;
}

// Fold constant addresses.
static bool selectConstantAddr(SelectionDAG *CurDAG, const SDLoc &DL,
                               const MVT VT, const RISCVSubtarget *Subtarget,
                               SDValue Addr, SDValue &Base, SDValue &Offset) {
  if (!isa<ConstantSDNode>(Addr))
    return false;

  int64_t CVal = cast<ConstantSDNode>(Addr)->getSExtValue();

  // If the constant is a simm12, we can fold the whole constant and use X0 as
  // the base. If the constant can be materialized with LUI+simm12, use LUI as
  // the base. We can't use generateInstSeq because it favors LUI+ADDIW.
  int64_t Lo12 = SignExtend64<12>(CVal);
  int64_t Hi = (uint64_t)CVal - (uint64_t)Lo12;
  if (!Subtarget->is64Bit() || isInt<32>(Hi)) {
    if (Hi) {
      int64_t Hi20 = (Hi >> 12) & 0xfffff;
      Base = SDValue(
          CurDAG->getMachineNode(RISCV::LUI, DL, VT,
                                 CurDAG->getTargetConstant(Hi20, DL, VT)),
          0);
    } else {
      Base = CurDAG->getRegister(RISCV::X0, VT);
    }
    Offset = CurDAG->getTargetConstant(Lo12, DL, VT);
    return true;
  }

  // Ask how constant materialization would handle this constant.
  RISCVMatInt::InstSeq Seq =
      RISCVMatInt::generateInstSeq(CVal, Subtarget->getFeatureBits());

  // If the last instruction would be an ADDI, we can fold its immediate and
  // emit the rest of the sequence as the base.
  if (Seq.back().getOpcode() != RISCV::ADDI)
    return false;
  Lo12 = Seq.back().getImm();

  // Drop the last instruction.
  Seq.pop_back();
  assert(!Seq.empty() && "Expected more instructions in sequence");

  Base = SDValue(selectImmSeq(CurDAG, DL, VT, Seq), 0);
  Offset = CurDAG->getTargetConstant(Lo12, DL, VT);
  return true;
}

// Is this ADD instruction only used as the base pointer of scalar loads and
// stores?
static bool isWorthFoldingAdd(SDValue Add) {
  for (auto *Use : Add->uses()) {
    if (Use->getOpcode() != ISD::LOAD && Use->getOpcode() != ISD::STORE &&
        Use->getOpcode() != ISD::ATOMIC_LOAD &&
        Use->getOpcode() != ISD::ATOMIC_STORE)
      return false;
    EVT VT = cast<MemSDNode>(Use)->getMemoryVT();
    if (!VT.isScalarInteger() && VT != MVT::f16 && VT != MVT::f32 &&
        VT != MVT::f64)
      return false;
    // Don't allow stores of the value. It must be used as the address.
    if (Use->getOpcode() == ISD::STORE &&
        cast<StoreSDNode>(Use)->getValue() == Add)
      return false;
    if (Use->getOpcode() == ISD::ATOMIC_STORE &&
        cast<AtomicSDNode>(Use)->getVal() == Add)
      return false;
  }

  return true;
}

bool RISCVDAGToDAGISel::SelectAddrRegImm(SDValue Addr, SDValue &Base,
                                         SDValue &Offset) {
  if (SelectAddrFrameIndex(Addr, Base, Offset))
    return true;

  SDLoc DL(Addr);
  MVT VT = Addr.getSimpleValueType();

  if (Addr.getOpcode() == RISCVISD::ADD_LO) {
    Base = Addr.getOperand(0);
    Offset = Addr.getOperand(1);
    return true;
  }

  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    int64_t CVal = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    if (isInt<12>(CVal)) {
      Base = Addr.getOperand(0);
      if (Base.getOpcode() == RISCVISD::ADD_LO) {
        SDValue LoOperand = Base.getOperand(1);
        if (auto *GA = dyn_cast<GlobalAddressSDNode>(LoOperand)) {
          // If the Lo in (ADD_LO hi, lo) is a global variable's address
          // (its low part, really), then we can rely on the alignment of that
          // variable to provide a margin of safety before low part can overflow
          // the 12 bits of the load/store offset. Check if CVal falls within
          // that margin; if so (low part + CVal) can't overflow.
          const DataLayout &DL = CurDAG->getDataLayout();
          Align Alignment = commonAlignment(
              GA->getGlobal()->getPointerAlignment(DL), GA->getOffset());
          if (CVal == 0 || Alignment > CVal) {
            int64_t CombinedOffset = CVal + GA->getOffset();
            Base = Base.getOperand(0);
            Offset = CurDAG->getTargetGlobalAddress(
                GA->getGlobal(), SDLoc(LoOperand), LoOperand.getValueType(),
                CombinedOffset, GA->getTargetFlags());
            return true;
          }
        }
      }

      if (auto *FIN = dyn_cast<FrameIndexSDNode>(Base))
        Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), VT);
      Offset = CurDAG->getTargetConstant(CVal, DL, VT);
      return true;
    }
  }

  // Handle ADD with large immediates.
  if (Addr.getOpcode() == ISD::ADD && isa<ConstantSDNode>(Addr.getOperand(1))) {
    int64_t CVal = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    assert(!isInt<12>(CVal) && "simm12 not already handled?");

    // Handle immediates in the range [-4096,-2049] or [2048, 4094]. We can use
    // an ADDI for part of the offset and fold the rest into the load/store.
    // This mirrors the AddiPair PatFrag in RISCVInstrInfo.td.
    if (isInt<12>(CVal / 2) && isInt<12>(CVal - CVal / 2)) {
      int64_t Adj = CVal < 0 ? -2048 : 2047;
      Base = SDValue(
          CurDAG->getMachineNode(RISCV::ADDI, DL, VT, Addr.getOperand(0),
                                 CurDAG->getTargetConstant(Adj, DL, VT)),
          0);
      Offset = CurDAG->getTargetConstant(CVal - Adj, DL, VT);
      return true;
    }

    // For larger immediates, we might be able to save one instruction from
    // constant materialization by folding the Lo12 bits of the immediate into
    // the address. We should only do this if the ADD is only used by loads and
    // stores that can fold the lo12 bits. Otherwise, the ADD will get iseled
    // separately with the full materialized immediate creating extra
    // instructions.
    if (isWorthFoldingAdd(Addr) &&
        selectConstantAddr(CurDAG, DL, VT, Subtarget, Addr.getOperand(1), Base,
                           Offset)) {
      // Insert an ADD instruction with the materialized Hi52 bits.
      Base = SDValue(
          CurDAG->getMachineNode(RISCV::ADD, DL, VT, Addr.getOperand(0), Base),
          0);
      return true;
    }
  }

  if (selectConstantAddr(CurDAG, DL, VT, Subtarget, Addr, Base, Offset))
    return true;

  Base = Addr;
  Offset = CurDAG->getTargetConstant(0, DL, VT);
  return true;
}

bool RISCVDAGToDAGISel::selectShiftMask(SDValue N, unsigned ShiftWidth,
                                        SDValue &ShAmt) {
  ShAmt = N;

  // Shift instructions on RISCV only read the lower 5 or 6 bits of the shift
  // amount. If there is an AND on the shift amount, we can bypass it if it
  // doesn't affect any of those bits.
  if (ShAmt.getOpcode() == ISD::AND && isa<ConstantSDNode>(ShAmt.getOperand(1))) {
    const APInt &AndMask = ShAmt.getConstantOperandAPInt(1);

    // Since the max shift amount is a power of 2 we can subtract 1 to make a
    // mask that covers the bits needed to represent all shift amounts.
    assert(isPowerOf2_32(ShiftWidth) && "Unexpected max shift amount!");
    APInt ShMask(AndMask.getBitWidth(), ShiftWidth - 1);

    if (ShMask.isSubsetOf(AndMask)) {
      ShAmt = ShAmt.getOperand(0);
    } else {
      // SimplifyDemandedBits may have optimized the mask so try restoring any
      // bits that are known zero.
      KnownBits Known = CurDAG->computeKnownBits(ShAmt.getOperand(0));
      if (!ShMask.isSubsetOf(AndMask | Known.Zero))
        return true;
      ShAmt = ShAmt.getOperand(0);
    }
  }

  if (ShAmt.getOpcode() == ISD::ADD &&
      isa<ConstantSDNode>(ShAmt.getOperand(1))) {
    uint64_t Imm = ShAmt.getConstantOperandVal(1);
    // If we are shifting by X+N where N == 0 mod Size, then just shift by X
    // to avoid the ADD.
    if (Imm != 0 && Imm % ShiftWidth == 0) {
      ShAmt = ShAmt.getOperand(0);
      return true;
    }
  } else if (ShAmt.getOpcode() == ISD::SUB &&
             isa<ConstantSDNode>(ShAmt.getOperand(0))) {
    uint64_t Imm = ShAmt.getConstantOperandVal(0);
    // If we are shifting by N-X where N == 0 mod Size, then just shift by -X to
    // generate a NEG instead of a SUB of a constant.
    if (Imm != 0 && Imm % ShiftWidth == 0) {
      SDLoc DL(ShAmt);
      EVT VT = ShAmt.getValueType();
      SDValue Zero = CurDAG->getRegister(RISCV::X0, VT);
      unsigned NegOpc = VT == MVT::i64 ? RISCV::SUBW : RISCV::SUB;
      MachineSDNode *Neg = CurDAG->getMachineNode(NegOpc, DL, VT, Zero,
                                                  ShAmt.getOperand(1));
      ShAmt = SDValue(Neg, 0);
      return true;
    }
    // If we are shifting by N-X where N == -1 mod Size, then just shift by ~X
    // to generate a NOT instead of a SUB of a constant.
    if (Imm % ShiftWidth == ShiftWidth - 1) {
      SDLoc DL(ShAmt);
      EVT VT = ShAmt.getValueType();
      MachineSDNode *Not =
          CurDAG->getMachineNode(RISCV::XORI, DL, VT, ShAmt.getOperand(1),
                                 CurDAG->getTargetConstant(-1, DL, VT));
      ShAmt = SDValue(Not, 0);
      return true;
    }
  }

  return true;
}

bool RISCVDAGToDAGISel::selectSExti32(SDValue N, SDValue &Val) {
  if (N.getOpcode() == ISD::SIGN_EXTEND_INREG &&
      cast<VTSDNode>(N.getOperand(1))->getVT() == MVT::i32) {
    Val = N.getOperand(0);
    return true;
  }
  MVT VT = N.getSimpleValueType();
  if (CurDAG->ComputeNumSignBits(N) > (VT.getSizeInBits() - 32)) {
    Val = N;
    return true;
  }

  return false;
}

bool RISCVDAGToDAGISel::selectZExtBits(SDValue N, unsigned Bits, SDValue &Val) {
  if (N.getOpcode() == ISD::AND) {
    auto *C = dyn_cast<ConstantSDNode>(N.getOperand(1));
    if (C && C->getZExtValue() == maskTrailingOnes<uint64_t>(Bits)) {
      Val = N.getOperand(0);
      return true;
    }
  }
  MVT VT = N.getSimpleValueType();
  APInt Mask = APInt::getBitsSetFrom(VT.getSizeInBits(), Bits);
  if (CurDAG->MaskedValueIsZero(N, Mask)) {
    Val = N;
    return true;
  }

  return false;
}

/// Look for various patterns that can be done with a SHL that can be folded
/// into a SHXADD. \p ShAmt contains 1, 2, or 3 and is set based on which
/// SHXADD we are trying to match.
bool RISCVDAGToDAGISel::selectSHXADDOp(SDValue N, unsigned ShAmt,
                                       SDValue &Val) {
  if (N.getOpcode() == ISD::AND && isa<ConstantSDNode>(N.getOperand(1))) {
    SDValue N0 = N.getOperand(0);

    bool LeftShift = N0.getOpcode() == ISD::SHL;
    if ((LeftShift || N0.getOpcode() == ISD::SRL) &&
        isa<ConstantSDNode>(N0.getOperand(1))) {
      uint64_t Mask = N.getConstantOperandVal(1);
      unsigned C2 = N0.getConstantOperandVal(1);

      unsigned XLen = Subtarget->getXLen();
      if (LeftShift)
        Mask &= maskTrailingZeros<uint64_t>(C2);
      else
        Mask &= maskTrailingOnes<uint64_t>(XLen - C2);

      // Look for (and (shl y, c2), c1) where c1 is a shifted mask with no
      // leading zeros and c3 trailing zeros. We can use an SRLI by c2+c3
      // followed by a SHXADD with c3 for the X amount.
      if (isShiftedMask_64(Mask)) {
        unsigned Leading = XLen - llvm::bit_width(Mask);
        unsigned Trailing = countTrailingZeros(Mask);
        if (LeftShift && Leading == 0 && C2 < Trailing && Trailing == ShAmt) {
          SDLoc DL(N);
          EVT VT = N.getValueType();
          Val = SDValue(CurDAG->getMachineNode(
                            RISCV::SRLI, DL, VT, N0.getOperand(0),
                            CurDAG->getTargetConstant(Trailing - C2, DL, VT)),
                        0);
          return true;
        }
        // Look for (and (shr y, c2), c1) where c1 is a shifted mask with c2
        // leading zeros and c3 trailing zeros. We can use an SRLI by C3
        // followed by a SHXADD using c3 for the X amount.
        if (!LeftShift && Leading == C2 && Trailing == ShAmt) {
          SDLoc DL(N);
          EVT VT = N.getValueType();
          Val = SDValue(
              CurDAG->getMachineNode(
                  RISCV::SRLI, DL, VT, N0.getOperand(0),
                  CurDAG->getTargetConstant(Leading + Trailing, DL, VT)),
              0);
          return true;
        }
      }
    }
  }

  bool LeftShift = N.getOpcode() == ISD::SHL;
  if ((LeftShift || N.getOpcode() == ISD::SRL) &&
      isa<ConstantSDNode>(N.getOperand(1))) {
    SDValue N0 = N.getOperand(0);
    if (N0.getOpcode() == ISD::AND && N0.hasOneUse() &&
        isa<ConstantSDNode>(N0.getOperand(1))) {
      uint64_t Mask = N0.getConstantOperandVal(1);
      if (isShiftedMask_64(Mask)) {
        unsigned C1 = N.getConstantOperandVal(1);
        unsigned XLen = Subtarget->getXLen();
        unsigned Leading = XLen - llvm::bit_width(Mask);
        unsigned Trailing = countTrailingZeros(Mask);
        // Look for (shl (and X, Mask), C1) where Mask has 32 leading zeros and
        // C3 trailing zeros. If C1+C3==ShAmt we can use SRLIW+SHXADD.
        if (LeftShift && Leading == 32 && Trailing > 0 &&
            (Trailing + C1) == ShAmt) {
          SDLoc DL(N);
          EVT VT = N.getValueType();
          Val = SDValue(CurDAG->getMachineNode(
                            RISCV::SRLIW, DL, VT, N0.getOperand(0),
                            CurDAG->getTargetConstant(Trailing, DL, VT)),
                        0);
          return true;
        }
        // Look for (srl (and X, Mask), C1) where Mask has 32 leading zeros and
        // C3 trailing zeros. If C3-C1==ShAmt we can use SRLIW+SHXADD.
        if (!LeftShift && Leading == 32 && Trailing > C1 &&
            (Trailing - C1) == ShAmt) {
          SDLoc DL(N);
          EVT VT = N.getValueType();
          Val = SDValue(CurDAG->getMachineNode(
                            RISCV::SRLIW, DL, VT, N0.getOperand(0),
                            CurDAG->getTargetConstant(Trailing, DL, VT)),
                        0);
          return true;
        }
      }
    }
  }

  return false;
}

/// Look for various patterns that can be done with a SHL that can be folded
/// into a SHXADD_UW. \p ShAmt contains 1, 2, or 3 and is set based on which
/// SHXADD_UW we are trying to match.
bool RISCVDAGToDAGISel::selectSHXADD_UWOp(SDValue N, unsigned ShAmt,
                                          SDValue &Val) {
  if (N.getOpcode() == ISD::AND && isa<ConstantSDNode>(N.getOperand(1)) &&
      N.hasOneUse()) {
    SDValue N0 = N.getOperand(0);
    if (N0.getOpcode() == ISD::SHL && isa<ConstantSDNode>(N0.getOperand(1)) &&
        N0.hasOneUse()) {
      uint64_t Mask = N.getConstantOperandVal(1);
      unsigned C2 = N0.getConstantOperandVal(1);

      Mask &= maskTrailingZeros<uint64_t>(C2);

      // Look for (and (shl y, c2), c1) where c1 is a shifted mask with
      // 32-ShAmt leading zeros and c2 trailing zeros. We can use SLLI by
      // c2-ShAmt followed by SHXADD_UW with ShAmt for the X amount.
      if (isShiftedMask_64(Mask)) {
        unsigned Leading = countLeadingZeros(Mask);
        unsigned Trailing = countTrailingZeros(Mask);
        if (Leading == 32 - ShAmt && Trailing == C2 && Trailing > ShAmt) {
          SDLoc DL(N);
          EVT VT = N.getValueType();
          Val = SDValue(CurDAG->getMachineNode(
                            RISCV::SLLI, DL, VT, N0.getOperand(0),
                            CurDAG->getTargetConstant(C2 - ShAmt, DL, VT)),
                        0);
          return true;
        }
      }
    }
  }

  return false;
}

// Return true if all users of this SDNode* only consume the lower \p Bits.
// This can be used to form W instructions for add/sub/mul/shl even when the
// root isn't a sext_inreg. This can allow the ADDW/SUBW/MULW/SLLIW to CSE if
// SimplifyDemandedBits has made it so some users see a sext_inreg and some
// don't. The sext_inreg+add/sub/mul/shl will get selected, but still leave
// the add/sub/mul/shl to become non-W instructions. By checking the users we
// may be able to use a W instruction and CSE with the other instruction if
// this has happened. We could try to detect that the CSE opportunity exists
// before doing this, but that would be more complicated.
bool RISCVDAGToDAGISel::hasAllNBitUsers(SDNode *Node, unsigned Bits,
                                        const unsigned Depth) const {
  assert((Node->getOpcode() == ISD::ADD || Node->getOpcode() == ISD::SUB ||
          Node->getOpcode() == ISD::MUL || Node->getOpcode() == ISD::SHL ||
          Node->getOpcode() == ISD::SRL || Node->getOpcode() == ISD::AND ||
          Node->getOpcode() == ISD::OR || Node->getOpcode() == ISD::XOR ||
          Node->getOpcode() == ISD::SIGN_EXTEND_INREG ||
          isa<ConstantSDNode>(Node) || Depth != 0) &&
         "Unexpected opcode");

  if (Depth >= SelectionDAG::MaxRecursionDepth)
    return false;

  for (auto UI = Node->use_begin(), UE = Node->use_end(); UI != UE; ++UI) {
    SDNode *User = *UI;
    // Users of this node should have already been instruction selected
    if (!User->isMachineOpcode())
      return false;

    // TODO: Add more opcodes?
    switch (User->getMachineOpcode()) {
    default:
      return false;
    case RISCV::ADDW:
    case RISCV::ADDIW:
    case RISCV::SUBW:
    case RISCV::MULW:
    case RISCV::SLLW:
    case RISCV::SLLIW:
    case RISCV::SRAW:
    case RISCV::SRAIW:
    case RISCV::SRLW:
    case RISCV::SRLIW:
    case RISCV::DIVW:
    case RISCV::DIVUW:
    case RISCV::REMW:
    case RISCV::REMUW:
    case RISCV::ROLW:
    case RISCV::RORW:
    case RISCV::RORIW:
    case RISCV::CLZW:
    case RISCV::CTZW:
    case RISCV::CPOPW:
    case RISCV::SLLI_UW:
    case RISCV::FMV_W_X:
    case RISCV::FCVT_H_W:
    case RISCV::FCVT_H_WU:
    case RISCV::FCVT_S_W:
    case RISCV::FCVT_S_WU:
    case RISCV::FCVT_D_W:
    case RISCV::FCVT_D_WU:
      if (Bits < 32)
        return false;
      break;
    case RISCV::SLL:
    case RISCV::SRA:
    case RISCV::SRL:
    case RISCV::ROL:
    case RISCV::ROR:
    case RISCV::BSET:
    case RISCV::BCLR:
    case RISCV::BINV:
      // Shift amount operands only use log2(Xlen) bits.
      if (UI.getOperandNo() != 1 || Bits < Log2_32(Subtarget->getXLen()))
        return false;
      break;
    case RISCV::SLLI:
      // SLLI only uses the lower (XLen - ShAmt) bits.
      if (Bits < Subtarget->getXLen() - User->getConstantOperandVal(1))
        return false;
      break;
    case RISCV::ANDI:
      if (Bits >= (unsigned)llvm::bit_width(User->getConstantOperandVal(1)))
        break;
      goto RecCheck;
    case RISCV::ORI: {
      uint64_t Imm = cast<ConstantSDNode>(User->getOperand(1))->getSExtValue();
      if (Bits >= (unsigned)llvm::bit_width<uint64_t>(~Imm))
        break;
      [[fallthrough]];
    }
    case RISCV::AND:
    case RISCV::OR:
    case RISCV::XOR:
    case RISCV::XORI:
    case RISCV::ANDN:
    case RISCV::ORN:
    case RISCV::XNOR:
    case RISCV::SH1ADD:
    case RISCV::SH2ADD:
    case RISCV::SH3ADD:
    RecCheck:
      if (hasAllNBitUsers(User, Bits, Depth + 1))
        break;
      return false;
    case RISCV::SRLI: {
      unsigned ShAmt = User->getConstantOperandVal(1);
      // If we are shifting right by less than Bits, and users don't demand any
      // bits that were shifted into [Bits-1:0], then we can consider this as an
      // N-Bit user.
      if (Bits > ShAmt && hasAllNBitUsers(User, Bits - ShAmt, Depth + 1))
        break;
      return false;
    }
    case RISCV::SEXT_B:
    case RISCV::PACKH:
      if (Bits < 8)
        return false;
      break;
    case RISCV::SEXT_H:
    case RISCV::FMV_H_X:
    case RISCV::ZEXT_H_RV32:
    case RISCV::ZEXT_H_RV64:
    case RISCV::PACKW:
      if (Bits < 16)
        return false;
      break;
    case RISCV::PACK:
      if (Bits < (Subtarget->getXLen() / 2))
        return false;
      break;
    case RISCV::ADD_UW:
    case RISCV::SH1ADD_UW:
    case RISCV::SH2ADD_UW:
    case RISCV::SH3ADD_UW:
      // The first operand to add.uw/shXadd.uw is implicitly zero extended from
      // 32 bits.
      if (UI.getOperandNo() != 0 || Bits < 32)
        return false;
      break;
    case RISCV::SB:
      if (UI.getOperandNo() != 0 || Bits < 8)
        return false;
      break;
    case RISCV::SH:
      if (UI.getOperandNo() != 0 || Bits < 16)
        return false;
      break;
    case RISCV::SW:
      if (UI.getOperandNo() != 0 || Bits < 32)
        return false;
      break;
    }
  }

  return true;
}

// Select VL as a 5 bit immediate or a value that will become a register. This
// allows us to choose betwen VSETIVLI or VSETVLI later.
bool RISCVDAGToDAGISel::selectVLOp(SDValue N, SDValue &VL) {
  auto *C = dyn_cast<ConstantSDNode>(N);
  if (C && isUInt<5>(C->getZExtValue())) {
    VL = CurDAG->getTargetConstant(C->getZExtValue(), SDLoc(N),
                                   N->getValueType(0));
  } else if (C && C->isAllOnesValue()) {
    // Treat all ones as VLMax.
    VL = CurDAG->getTargetConstant(RISCV::VLMaxSentinel, SDLoc(N),
                                   N->getValueType(0));
  } else if (isa<RegisterSDNode>(N) &&
             cast<RegisterSDNode>(N)->getReg() == RISCV::X0) {
    // All our VL operands use an operand that allows GPRNoX0 or an immediate
    // as the register class. Convert X0 to a special immediate to pass the
    // MachineVerifier. This is recognized specially by the vsetvli insertion
    // pass.
    VL = CurDAG->getTargetConstant(RISCV::VLMaxSentinel, SDLoc(N),
                                   N->getValueType(0));
  } else {
    VL = N;
  }

  return true;
}

bool RISCVDAGToDAGISel::selectVSplat(SDValue N, SDValue &SplatVal) {
  if (N.getOpcode() != RISCVISD::VMV_V_X_VL || !N.getOperand(0).isUndef())
    return false;
  assert(N.getNumOperands() == 3 && "Unexpected number of operands");
  SplatVal = N.getOperand(1);
  return true;
}

using ValidateFn = bool (*)(int64_t);

static bool selectVSplatSimmHelper(SDValue N, SDValue &SplatVal,
                                   SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget,
                                   ValidateFn ValidateImm) {
  if (N.getOpcode() != RISCVISD::VMV_V_X_VL || !N.getOperand(0).isUndef() ||
      !isa<ConstantSDNode>(N.getOperand(1)))
    return false;
  assert(N.getNumOperands() == 3 && "Unexpected number of operands");

  int64_t SplatImm =
      cast<ConstantSDNode>(N.getOperand(1))->getSExtValue();

  // The semantics of RISCVISD::VMV_V_X_VL is that when the operand
  // type is wider than the resulting vector element type: an implicit
  // truncation first takes place. Therefore, perform a manual
  // truncation/sign-extension in order to ignore any truncated bits and catch
  // any zero-extended immediate.
  // For example, we wish to match (i8 -1) -> (XLenVT 255) as a simm5 by first
  // sign-extending to (XLenVT -1).
  MVT XLenVT = Subtarget.getXLenVT();
  assert(XLenVT == N.getOperand(1).getSimpleValueType() &&
         "Unexpected splat operand type");
  MVT EltVT = N.getSimpleValueType().getVectorElementType();
  if (EltVT.bitsLT(XLenVT))
    SplatImm = SignExtend64(SplatImm, EltVT.getSizeInBits());

  if (!ValidateImm(SplatImm))
    return false;

  SplatVal = DAG.getTargetConstant(SplatImm, SDLoc(N), XLenVT);
  return true;
}

bool RISCVDAGToDAGISel::selectVSplatSimm5(SDValue N, SDValue &SplatVal) {
  return selectVSplatSimmHelper(N, SplatVal, *CurDAG, *Subtarget,
                                [](int64_t Imm) { return isInt<5>(Imm); });
}

bool RISCVDAGToDAGISel::selectVSplatSimm5Plus1(SDValue N, SDValue &SplatVal) {
  return selectVSplatSimmHelper(
      N, SplatVal, *CurDAG, *Subtarget,
      [](int64_t Imm) { return (isInt<5>(Imm) && Imm != -16) || Imm == 16; });
}

bool RISCVDAGToDAGISel::selectVSplatSimm5Plus1NonZero(SDValue N,
                                                      SDValue &SplatVal) {
  return selectVSplatSimmHelper(
      N, SplatVal, *CurDAG, *Subtarget, [](int64_t Imm) {
        return Imm != 0 && ((isInt<5>(Imm) && Imm != -16) || Imm == 16);
      });
}

bool RISCVDAGToDAGISel::selectVSplatUimm5(SDValue N, SDValue &SplatVal) {
  if (N.getOpcode() != RISCVISD::VMV_V_X_VL || !N.getOperand(0).isUndef() ||
      !isa<ConstantSDNode>(N.getOperand(1)))
    return false;

  int64_t SplatImm =
      cast<ConstantSDNode>(N.getOperand(1))->getSExtValue();

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

// Try to remove sext.w if the input is a W instruction or can be made into
// a W instruction cheaply.
bool RISCVDAGToDAGISel::doPeepholeSExtW(SDNode *N) {
  // Look for the sext.w pattern, addiw rd, rs1, 0.
  if (N->getMachineOpcode() != RISCV::ADDIW ||
      !isNullConstant(N->getOperand(1)))
    return false;

  SDValue N0 = N->getOperand(0);
  if (!N0.isMachineOpcode())
    return false;

  switch (N0.getMachineOpcode()) {
  default:
    break;
  case RISCV::ADD:
  case RISCV::ADDI:
  case RISCV::SUB:
  case RISCV::MUL:
  case RISCV::SLLI: {
    // Convert sext.w+add/sub/mul to their W instructions. This will create
    // a new independent instruction. This improves latency.
    unsigned Opc;
    switch (N0.getMachineOpcode()) {
    default:
      llvm_unreachable("Unexpected opcode!");
    case RISCV::ADD:  Opc = RISCV::ADDW;  break;
    case RISCV::ADDI: Opc = RISCV::ADDIW; break;
    case RISCV::SUB:  Opc = RISCV::SUBW;  break;
    case RISCV::MUL:  Opc = RISCV::MULW;  break;
    case RISCV::SLLI: Opc = RISCV::SLLIW; break;
    }

    SDValue N00 = N0.getOperand(0);
    SDValue N01 = N0.getOperand(1);

    // Shift amount needs to be uimm5.
    if (N0.getMachineOpcode() == RISCV::SLLI &&
        !isUInt<5>(cast<ConstantSDNode>(N01)->getSExtValue()))
      break;

    SDNode *Result =
        CurDAG->getMachineNode(Opc, SDLoc(N), N->getValueType(0),
                               N00, N01);
    ReplaceUses(N, Result);
    return true;
  }
  case RISCV::ADDW:
  case RISCV::ADDIW:
  case RISCV::SUBW:
  case RISCV::MULW:
  case RISCV::SLLIW:
  case RISCV::PACKW:
    // Result is already sign extended just remove the sext.w.
    // NOTE: We only handle the nodes that are selected with hasAllWUsers.
    ReplaceUses(N, N0.getNode());
    return true;
  }

  return false;
}

// Return true if we can make sure mask of N is all-ones mask.
static bool usesAllOnesMask(SDNode *N, unsigned MaskOpIdx) {
  // Check that we're using V0 as a mask register.
  if (!isa<RegisterSDNode>(N->getOperand(MaskOpIdx)) ||
      cast<RegisterSDNode>(N->getOperand(MaskOpIdx))->getReg() != RISCV::V0)
    return false;

  // The glued user defines V0.
  const auto *Glued = N->getGluedNode();

  if (!Glued || Glued->getOpcode() != ISD::CopyToReg)
    return false;

  // Check that we're defining V0 as a mask register.
  if (!isa<RegisterSDNode>(Glued->getOperand(1)) ||
      cast<RegisterSDNode>(Glued->getOperand(1))->getReg() != RISCV::V0)
    return false;

  // Check the instruction defining V0; it needs to be a VMSET pseudo.
  SDValue MaskSetter = Glued->getOperand(2);

  const auto IsVMSet = [](unsigned Opc) {
    return Opc == RISCV::PseudoVMSET_M_B1 || Opc == RISCV::PseudoVMSET_M_B16 ||
           Opc == RISCV::PseudoVMSET_M_B2 || Opc == RISCV::PseudoVMSET_M_B32 ||
           Opc == RISCV::PseudoVMSET_M_B4 || Opc == RISCV::PseudoVMSET_M_B64 ||
           Opc == RISCV::PseudoVMSET_M_B8;
  };

  // TODO: Check that the VMSET is the expected bitwidth? The pseudo has
  // undefined behaviour if it's the wrong bitwidth, so we could choose to
  // assume that it's all-ones? Same applies to its VL.
  return MaskSetter->isMachineOpcode() &&
         IsVMSet(MaskSetter.getMachineOpcode());
}

// Optimize masked RVV pseudo instructions with a known all-ones mask to their
// corresponding "unmasked" pseudo versions. The mask we're interested in will
// take the form of a V0 physical register operand, with a glued
// register-setting instruction.
bool RISCVDAGToDAGISel::doPeepholeMaskedRVV(SDNode *N) {
  const RISCV::RISCVMaskedPseudoInfo *I =
      RISCV::getMaskedPseudoInfo(N->getMachineOpcode());
  if (!I)
    return false;

  unsigned MaskOpIdx = I->MaskOpIdx;

  if (!usesAllOnesMask(N, MaskOpIdx))
    return false;

  // Retrieve the tail policy operand index, if any.
  std::optional<unsigned> TailPolicyOpIdx;
  const RISCVInstrInfo &TII = *Subtarget->getInstrInfo();
  const MCInstrDesc &MaskedMCID = TII.get(N->getMachineOpcode());

  bool IsTA = true;
  if (RISCVII::hasVecPolicyOp(MaskedMCID.TSFlags)) {
    TailPolicyOpIdx = getVecPolicyOpIdx(N, MaskedMCID);
    if (!(N->getConstantOperandVal(*TailPolicyOpIdx) &
          RISCVII::TAIL_AGNOSTIC)) {
      // Keep the true-masked instruction when there is no unmasked TU
      // instruction
      if (I->UnmaskedTUPseudo == I->MaskedPseudo && !N->getOperand(0).isUndef())
        return false;
      // We can't use TA if the tie-operand is not IMPLICIT_DEF
      if (!N->getOperand(0).isUndef())
        IsTA = false;
    }
  }

  unsigned Opc = IsTA ? I->UnmaskedPseudo : I->UnmaskedTUPseudo;

  // Check that we're dropping the mask operand and any policy operand
  // when we transform to this unmasked pseudo. Additionally, if this insturtion
  // is tail agnostic, the unmasked instruction should not have a merge op.
  uint64_t TSFlags = TII.get(Opc).TSFlags;
  assert((IsTA != RISCVII::hasMergeOp(TSFlags)) &&
         RISCVII::hasDummyMaskOp(TSFlags) &&
         !RISCVII::hasVecPolicyOp(TSFlags) &&
         "Unexpected pseudo to transform to");
  (void)TSFlags;

  SmallVector<SDValue, 8> Ops;
  // Skip the merge operand at index 0 if IsTA
  for (unsigned I = IsTA, E = N->getNumOperands(); I != E; I++) {
    // Skip the mask, the policy, and the Glue.
    SDValue Op = N->getOperand(I);
    if (I == MaskOpIdx || I == TailPolicyOpIdx ||
        Op.getValueType() == MVT::Glue)
      continue;
    Ops.push_back(Op);
  }

  // Transitively apply any node glued to our new node.
  const auto *Glued = N->getGluedNode();
  if (auto *TGlued = Glued->getGluedNode())
    Ops.push_back(SDValue(TGlued, TGlued->getNumValues() - 1));

  SDNode *Result = CurDAG->getMachineNode(Opc, SDLoc(N), N->getVTList(), Ops);
  Result->setFlags(N->getFlags());
  ReplaceUses(N, Result);

  return true;
}

// Try to fold VMERGE_VVM with unmasked intrinsic to masked intrinsic. The
// peephole only deals with VMERGE_VVM which is TU and has false operand same as
// its true operand now. E.g. (VMERGE_VVM_M1_TU False, False, (VADD_M1 ...),
// ...) -> (VADD_VV_M1_MASK)
bool RISCVDAGToDAGISel::performCombineVMergeAndVOps(SDNode *N, bool IsTA) {
  unsigned Offset = IsTA ? 0 : 1;
  uint64_t Policy = IsTA ? RISCVII::TAIL_AGNOSTIC : /*TUMU*/ 0;

  SDValue False = N->getOperand(0 + Offset);
  SDValue True = N->getOperand(1 + Offset);
  SDValue Mask = N->getOperand(2 + Offset);
  SDValue VL = N->getOperand(3 + Offset);

  assert(True.getResNo() == 0 &&
         "Expect True is the first output of an instruction.");

  // Need N is the exactly one using True.
  if (!True.hasOneUse())
    return false;

  if (!True.isMachineOpcode())
    return false;

  unsigned TrueOpc = True.getMachineOpcode();

  // Skip if True has merge operand.
  // TODO: Deal with True having same merge operand with N.
  if (RISCVII::hasMergeOp(TII->get(TrueOpc).TSFlags))
    return false;

  // Skip if True has side effect.
  // TODO: Support velff and vlsegff.
  if (TII->get(TrueOpc).hasUnmodeledSideEffects())
    return false;

  // Only deal with True when True is unmasked intrinsic now.
  const RISCV::RISCVMaskedPseudoInfo *Info =
      RISCV::lookupMaskedIntrinsicByUnmaskedTA(TrueOpc);

  if (!Info)
    return false;

  // The last operand of unmasked intrinsic should be sew or chain.
  bool HasChainOp =
      True.getOperand(True.getNumOperands() - 1).getValueType() == MVT::Other;

  if (HasChainOp) {
    // Avoid creating cycles in the DAG. We must ensure that none of the other
    // operands depend on True through it's Chain.
    SmallVector<const SDNode *, 4> LoopWorklist;
    SmallPtrSet<const SDNode *, 16> Visited;
    LoopWorklist.push_back(False.getNode());
    LoopWorklist.push_back(Mask.getNode());
    LoopWorklist.push_back(VL.getNode());
    if (SDNode *Glued = N->getGluedNode())
      LoopWorklist.push_back(Glued);
    if (SDNode::hasPredecessorHelper(True.getNode(), Visited, LoopWorklist))
      return false;
  }

  // Need True has same VL with N.
  unsigned TrueVLIndex = True.getNumOperands() - HasChainOp - 2;
  SDValue TrueVL = True.getOperand(TrueVLIndex);

  auto IsNoFPExcept = [this](SDValue N) {
    return !this->mayRaiseFPException(N.getNode()) ||
           N->getFlags().hasNoFPExcept();
  };

  // Allow the peephole for non-exception True with VLMAX vector length, since
  // all the values after VL of N are dependent on Merge. VLMAX should be
  // lowered to (XLenVT -1).
  if (TrueVL != VL && !(IsNoFPExcept(True) && isAllOnesConstant(TrueVL)))
    return false;

  SDLoc DL(N);
  unsigned MaskedOpc = Info->MaskedPseudo;
  assert(RISCVII::hasVecPolicyOp(TII->get(MaskedOpc).TSFlags) &&
         "Expected instructions with mask have policy operand.");
  assert(RISCVII::hasMergeOp(TII->get(MaskedOpc).TSFlags) &&
         "Expected instructions with mask have merge operand.");

  SmallVector<SDValue, 8> Ops;
  Ops.push_back(False);
  Ops.append(True->op_begin(), True->op_begin() + TrueVLIndex);
  Ops.append({Mask, VL, /* SEW */ True.getOperand(TrueVLIndex + 1)});
  Ops.push_back(CurDAG->getTargetConstant(Policy, DL, Subtarget->getXLenVT()));

  // Result node should have chain operand of True.
  if (HasChainOp)
    Ops.push_back(True.getOperand(True.getNumOperands() - 1));

  // Result node should take over glued node of N.
  if (N->getGluedNode())
    Ops.push_back(N->getOperand(N->getNumOperands() - 1));

  SDNode *Result =
      CurDAG->getMachineNode(MaskedOpc, DL, True->getVTList(), Ops);
  Result->setFlags(True->getFlags());

  // Replace vmerge.vvm node by Result.
  ReplaceUses(SDValue(N, 0), SDValue(Result, 0));

  // Replace another value of True. E.g. chain and VL.
  for (unsigned Idx = 1; Idx < True->getNumValues(); ++Idx)
    ReplaceUses(True.getValue(Idx), SDValue(Result, Idx));

  // Try to transform Result to unmasked intrinsic.
  doPeepholeMaskedRVV(Result);
  return true;
}

// Transform (VMERGE_VVM_<LMUL>_TU false, false, true, allones, vl, sew) to
// (VADD_VI_<LMUL>_TU false, true, 0, vl, sew). It may decrease uses of VMSET.
bool RISCVDAGToDAGISel::performVMergeToVAdd(SDNode *N) {
  unsigned NewOpc;
  switch (N->getMachineOpcode()) {
  default:
    llvm_unreachable("Expected VMERGE_VVM_<LMUL>_TU instruction.");
  case RISCV::PseudoVMERGE_VVM_MF8_TU:
    NewOpc = RISCV::PseudoVADD_VI_MF8_TU;
    break;
  case RISCV::PseudoVMERGE_VVM_MF4_TU:
    NewOpc = RISCV::PseudoVADD_VI_MF4_TU;
    break;
  case RISCV::PseudoVMERGE_VVM_MF2_TU:
    NewOpc = RISCV::PseudoVADD_VI_MF2_TU;
    break;
  case RISCV::PseudoVMERGE_VVM_M1_TU:
    NewOpc = RISCV::PseudoVADD_VI_M1_TU;
    break;
  case RISCV::PseudoVMERGE_VVM_M2_TU:
    NewOpc = RISCV::PseudoVADD_VI_M2_TU;
    break;
  case RISCV::PseudoVMERGE_VVM_M4_TU:
    NewOpc = RISCV::PseudoVADD_VI_M4_TU;
    break;
  case RISCV::PseudoVMERGE_VVM_M8_TU:
    NewOpc = RISCV::PseudoVADD_VI_M8_TU;
    break;
  }

  if (!usesAllOnesMask(N, /* MaskOpIdx */ 3))
    return false;

  SDLoc DL(N);
  EVT VT = N->getValueType(0);
  SDValue Ops[] = {N->getOperand(1), N->getOperand(2),
                   CurDAG->getTargetConstant(0, DL, Subtarget->getXLenVT()),
                   N->getOperand(4), N->getOperand(5)};
  SDNode *Result = CurDAG->getMachineNode(NewOpc, DL, VT, Ops);
  ReplaceUses(N, Result);
  return true;
}

bool RISCVDAGToDAGISel::doPeepholeMergeVVMFold() {
  bool MadeChange = false;
  SelectionDAG::allnodes_iterator Position = CurDAG->allnodes_end();

  while (Position != CurDAG->allnodes_begin()) {
    SDNode *N = &*--Position;
    if (N->use_empty() || !N->isMachineOpcode())
      continue;

    auto IsVMergeTU = [](unsigned Opcode) {
      return Opcode == RISCV::PseudoVMERGE_VVM_MF8_TU ||
             Opcode == RISCV::PseudoVMERGE_VVM_MF4_TU ||
             Opcode == RISCV::PseudoVMERGE_VVM_MF2_TU ||
             Opcode == RISCV::PseudoVMERGE_VVM_M1_TU ||
             Opcode == RISCV::PseudoVMERGE_VVM_M2_TU ||
             Opcode == RISCV::PseudoVMERGE_VVM_M4_TU ||
             Opcode == RISCV::PseudoVMERGE_VVM_M8_TU;
    };

    auto IsVMergeTA = [](unsigned Opcode) {
      return Opcode == RISCV::PseudoVMERGE_VVM_MF8 ||
             Opcode == RISCV::PseudoVMERGE_VVM_MF4 ||
             Opcode == RISCV::PseudoVMERGE_VVM_MF2 ||
             Opcode == RISCV::PseudoVMERGE_VVM_M1 ||
             Opcode == RISCV::PseudoVMERGE_VVM_M2 ||
             Opcode == RISCV::PseudoVMERGE_VVM_M4 ||
             Opcode == RISCV::PseudoVMERGE_VVM_M8;
    };

    unsigned Opc = N->getMachineOpcode();
    // The following optimizations require that the merge operand of N is same
    // as the false operand of N.
    if ((IsVMergeTU(Opc) && N->getOperand(0) == N->getOperand(1)) ||
        IsVMergeTA(Opc))
      MadeChange |= performCombineVMergeAndVOps(N, IsVMergeTA(Opc));
    if (IsVMergeTU(Opc) && N->getOperand(0) == N->getOperand(1))
      MadeChange |= performVMergeToVAdd(N);
  }
  return MadeChange;
}

// This pass converts a legalized DAG into a RISCV-specific DAG, ready
// for instruction scheduling.
FunctionPass *llvm::createRISCVISelDag(RISCVTargetMachine &TM,
                                       CodeGenOpt::Level OptLevel) {
  return new RISCVDAGToDAGISel(TM, OptLevel);
}

char RISCVDAGToDAGISel::ID = 0;

INITIALIZE_PASS(RISCVDAGToDAGISel, DEBUG_TYPE, PASS_NAME, false, false)
