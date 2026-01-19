//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PPCSelectionDAGInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"

#define GET_SDNODE_DESC
#include "PPCGenSDNodeInfo.inc"

using namespace llvm;

PPCSelectionDAGInfo::PPCSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(PPCGenSDNodeInfo) {}

PPCSelectionDAGInfo::~PPCSelectionDAGInfo() = default;

const char *PPCSelectionDAGInfo::getTargetNodeName(unsigned Opcode) const {
  switch (static_cast<PPCISD::NodeType>(Opcode)) {
  case PPCISD::GlobalBaseReg:
    return "PPCISD::GlobalBaseReg";
  case PPCISD::SRA_ADDZE:
    return "PPCISD::SRA_ADDZE";
  case PPCISD::READ_TIME_BASE:
    return "PPCISD::READ_TIME_BASE";
  case PPCISD::MFOCRF:
    return "PPCISD::MFOCRF";
  case PPCISD::ANDI_rec_1_EQ_BIT:
    return "PPCISD::ANDI_rec_1_EQ_BIT";
  case PPCISD::ANDI_rec_1_GT_BIT:
    return "PPCISD::ANDI_rec_1_GT_BIT";
  case PPCISD::BDNZ:
    return "PPCISD::BDNZ";
  case PPCISD::BDZ:
    return "PPCISD::BDZ";
  case PPCISD::PPC32_PICGOT:
    return "PPCISD::PPC32_PICGOT";
  case PPCISD::VADD_SPLAT:
    return "PPCISD::VADD_SPLAT";
  }

  return SelectionDAGGenTargetInfo::getTargetNodeName(Opcode);
}

void PPCSelectionDAGInfo::verifyTargetNode(const SelectionDAG &DAG,
                                           const SDNode *N) const {
  switch (N->getOpcode()) {
  default:
    break;
  case PPCISD::DYNAREAOFFSET:
    // invalid number of results; expected 2, got 1
  case PPCISD::TOC_ENTRY:
    // invalid number of results; expected 1, got 2
  case PPCISD::STORE_COND:
    // invalid number of results; expected 2, got 3
  case PPCISD::LD_SPLAT:
  case PPCISD::SEXT_LD_SPLAT:
  case PPCISD::ZEXT_LD_SPLAT:
    // invalid number of operands; expected 2, got 3
  case PPCISD::ST_VSR_SCAL_INT:
    // invalid number of operands; expected 4, got 5
  case PPCISD::XXPERM:
    // operand #1 must have type v2f64, but has type v16i8
  case PPCISD::ACC_BUILD:
    // operand #3 must have type v4i32, but has type v16i8
  case PPCISD::PAIR_BUILD:
    // operand #1 must have type v4i32, but has type v16i8
    return;
  }

  SelectionDAGGenTargetInfo::verifyTargetNode(DAG, N);
}

std::pair<SDValue, SDValue> PPCSelectionDAGInfo::EmitTargetCodeForMemcmp(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Op1, SDValue Op2,
    SDValue Op3, const CallInst *CI) const {
  return DAG.getMemcmp(Chain, dl, Op1, Op2, Op3, CI);
}

std::pair<SDValue, SDValue> PPCSelectionDAGInfo::EmitTargetCodeForStrcpy(
    SelectionDAG &DAG, const SDLoc &DL, SDValue Chain, SDValue Dest,
    SDValue Src, MachinePointerInfo DestPtrInfo, MachinePointerInfo SrcPtrInfo,
    bool isStpcpy, const CallInst *CI) const {
  if (isStpcpy)
    return SelectionDAGTargetInfo::EmitTargetCodeForStrcpy(
        DAG, DL, Chain, Dest, Src, DestPtrInfo, SrcPtrInfo, isStpcpy, CI);
  return DAG.getStrcpy(Chain, DL, Dest, Src, CI);
}

std::pair<SDValue, SDValue>
PPCSelectionDAGInfo::EmitTargetCodeForStrlen(SelectionDAG &DAG, const SDLoc &DL,
                                             SDValue Chain, SDValue Src,
                                             const CallInst *CI) const {
  return DAG.getStrlen(Chain, DL, Src, CI);
}

std::pair<SDValue, SDValue> PPCSelectionDAGInfo::EmitTargetCodeForStrstr(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Op1, SDValue Op2,
    const CallInst *CI) const {
  return DAG.getStrstr(Chain, dl, Op1, Op2, CI);
}
