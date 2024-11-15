//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MipsSelectionDAGInfo.h"

#define GET_SDNODE_DESC
#include "MipsGenSDNodeInfo.inc"

using namespace llvm;

MipsSelectionDAGInfo::MipsSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(MipsGenSDNodeInfo) {}

MipsSelectionDAGInfo::~MipsSelectionDAGInfo() = default;

const char *MipsSelectionDAGInfo::getTargetNodeName(unsigned Opcode) const {
  // These nodes don't have corresponding entries in *.td files yet.
  switch (static_cast<MipsISD::NodeType>(Opcode)) {
    // clang-format off
  case MipsISD::FAbs:              return "MipsISD::FAbs";
  case MipsISD::DynAlloc:          return "MipsISD::DynAlloc";
  case MipsISD::DOUBLE_SELECT_I:   return "MipsISD::DOUBLE_SELECT_I";
  case MipsISD::DOUBLE_SELECT_I64: return "MipsISD::DOUBLE_SELECT_I64";
    // clang-format on
  }

  return SelectionDAGGenTargetInfo::getTargetNodeName(Opcode);
}

void MipsSelectionDAGInfo::verifyTargetNode(const SelectionDAG &DAG,
                                            const SDNode *N) const {
  switch (N->getOpcode()) {
  default:
    break;
  case MipsISD::ERet:
    // invalid number of operands; expected at most 2, got 3
  case MipsISD::DPAQX_SA_W_PH:
  case MipsISD::DPAQX_S_W_PH:
  case MipsISD::DPAQ_S_W_PH:
  case MipsISD::DPAX_W_PH:
  case MipsISD::DPA_W_PH:
  case MipsISD::DPSQX_SA_W_PH:
  case MipsISD::DPSQX_S_W_PH:
  case MipsISD::DPSQ_S_W_PH:
  case MipsISD::DPSX_W_PH:
  case MipsISD::DPS_W_PH:
  case MipsISD::MAQ_SA_W_PHL:
  case MipsISD::MAQ_SA_W_PHR:
  case MipsISD::MAQ_S_W_PHL:
  case MipsISD::MAQ_S_W_PHR:
  case MipsISD::MULSAQ_S_W_PH:
  case MipsISD::MULSA_W_PH:
    // operand #0 must have type i32, but has type v2i16
  case MipsISD::DPAU_H_QBL:
  case MipsISD::DPAU_H_QBR:
  case MipsISD::DPSU_H_QBL:
  case MipsISD::DPSU_H_QBR:
    // operand #0 must have type i32, but has type v4i8
    return;
  }

  SelectionDAGGenTargetInfo::verifyTargetNode(DAG, N);
}
