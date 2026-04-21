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
    return;
  }

  SelectionDAGGenTargetInfo::verifyTargetNode(DAG, N);
}
