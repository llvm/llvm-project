//===- LoongArchSelectionDAGInfo.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LoongArchSelectionDAGInfo.h"

#define GET_SDNODE_DESC
#include "LoongArchGenSDNodeInfo.inc"

using namespace llvm;

LoongArchSelectionDAGInfo::LoongArchSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(LoongArchGenSDNodeInfo) {}

LoongArchSelectionDAGInfo::~LoongArchSelectionDAGInfo() = default;

const char *
LoongArchSelectionDAGInfo::getTargetNodeName(unsigned Opcode) const {
  switch (static_cast<LoongArchISD::NodeType>(Opcode)) {
  case LoongArchISD::VSHUF4I:
    return "LoongArchISD::VSHUF4I";
  }

  return SelectionDAGGenTargetInfo::getTargetNodeName(Opcode);
}

void LoongArchSelectionDAGInfo::verifyTargetNode(const SelectionDAG &DAG,
                                                 const SDNode *N) const {
  if (N->getOpcode() == LoongArchISD::VLDREPL) {
    // invalid number of operands; expected 2, got 3
    return;
  }
  SelectionDAGGenTargetInfo::verifyTargetNode(DAG, N);
}
