//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VESelectionDAGInfo.h"

#define GET_SDNODE_DESC
#include "VEGenSDNodeInfo.inc"

using namespace llvm;

VESelectionDAGInfo::VESelectionDAGInfo()
    : SelectionDAGGenTargetInfo(VEGenSDNodeInfo) {}

VESelectionDAGInfo::~VESelectionDAGInfo() = default;

const char *VESelectionDAGInfo::getTargetNodeName(unsigned Opcode) const {
#define TARGET_NODE_CASE(NAME)                                                 \
  case VEISD::NAME:                                                            \
    return "VEISD::" #NAME;

  switch (static_cast<VEISD::NodeType>(Opcode)) {
    TARGET_NODE_CASE(GLOBAL_BASE_REG)
    TARGET_NODE_CASE(LEGALAVL)
  }
#undef TARGET_NODE_CASE

  return SelectionDAGGenTargetInfo::getTargetNodeName(Opcode);
}

void VESelectionDAGInfo::verifyTargetNode(const SelectionDAG &DAG,
                                          const SDNode *N) const {
  switch (N->getOpcode()) {
  case VEISD::GETSTACKTOP:
    // result #0 has invalid type; expected ch, got i64
    return;
  }

  SelectionDAGGenTargetInfo::verifyTargetNode(DAG, N);
}
