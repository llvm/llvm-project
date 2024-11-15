//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVSelectionDAGInfo.h"

#define GET_SDNODE_DESC
#include "RISCVGenSDNodeInfo.inc"

using namespace llvm;

RISCVSelectionDAGInfo::RISCVSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(RISCVGenSDNodeInfo) {}

RISCVSelectionDAGInfo::~RISCVSelectionDAGInfo() = default;

const char *RISCVSelectionDAGInfo::getTargetNodeName(unsigned Opcode) const {
#define NODE_NAME_CASE(NODE)                                                   \
  case RISCVISD::NODE:                                                         \
    return "RISCVISD::" #NODE;

  // These nodes don't have corresponding entries in *.td files yet.
  switch (static_cast<RISCVISD::NodeType>(Opcode)) {
    NODE_NAME_CASE(BuildGPRPair)
    NODE_NAME_CASE(SplitGPRPair)
    NODE_NAME_CASE(SPLAT_VECTOR_SPLIT_I64_VL)
    NODE_NAME_CASE(TUPLE_INSERT)
    NODE_NAME_CASE(TUPLE_EXTRACT)
  }
#undef NODE_NAME_CASE

  return SelectionDAGGenTargetInfo::getTargetNodeName(Opcode);
}

void RISCVSelectionDAGInfo::verifyTargetNode(const SelectionDAG &DAG,
                                             const SDNode *N) const {
  // invalid number of results; expected 2, got 1
  if (N->getOpcode() == RISCVISD::PROBED_ALLOCA)
    return;
  SelectionDAGGenTargetInfo::verifyTargetNode(DAG, N);
}
