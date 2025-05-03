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

RISCVSelectionDAGInfo::RISCVSelectionDAGInfo() : SelectionDAGGenTargetInfo(RISCVGenSDNodeInfo) {}

RISCVSelectionDAGInfo::~RISCVSelectionDAGInfo() = default;

void RISCVSelectionDAGInfo::verifyTargetNode(const SelectionDAG &DAG, const SDNode *N) const {
  switch (N->getOpcode()) {
  case RISCVISD::PROBED_ALLOCA:
    // FIXME: Current examples do not match the SDTypeProfile.
    // They get "invalid number of results; expected 2, got 1"
    return;
  }
  SelectionDAGGenTargetInfo::verifyTargetNode(DAG, N);
}
