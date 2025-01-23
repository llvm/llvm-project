//===- SparcSelectionDAGInfo.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SparcSelectionDAGInfo.h"

#define GET_SDNODE_DESC
#include "SparcGenSDNodeInfo.inc"

using namespace llvm;

SparcSelectionDAGInfo::SparcSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(SparcGenSDNodeInfo) {}

SparcSelectionDAGInfo::~SparcSelectionDAGInfo() = default;

const char *SparcSelectionDAGInfo::getTargetNodeName(unsigned Opcode) const {
  switch (static_cast<SPISD::NodeType>(Opcode)) {
  case SPISD::GLOBAL_BASE_REG:
    return "SPISD::GLOBAL_BASE_REG";
  }

  return SelectionDAGGenTargetInfo::getTargetNodeName(Opcode);
}
