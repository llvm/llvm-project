//===- M68kSelectionDAGInfo.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "M68kSelectionDAGInfo.h"

#define GET_SDNODE_DESC
#include "M68kGenSDNodeInfo.inc"

using namespace llvm;

M68kSelectionDAGInfo::M68kSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(M68kGenSDNodeInfo) {}

M68kSelectionDAGInfo::~M68kSelectionDAGInfo() = default;

const char *M68kSelectionDAGInfo::getTargetNodeName(unsigned Opcode) const {
  switch (static_cast<M68kISD::NodeType>(Opcode)) {
  case M68kISD::TAIL_CALL:
    return "M68kISD::TAIL_CALL";
  case M68kISD::SELECT:
    return "M68kISD::SELECT";
  case M68kISD::GLOBAL_BASE_REG:
    return "M68kISD::GLOBAL_BASE_REG";
  }

  return SelectionDAGGenTargetInfo::getTargetNodeName(Opcode);
}
