//===----------------------------------------------------------------------===//
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

void M68kSelectionDAGInfo::verifyTargetNode(const SelectionDAG &DAG,
                                            const SDNode *N) const {
  switch (N->getOpcode()) {
  case M68kISD::ADD:
  case M68kISD::SUBX:
    // result #1 must have type i8, but has type i32
    return;
  case M68kISD::SETCC:
    // operand #1 must have type i8, but has type i32
    return;
  }

  SelectionDAGGenTargetInfo::verifyTargetNode(DAG, N);
}

M68kSelectionDAGInfo::~M68kSelectionDAGInfo() = default;
