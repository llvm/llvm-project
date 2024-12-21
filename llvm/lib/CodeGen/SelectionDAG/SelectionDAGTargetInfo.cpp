//===- SelectionDAGTargetInfo.cpp - SelectionDAG Info ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the SelectionDAGTargetInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

using namespace llvm;

SelectionDAGTargetInfo::~SelectionDAGTargetInfo() = default;

bool SelectionDAGTargetInfo::mayRaiseFPException(unsigned Opcode) const {
  // FIXME: All target memory opcodes are currently automatically considered
  //  to possibly raise FP exceptions. See rev. 63336795.
  return isTargetStrictFPOpcode(Opcode) || isTargetMemoryOpcode(Opcode);
}
