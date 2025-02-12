//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MipsSelectionDAGInfo.h"
#include "MipsISelLowering.h"

using namespace llvm;

MipsSelectionDAGInfo::~MipsSelectionDAGInfo() = default;

bool MipsSelectionDAGInfo::isTargetMemoryOpcode(unsigned Opcode) const {
  return Opcode >= MipsISD::FIRST_MEMORY_OPCODE &&
         Opcode <= MipsISD::LAST_MEMORY_OPCODE;
}
