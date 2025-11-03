//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NVPTXSelectionDAGInfo.h"
#include "NVPTXISelLowering.h"

using namespace llvm;

NVPTXSelectionDAGInfo::~NVPTXSelectionDAGInfo() = default;

bool NVPTXSelectionDAGInfo::isTargetMemoryOpcode(unsigned Opcode) const {
  return Opcode >= NVPTXISD::FIRST_MEMORY_OPCODE &&
         Opcode <= NVPTXISD::LAST_MEMORY_OPCODE;
}
