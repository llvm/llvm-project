//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PPCSelectionDAGInfo.h"
#include "PPCISelLowering.h"

using namespace llvm;

PPCSelectionDAGInfo::~PPCSelectionDAGInfo() = default;

bool PPCSelectionDAGInfo::isTargetMemoryOpcode(unsigned Opcode) const {
  return Opcode >= PPCISD::FIRST_MEMORY_OPCODE &&
         Opcode <= PPCISD::LAST_MEMORY_OPCODE;
}

bool PPCSelectionDAGInfo::isTargetStrictFPOpcode(unsigned Opcode) const {
  switch (static_cast<PPCISD::NodeType>(Opcode)) {
  default:
    return false;
  case PPCISD::STRICT_FCTIDZ:
  case PPCISD::STRICT_FCTIWZ:
  case PPCISD::STRICT_FCTIDUZ:
  case PPCISD::STRICT_FCTIWUZ:
  case PPCISD::STRICT_FCFID:
  case PPCISD::STRICT_FCFIDU:
  case PPCISD::STRICT_FCFIDS:
  case PPCISD::STRICT_FCFIDUS:
  case PPCISD::STRICT_FADDRTZ:
    return true;
  }
}
