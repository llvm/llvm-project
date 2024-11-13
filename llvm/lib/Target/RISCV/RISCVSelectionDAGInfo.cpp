//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVSelectionDAGInfo.h"
#include "RISCVISelLowering.h"

using namespace llvm;

RISCVSelectionDAGInfo::~RISCVSelectionDAGInfo() = default;

bool RISCVSelectionDAGInfo::isTargetMemoryOpcode(unsigned Opcode) const {
  switch (static_cast<RISCVISD::NodeType>(Opcode)) {
  default:
    return false;
  case RISCVISD::TH_LWD:
  case RISCVISD::TH_LWUD:
  case RISCVISD::TH_LDD:
  case RISCVISD::TH_SWD:
  case RISCVISD::TH_SDD:
    return true;
  }
}

bool RISCVSelectionDAGInfo::isTargetStrictFPOpcode(unsigned Opcode) const {
  return Opcode >= RISCVISD::FIRST_STRICTFP_OPCODE &&
         Opcode <= RISCVISD::LAST_STRICTFP_OPCODE;
}
