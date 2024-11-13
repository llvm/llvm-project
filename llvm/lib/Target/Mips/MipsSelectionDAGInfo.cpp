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
  switch (static_cast<MipsISD::NodeType>(Opcode)) {
  default:
    return false;
  case MipsISD::LWL:
  case MipsISD::LWR:
  case MipsISD::SWL:
  case MipsISD::SWR:
  case MipsISD::LDL:
  case MipsISD::LDR:
  case MipsISD::SDL:
  case MipsISD::SDR:
    return true;
  }
}
