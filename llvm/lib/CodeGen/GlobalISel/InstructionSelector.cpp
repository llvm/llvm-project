//===- llvm/CodeGen/GlobalISel/InstructionSelector.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/CodeGen/TargetOpcodes.h"

namespace llvm {

// vtable anchor
InstructionSelector::~InstructionSelector() = default;

void InstructionSelector::renderFrameIndex(MachineInstrBuilder &MIB,
                                           const MachineInstr &MI,
                                           int OpIdx) const {
  assert(MI.getOpcode() == TargetOpcode::G_FRAME_INDEX && OpIdx == -1 &&
         "Expected G_FRAME_INDEX");
  MIB.add(MI.getOperand(1));
}

} // namespace llvm
