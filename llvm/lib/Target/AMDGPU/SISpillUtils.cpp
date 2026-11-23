//===- SISpillUtils.cpp - SI spill helper functions -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SISpillUtils.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"

using namespace llvm;

void llvm::clearDebugInfoForSpillFIs(MachineFrameInfo &MFI,
                                     MachineBasicBlock &MBB,
                                     const BitVector &SpillFIs) {
  // FIXME: The dead frame indices are replaced with a null register from the
  // debug value instructions. We should instead update it with the correct
  // register value. But not sure the register value alone is adequate to lower
  // the DIExpression. It should be worked out later.
  for (MachineInstr &MI : MBB) {
    if (!MI.isDebugValue())
      continue;

    for (MachineOperand &Op : MI.debug_operands()) {
      if (Op.isFI() && !MFI.isFixedObjectIndex(Op.getIndex()) &&
          SpillFIs[Op.getIndex()]) {
        Op.ChangeToRegister(Register(), /*isDef=*/false);
      }
    }
  }
}
