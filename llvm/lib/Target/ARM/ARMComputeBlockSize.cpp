//===--- ARMComputeBlockSize.cpp - Compute machine block sizes ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMBaseInstrInfo.h"
#include "ARMBasicBlockInfo.h"
#include "ARMMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include <vector>

using namespace llvm;

namespace llvm {

// mayOptimizeThumb2Instruction - Returns true if optimizeThumb2Instructions
// below may shrink MI.
static bool
mayOptimizeThumb2Instruction(const MachineInstr *MI) {
  switch(MI->getOpcode()) {
    // optimizeThumb2Instructions.
    case ARM::t2LEApcrel:
    case ARM::t2LDRpci:
    // optimizeThumb2Branches.
    case ARM::t2B:
    case ARM::t2Bcc:
    case ARM::tBcc:
    // optimizeThumb2JumpTables.
    case ARM::t2BR_JT:
    case ARM::tBR_JTr:
      return true;
  }
  return false;
}

void computeBlockSize(MachineFunction *MF, MachineBasicBlock *MBB,
                      BasicBlockInfo &BBI) {
  const ARMBaseInstrInfo *TII =
    static_cast<const ARMBaseInstrInfo *>(MF->getSubtarget().getInstrInfo());
  bool isThumb = MF->getInfo<ARMFunctionInfo>()->isThumbFunction();
  BBI.Size = 0;
  BBI.Unalign = 0;
  BBI.PostAlign = 0;

  for (MachineInstr &I : *MBB) {
    BBI.Size += TII->getInstSizeInBytes(I);
    // For inline asm, getInstSizeInBytes returns a conservative estimate.
    // The actual size may be smaller, but still a multiple of the instr size.
    if (I.isInlineAsm())
      BBI.Unalign = isThumb ? 1 : 2;
    // Also consider instructions that may be shrunk later.
    else if (isThumb && mayOptimizeThumb2Instruction(&I))
      BBI.Unalign = 1;
  }

  // tBR_JTr contains a .align 2 directive.
  if (!MBB->empty() && MBB->back().getOpcode() == ARM::tBR_JTr) {
    BBI.PostAlign = 2;
    MBB->getParent()->ensureAlignment(2);
  }
}

std::vector<BasicBlockInfo> computeAllBlockSizes(MachineFunction *MF) {
  std::vector<BasicBlockInfo> BBInfo;
  BBInfo.resize(MF->getNumBlockIDs());

  for (MachineBasicBlock &MBB : *MF)
    computeBlockSize(MF, &MBB, BBInfo[MBB.getNumber()]);

  return BBInfo;
}

} // end namespace llvm
