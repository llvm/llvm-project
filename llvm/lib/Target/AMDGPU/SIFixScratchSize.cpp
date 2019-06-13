//===- SIFixScratchSize.cpp - resolve scratch size symbols                -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass replaces references with to the scratch size symbol with the
/// actual scratch size.  This pass should be run late, i.e. when the scratch
/// size for a given machine function is known.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

#include <set>

using namespace llvm;

#define DEBUG_TYPE "si-fix-scratch-size"

namespace {

class SIFixScratchSize : public MachineFunctionPass {
public:
  static char ID;

  SIFixScratchSize() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end anonymous namespace

INITIALIZE_PASS(SIFixScratchSize, DEBUG_TYPE,
                "SI Resolve Scratch Size Symbols",
                false, false)

char SIFixScratchSize::ID = 0;

char &llvm::SIFixScratchSizeID = SIFixScratchSize::ID;

const char *const llvm::SIScratchSizeSymbol = "___SCRATCH_SIZE";

FunctionPass *llvm::createSIFixScratchSizePass() {
  return new SIFixScratchSize;
}

bool SIFixScratchSize::runOnMachineFunction(MachineFunction &MF) {
  const MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  const uint64_t StackSize = FrameInfo.getStackSize();

  if (!StackSize)
    return false;

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      for (MachineOperand &MO: MI.operands()) {
        if (MO.isSymbol()) {
          if (MO.getSymbolName() == SIScratchSizeSymbol) {
            LLVM_DEBUG(dbgs() << "Fixing: " << MI << "\n");
            MO.ChangeToImmediate(StackSize);
            Changed = true;
          }
        }
      }
    }
  }

  return Changed;
}
