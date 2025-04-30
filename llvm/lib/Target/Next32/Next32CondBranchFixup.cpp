//===-- Next32CondBranchFixup.cpp - use or replace LEA instructions ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Replace all BR_CC/BR or fall-through into BR_CC/BR explicit instructions
//
//===----------------------------------------------------------------------===//

#include "Next32.h"
#include "Next32InstrInfo.h"
#include "Next32PassTrace.h"
#include "Next32Subtarget.h"
#include "TargetInfo/Next32BaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"

using namespace llvm;

namespace llvm {
void initializeNext32CondBranchFixupPassPass(PassRegistry &);
}

#define NEXT32CONDBRANCHFIXUP_DESC "Next32 Cond Branch Fixup"
#define NEXT32CONDBRANCHFIXUP_NAME "Next32-Cond-Branch-Fixup"

#define DEBUG_TYPE NEXT32CONDBRANCHFIXUP_NAME

namespace {
class Next32CondBranchFixupPass : public MachineFunctionPass {

public:
  static char ID;

  StringRef getPassName() const override { return NEXT32CONDBRANCHFIXUP_DESC; }

  Next32CondBranchFixupPass() : MachineFunctionPass(ID) {
    initializeNext32CondBranchFixupPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &Func) override;
};
} // namespace

char Next32CondBranchFixupPass::ID = 0;

INITIALIZE_PASS(Next32CondBranchFixupPass, NEXT32CONDBRANCHFIXUP_NAME,
                NEXT32CONDBRANCHFIXUP_DESC, false, false)

FunctionPass *llvm::createNext32CondBranchFixup() {
  return new Next32CondBranchFixupPass();
}

bool Next32CondBranchFixupPass::runOnMachineFunction(MachineFunction &Func) {
  Next32PassTrace TFunc(DEBUG_TYPE, Func);
  bool Changed = false;

  const TargetInstrInfo &TII = *(Func.getSubtarget().getInstrInfo());

  for (auto &MBB : TFunc) {
    MachineBasicBlock::iterator LastInstr = MBB.getLastNonDebugInstr();
    if ((LastInstr == MBB.end()) ||
        !(LastInstr->getDesc().TSFlags & Next32II::IsWriterChain)) {
      // Ignore unreachable (no return) functions
      if (MBB.succ_size() == 0) {
        if (LastInstr != MBB.end() &&
            (LastInstr->getOpcode() != Next32::CALL_TERMINATOR &&
             LastInstr->getOpcode() != Next32::CALL_TERMINATOR_TID) &&
            LastInstr->getOpcode() != Next32::FEEDER_ARGS)
          llvm_unreachable("Error in fall-through validation");
        continue;
      }
      // Fix fall-through, add BR instruction
      assert(MBB.succ_size() == 1 && "Error is fall-through");
      Changed = true;
      BuildMI(MBB, MBB.end(), DebugLoc(), TII.get(Next32::BR))
          .addMBB(*MBB.succ_begin());
    } else if (LastInstr->getOpcode() == Next32::BR) {
      // Ignore single line BR instruction
      if (LastInstr == MBB.begin())
        continue;
      MachineBasicBlock::iterator BR = LastInstr;
      MachineBasicBlock::iterator BR_CC = --LastInstr;

      if (BR_CC->getOpcode() != Next32::BR_CC)
        continue;

      // Replacing BR_CC/BR pairs into two BR_CCs
      Changed = true;
      Next32Constants::CondCode Cond =
          (Next32Constants::CondCode)BR_CC->getOperand(0).getImm();
      unsigned CondReg = BR_CC->getOperand(1).getReg();
      Next32Constants::CondCode ReverseCond =
          Next32Helpers::GetReverseNext32CC(Cond);

      BuildMI(MBB, MBB.end(), BR->getDebugLoc(), TII.get(Next32::BR_CC))
          .addImm(ReverseCond)
          .addReg(CondReg)
          .addMBB(BR->getOperand(0).getMBB());
      MBB.erase(BR);
    }
  }
  return Changed;
}
