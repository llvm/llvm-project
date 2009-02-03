//===-- X86FloatingPoint.cpp - FP_REG_KILL inserter -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass which inserts FP_REG_KILL instructions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "x86-codegen"
#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/Instructions.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/CFG.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumFPKill, "Number of FP_REG_KILL instructions added");

namespace {
  struct VISIBILITY_HIDDEN FPRegKiller : public MachineFunctionPass {
    static char ID;
    FPRegKiller() : MachineFunctionPass(&ID) {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addPreservedID(MachineLoopInfoID);
      AU.addPreservedID(MachineDominatorsID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const { return "X86 FP_REG_KILL inserter"; }
  };
  char FPRegKiller::ID = 0;
}

FunctionPass *llvm::createX87FPRegKillInserterPass() { return new FPRegKiller(); }

bool FPRegKiller::runOnMachineFunction(MachineFunction &MF) {
  // If we are emitting FP stack code, scan the basic block to determine if this
  // block defines any FP values.  If so, put an FP_REG_KILL instruction before
  // the terminator of the block.

  // Note that FP stack instructions are used in all modes for long double,
  // so we always need to do this check.
  // Also note that it's possible for an FP stack register to be live across
  // an instruction that produces multiple basic blocks (SSE CMOV) so we
  // must check all the generated basic blocks.

  // Scan all of the machine instructions in these MBBs, checking for FP
  // stores.  (RFP32 and RFP64 will not exist in SSE mode, but RFP80 might.)

  // Fast-path: If nothing is using the x87 registers, we don't need to do
  // any scanning.
  MachineRegisterInfo &MRI = MF.getRegInfo();
  if (MRI.getRegClassVirtRegs(X86::RFP80RegisterClass).empty() &&
      MRI.getRegClassVirtRegs(X86::RFP64RegisterClass).empty() &&
      MRI.getRegClassVirtRegs(X86::RFP32RegisterClass).empty())
    return false;

  bool Changed = false;
  const X86Subtarget &Subtarget = MF.getTarget().getSubtarget<X86Subtarget>();
  MachineFunction::iterator MBBI = MF.begin();
  MachineFunction::iterator EndMBB = MF.end();
  for (; MBBI != EndMBB; ++MBBI) {
    MachineBasicBlock *MBB = MBBI;
    
    // If this block returns, ignore it.  We don't want to insert an FP_REG_KILL
    // before the return.
    if (!MBB->empty()) {
      MachineBasicBlock::iterator EndI = MBB->end();
      --EndI;
      if (EndI->getDesc().isReturn())
        continue;
    }
    
    bool ContainsFPCode = false;
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
         !ContainsFPCode && I != E; ++I) {
      if (I->getNumOperands() != 0 && I->getOperand(0).isReg()) {
        const TargetRegisterClass *clas;
        for (unsigned op = 0, e = I->getNumOperands(); op != e; ++op) {
          if (I->getOperand(op).isReg() && I->getOperand(op).isDef() &&
            TargetRegisterInfo::isVirtualRegister(I->getOperand(op).getReg()) &&
              ((clas = MRI.getRegClass(I->getOperand(op).getReg())) == 
                 X86::RFP32RegisterClass ||
               clas == X86::RFP64RegisterClass ||
               clas == X86::RFP80RegisterClass)) {
            ContainsFPCode = true;
            break;
          }
        }
      }
    }
    // Check PHI nodes in successor blocks.  These PHI's will be lowered to have
    // a copy of the input value in this block.  In SSE mode, we only care about
    // 80-bit values.
    if (!ContainsFPCode) {
      // Final check, check LLVM BB's that are successors to the LLVM BB
      // corresponding to BB for FP PHI nodes.
      const BasicBlock *LLVMBB = MBB->getBasicBlock();
      const PHINode *PN;
      for (succ_const_iterator SI = succ_begin(LLVMBB), E = succ_end(LLVMBB);
           !ContainsFPCode && SI != E; ++SI) {
        for (BasicBlock::const_iterator II = SI->begin();
             (PN = dyn_cast<PHINode>(II)); ++II) {
          if (PN->getType()==Type::X86_FP80Ty ||
              (!Subtarget.hasSSE1() && PN->getType()->isFloatingPoint()) ||
              (!Subtarget.hasSSE2() && PN->getType()==Type::DoubleTy)) {
            ContainsFPCode = true;
            break;
          }
        }
      }
    }
    // Finally, if we found any FP code, emit the FP_REG_KILL instruction.
    if (ContainsFPCode) {
      BuildMI(*MBB, MBBI->getFirstTerminator(),
              MF.getTarget().getInstrInfo()->get(X86::FP_REG_KILL));
      ++NumFPKill;
      Changed = true;
    }
  }

  return Changed;
}
