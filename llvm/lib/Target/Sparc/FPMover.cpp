//===-- FPMover.cpp - Sparc double-precision floating point move fixer ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Expand FpMOVD/FpABSD/FpNEGD instructions into their single-precision pieces.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "fpmover"
#include "Sparc.h"
#include "SparcSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

STATISTIC(NumFpDs , "Number of instructions translated");
STATISTIC(NoopFpDs, "Number of noop instructions removed");

namespace {
  struct FPMover : public MachineFunctionPass {
    /// Target machine description which we query for reg. names, data
    /// layout, etc.
    ///
    TargetMachine &TM;
    
    static char ID;
    explicit FPMover(TargetMachine &tm) 
      : MachineFunctionPass(ID), TM(tm) { }

    virtual const char *getPassName() const {
      return "Sparc Double-FP Move Fixer";
    }

    bool runOnMachineBasicBlock(MachineBasicBlock &MBB);
    bool runOnMachineFunction(MachineFunction &F);
  };
  char FPMover::ID = 0;
} // end of anonymous namespace

/// createSparcFPMoverPass - Returns a pass that turns FpMOVD
/// instructions into FMOVS instructions
///
FunctionPass *llvm::createSparcFPMoverPass(TargetMachine &tm) {
  return new FPMover(tm);
}

/// getDoubleRegPair - Given a DFP register, return the even and odd FP
/// registers that correspond to it.
static void getDoubleRegPair(unsigned DoubleReg, unsigned &EvenReg,
                             unsigned &OddReg) {
  static const uint16_t EvenHalvesOfPairs[] = {
    SP::F0, SP::F2, SP::F4, SP::F6, SP::F8, SP::F10, SP::F12, SP::F14,
    SP::F16, SP::F18, SP::F20, SP::F22, SP::F24, SP::F26, SP::F28, SP::F30
  };
  static const uint16_t OddHalvesOfPairs[] = {
    SP::F1, SP::F3, SP::F5, SP::F7, SP::F9, SP::F11, SP::F13, SP::F15,
    SP::F17, SP::F19, SP::F21, SP::F23, SP::F25, SP::F27, SP::F29, SP::F31
  };
  static const uint16_t DoubleRegsInOrder[] = {
    SP::D0, SP::D1, SP::D2, SP::D3, SP::D4, SP::D5, SP::D6, SP::D7, SP::D8,
    SP::D9, SP::D10, SP::D11, SP::D12, SP::D13, SP::D14, SP::D15
  };
  for (unsigned i = 0; i < array_lengthof(DoubleRegsInOrder); ++i)
    if (DoubleRegsInOrder[i] == DoubleReg) {
      EvenReg = EvenHalvesOfPairs[i];
      OddReg = OddHalvesOfPairs[i];
      return;
    }
  llvm_unreachable("Can't find reg");
}

/// runOnMachineBasicBlock - Fixup FpMOVD instructions in this MBB.
///
bool FPMover::runOnMachineBasicBlock(MachineBasicBlock &MBB) {
  bool Changed = false;
  for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ) {
    MachineInstr *MI = I++;
    DebugLoc dl = MI->getDebugLoc();
    if (MI->getOpcode() == SP::FpMOVD || MI->getOpcode() == SP::FpABSD ||
        MI->getOpcode() == SP::FpNEGD) {
      Changed = true;
      unsigned DestDReg = MI->getOperand(0).getReg();
      unsigned SrcDReg  = MI->getOperand(1).getReg();
      if (DestDReg == SrcDReg && MI->getOpcode() == SP::FpMOVD) {
        MBB.erase(MI);   // Eliminate the noop copy.
        ++NoopFpDs;
        continue;
      }
      
      unsigned EvenSrcReg = 0, OddSrcReg = 0, EvenDestReg = 0, OddDestReg = 0;
      getDoubleRegPair(DestDReg, EvenDestReg, OddDestReg);
      getDoubleRegPair(SrcDReg, EvenSrcReg, OddSrcReg);

      const TargetInstrInfo *TII = TM.getInstrInfo();
      if (MI->getOpcode() == SP::FpMOVD)
        MI->setDesc(TII->get(SP::FMOVS));
      else if (MI->getOpcode() == SP::FpNEGD)
        MI->setDesc(TII->get(SP::FNEGS));
      else if (MI->getOpcode() == SP::FpABSD)
        MI->setDesc(TII->get(SP::FABSS));
      else
        llvm_unreachable("Unknown opcode!");
        
      MI->getOperand(0).setReg(EvenDestReg);
      MI->getOperand(1).setReg(EvenSrcReg);
      DEBUG(errs() << "FPMover: the modified instr is: " << *MI);
      // Insert copy for the other half of the double.
      if (DestDReg != SrcDReg) {
        MI = BuildMI(MBB, I, dl, TM.getInstrInfo()->get(SP::FMOVS), OddDestReg)
          .addReg(OddSrcReg);
        DEBUG(errs() << "FPMover: the inserted instr is: " << *MI);
      }
      ++NumFpDs;
    }
  }
  return Changed;
}

bool FPMover::runOnMachineFunction(MachineFunction &F) {
  // If the target has V9 instructions, the fp-mover pseudos will never be
  // emitted.  Avoid a scan of the instructions to improve compile time.
  if (TM.getSubtarget<SparcSubtarget>().isV9())
    return false;
  
  bool Changed = false;
  for (MachineFunction::iterator FI = F.begin(), FE = F.end();
       FI != FE; ++FI)
    Changed |= runOnMachineBasicBlock(*FI);
  return Changed;
}
