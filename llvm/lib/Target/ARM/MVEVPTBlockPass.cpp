//===-- MVEVPTBlockPass.cpp - Insert MVE VPT blocks -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMSubtarget.h"
#include "MCTargetDesc/ARMBaseInfo.h"
#include "Thumb2InstrInfo.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCRegisterInfo.h"
#include <cassert>
#include <new>

using namespace llvm;

#define DEBUG_TYPE "arm-mve-vpt"

namespace {
  class MVEVPTBlock : public MachineFunctionPass {
  public:
    static char ID;
    const Thumb2InstrInfo *TII;
    const TargetRegisterInfo *TRI;

    MVEVPTBlock() : MachineFunctionPass(ID) {}

    bool runOnMachineFunction(MachineFunction &Fn) override;

    MachineFunctionProperties getRequiredProperties() const override {
      return MachineFunctionProperties().set(
          MachineFunctionProperties::Property::NoVRegs);
    }

    StringRef getPassName() const override {
      return "MVE VPT block insertion pass";
    }

  private:
    bool InsertVPTBlocks(MachineBasicBlock &MBB);
  };

  char MVEVPTBlock::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(MVEVPTBlock, DEBUG_TYPE, "ARM MVE VPT block pass", false, false)

enum VPTMaskValue {
  T     =  8, // 0b1000
  TT    =  4, // 0b0100
  TE    = 12, // 0b1100
  TTT   =  2, // 0b0010
  TTE   =  6, // 0b0110
  TEE   = 10, // 0b1010
  TET   = 14, // 0b1110
  TTTT  =  1, // 0b0001
  TTTE  =  3, // 0b0011
  TTEE  =  5, // 0b0101
  TTET  =  7, // 0b0111
  TEEE  =  9, // 0b1001
  TEET  = 11, // 0b1011
  TETT  = 13, // 0b1101
  TETE  = 15  // 0b1111
};

bool MVEVPTBlock::InsertVPTBlocks(MachineBasicBlock &Block) {
  bool Modified = false;
  MachineBasicBlock::iterator MBIter = Block.begin();
  MachineBasicBlock::iterator EndIter = Block.end();

  while (MBIter != EndIter) {
    MachineInstr *MI = &*MBIter;
    unsigned PredReg = 0;
    DebugLoc dl = MI->getDebugLoc();

    ARMVCC::VPTCodes Pred = getVPTInstrPredicate(*MI, PredReg);

    // The idea of the predicate is that None, Then and Else are for use when
    // handling assembly language: they correspond to the three possible
    // suffixes "", "t" and "e" on the mnemonic. So when instructions are read
    // from assembly source or disassembled from object code, you expect to see
    // a mixture whenever there's a long VPT block. But in code generation, we
    // hope we'll never generate an Else as input to this pass.
    assert(Pred != ARMVCC::Else && "VPT block pass does not expect Else preds");

    if (Pred == ARMVCC::None) {
      ++MBIter;
      continue;
    }

    MachineInstrBuilder MIBuilder =
        BuildMI(Block, MBIter, dl, TII->get(ARM::MVE_VPST));

    MachineBasicBlock::iterator VPSTInsertPos = MIBuilder.getInstr();
    int VPTInstCnt = 1;
    ARMVCC::VPTCodes NextPred;

    do {
      ++MBIter;
      NextPred = getVPTInstrPredicate(*MBIter, PredReg);
    } while (NextPred != ARMVCC::None && NextPred == Pred && ++VPTInstCnt < 4);

    switch (VPTInstCnt) {
    case 1:
      MIBuilder.addImm(VPTMaskValue::T);
      break;
    case 2:
      MIBuilder.addImm(VPTMaskValue::TT);
      break;
    case 3:
      MIBuilder.addImm(VPTMaskValue::TTT);
      break;
    case 4:
      MIBuilder.addImm(VPTMaskValue::TTTT);
      break;
    default:
      llvm_unreachable("Unexpected number of instruction in a VPT block");
    };

    MachineInstr *LastMI = &*MBIter;
    finalizeBundle(Block, VPSTInsertPos.getInstrIterator(),
                   ++LastMI->getIterator());

    Modified = true;
    LLVM_DEBUG(dbgs() << "VPT block created for: "; MI->dump());

    ++MBIter;
  }
  return Modified;
}

bool MVEVPTBlock::runOnMachineFunction(MachineFunction &Fn) {
  const ARMSubtarget &STI =
      static_cast<const ARMSubtarget &>(Fn.getSubtarget());

  if (!STI.isThumb2() || !STI.hasMVEIntegerOps())
    return false;

  TII = static_cast<const Thumb2InstrInfo *>(STI.getInstrInfo());
  TRI = STI.getRegisterInfo();

  LLVM_DEBUG(dbgs() << "********** ARM MVE VPT BLOCKS **********\n"
                    << "********** Function: " << Fn.getName() << '\n');

  bool Modified = false;
  for (MachineBasicBlock &MBB : Fn)
    Modified |= InsertVPTBlocks(MBB);

  LLVM_DEBUG(dbgs() << "**************************************\n");
  return Modified;
}

/// createMVEVPTBlock - Returns an instance of the MVE VPT block
/// insertion pass.
FunctionPass *llvm::createMVEVPTBlockPass() { return new MVEVPTBlock(); }
