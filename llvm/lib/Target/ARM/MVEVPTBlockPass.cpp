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
#include "llvm/Support/Debug.h"
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

static MachineInstr *findVCMPToFoldIntoVPST(MachineBasicBlock::iterator MI,
                                            const TargetRegisterInfo *TRI,
                                            unsigned &NewOpcode) {
  // Search backwards to the instruction that defines VPR. This may or not
  // be a VCMP, we check that after this loop. If we find another instruction
  // that reads cpsr, we return nullptr.
  MachineBasicBlock::iterator CmpMI = MI;
  while (CmpMI != MI->getParent()->begin()) {
    --CmpMI;
    if (CmpMI->modifiesRegister(ARM::VPR, TRI))
      break;
    if (CmpMI->readsRegister(ARM::VPR, TRI))
      break;
  }

  if (CmpMI == MI)
    return nullptr;
  NewOpcode = VCMPOpcodeToVPT(CmpMI->getOpcode());
  if (NewOpcode == 0)
    return nullptr;

  // Search forward from CmpMI to MI, checking if either register was def'd
  if (registerDefinedBetween(CmpMI->getOperand(1).getReg(), std::next(CmpMI),
                             MI, TRI))
    return nullptr;
  if (registerDefinedBetween(CmpMI->getOperand(2).getReg(), std::next(CmpMI),
                             MI, TRI))
    return nullptr;
  return &*CmpMI;
}

bool MVEVPTBlock::InsertVPTBlocks(MachineBasicBlock &Block) {
  bool Modified = false;
  MachineBasicBlock::instr_iterator MBIter = Block.instr_begin();
  MachineBasicBlock::instr_iterator EndIter = Block.instr_end();

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

    LLVM_DEBUG(dbgs() << "VPT block created for: "; MI->dump());
    int VPTInstCnt = 1;
    ARMVCC::VPTCodes NextPred;

    // Look at subsequent instructions, checking if they can be in the same VPT
    // block.
    ++MBIter;
    while (MBIter != EndIter && VPTInstCnt < 4) {
      NextPred = getVPTInstrPredicate(*MBIter, PredReg);
      assert(NextPred != ARMVCC::Else &&
             "VPT block pass does not expect Else preds");
      if (NextPred != Pred)
        break;
      LLVM_DEBUG(dbgs() << "  adding : "; MBIter->dump());
      ++VPTInstCnt;
      ++MBIter;
    };

    unsigned BlockMask = getARMVPTBlockMask(VPTInstCnt);

    // Search back for a VCMP that can be folded to create a VPT, or else create
    // a VPST directly
    MachineInstrBuilder MIBuilder;
    unsigned NewOpcode;
    MachineInstr *VCMP = findVCMPToFoldIntoVPST(MI, TRI, NewOpcode);
    if (VCMP) {
      LLVM_DEBUG(dbgs() << "  folding VCMP into VPST: "; VCMP->dump());
      MIBuilder = BuildMI(Block, MI, dl, TII->get(NewOpcode));
      MIBuilder.addImm(BlockMask);
      MIBuilder.add(VCMP->getOperand(1));
      MIBuilder.add(VCMP->getOperand(2));
      MIBuilder.add(VCMP->getOperand(3));
      VCMP->eraseFromParent();
    } else {
      MIBuilder = BuildMI(Block, MI, dl, TII->get(ARM::MVE_VPST));
      MIBuilder.addImm(BlockMask);
    }

    finalizeBundle(
        Block, MachineBasicBlock::instr_iterator(MIBuilder.getInstr()), MBIter);

    Modified = true;
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
