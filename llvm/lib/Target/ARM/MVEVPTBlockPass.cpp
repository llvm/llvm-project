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

static unsigned VCMPOpcodeToVPT(unsigned Opcode) {
  switch (Opcode) {
  case ARM::MVE_VCMPf32:
    return ARM::MVE_VPTv4f32;
  case ARM::MVE_VCMPf16:
    return ARM::MVE_VPTv8f16;
  case ARM::MVE_VCMPi8:
    return ARM::MVE_VPTv16i8;
  case ARM::MVE_VCMPi16:
    return ARM::MVE_VPTv8i16;
  case ARM::MVE_VCMPi32:
    return ARM::MVE_VPTv4i32;
  case ARM::MVE_VCMPu8:
    return ARM::MVE_VPTv16u8;
  case ARM::MVE_VCMPu16:
    return ARM::MVE_VPTv8u16;
  case ARM::MVE_VCMPu32:
    return ARM::MVE_VPTv4u32;
  case ARM::MVE_VCMPs8:
    return ARM::MVE_VPTv16s8;
  case ARM::MVE_VCMPs16:
    return ARM::MVE_VPTv8s16;
  case ARM::MVE_VCMPs32:
    return ARM::MVE_VPTv4s32;

  case ARM::MVE_VCMPf32r:
    return ARM::MVE_VPTv4f32r;
  case ARM::MVE_VCMPf16r:
    return ARM::MVE_VPTv8f16r;
  case ARM::MVE_VCMPi8r:
    return ARM::MVE_VPTv16i8r;
  case ARM::MVE_VCMPi16r:
    return ARM::MVE_VPTv8i16r;
  case ARM::MVE_VCMPi32r:
    return ARM::MVE_VPTv4i32r;
  case ARM::MVE_VCMPu8r:
    return ARM::MVE_VPTv16u8r;
  case ARM::MVE_VCMPu16r:
    return ARM::MVE_VPTv8u16r;
  case ARM::MVE_VCMPu32r:
    return ARM::MVE_VPTv4u32r;
  case ARM::MVE_VCMPs8r:
    return ARM::MVE_VPTv16s8r;
  case ARM::MVE_VCMPs16r:
    return ARM::MVE_VPTv8s16r;
  case ARM::MVE_VCMPs32r:
    return ARM::MVE_VPTv4s32r;

  default:
    return 0;
  }
}

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

    unsigned BlockMask = 0;
    switch (VPTInstCnt) {
    case 1:
      BlockMask = VPTMaskValue::T;
      break;
    case 2:
      BlockMask = VPTMaskValue::TT;
      break;
    case 3:
      BlockMask = VPTMaskValue::TTT;
      break;
    case 4:
      BlockMask = VPTMaskValue::TTTT;
      break;
    default:
      llvm_unreachable("Unexpected number of instruction in a VPT block");
    };

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
