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

static ARM::PredBlockMask ExpandBlockMask(ARM::PredBlockMask BlockMask,
                                          ARMVCC::VPTCodes Kind) {
  using PredBlockMask = ARM::PredBlockMask;
  assert(Kind != ARMVCC::None && "Cannot expand mask with 'None'");
  assert(countTrailingZeros((unsigned)BlockMask) != 0 &&
         "Mask is already full");

  auto ChooseMask = [&](PredBlockMask AddedThen, PredBlockMask AddedElse) {
    return (Kind == ARMVCC::Then) ? AddedThen : AddedElse;
  };

  switch (BlockMask) {
  case PredBlockMask::T:
    return ChooseMask(PredBlockMask::TT, PredBlockMask::TE);
  case PredBlockMask::TT:
    return ChooseMask(PredBlockMask::TTT, PredBlockMask::TTE);
  case PredBlockMask::TE:
    return ChooseMask(PredBlockMask::TET, PredBlockMask::TEE);
  case PredBlockMask::TTT:
    return ChooseMask(PredBlockMask::TTTT, PredBlockMask::TTTE);
  case PredBlockMask::TTE:
    return ChooseMask(PredBlockMask::TTET, PredBlockMask::TTEE);
  case PredBlockMask::TET:
    return ChooseMask(PredBlockMask::TETT, PredBlockMask::TETE);
  case PredBlockMask::TEE:
    return ChooseMask(PredBlockMask::TEET, PredBlockMask::TEEE);
  default:
    llvm_unreachable("Unknown Mask");
  }
}

// Advances Iter past a block of predicated instructions.
// Returns true if it successfully skipped the whole block of predicated
// instructions. Returns false when it stopped early (due to MaxSteps), or if
// Iter didn't point to a predicated instruction.
static bool StepOverPredicatedInstrs(MachineBasicBlock::instr_iterator &Iter,
                                     MachineBasicBlock::instr_iterator EndIter,
                                     unsigned MaxSteps,
                                     unsigned &NumInstrsSteppedOver) {
  ARMVCC::VPTCodes NextPred = ARMVCC::None;
  unsigned PredReg;
  NumInstrsSteppedOver = 0;

  while (Iter != EndIter) {
    NextPred = getVPTInstrPredicate(*Iter, PredReg);
    assert(NextPred != ARMVCC::Else &&
           "VPT block pass does not expect Else preds");
    if (NextPred == ARMVCC::None || MaxSteps == 0)
      break;
    --MaxSteps;
    ++Iter;
    ++NumInstrsSteppedOver;
  };

  return NumInstrsSteppedOver != 0 &&
         (NextPred == ARMVCC::None || Iter == EndIter);
}

// Returns true if at least one instruction in the range [Iter, End) defines
// or kills VPR.
static bool IsVPRDefinedOrKilledByBlock(MachineBasicBlock::iterator Iter,
                                        MachineBasicBlock::iterator End) {
  for (; Iter != End; ++Iter)
    if (Iter->definesRegister(ARM::VPR) || Iter->killsRegister(ARM::VPR))
      return true;
  return false;
}

// Given an iterator (Iter) that points at an instruction with a "Then"
// predicate, tries to create the largest block of continuous predicated
// instructions possible, and returns the VPT Block Mask of that block.
//
// This will try to perform some minor optimization in order to maximize the
// size of the block.
static ARM::PredBlockMask
CreateVPTBlock(MachineBasicBlock::instr_iterator &Iter,
               MachineBasicBlock::instr_iterator EndIter,
               SmallVectorImpl<MachineInstr *> &DeadInstructions) {
  MachineBasicBlock::instr_iterator BlockBeg = Iter;
  (void)BlockBeg;
  assert(getVPTInstrPredicate(*Iter) == ARMVCC::Then &&
         "Expected a Predicated Instruction");

  LLVM_DEBUG(dbgs() << "VPT block created for: "; Iter->dump());

  unsigned BlockSize;
  StepOverPredicatedInstrs(Iter, EndIter, 4, BlockSize);

  LLVM_DEBUG(for (MachineBasicBlock::instr_iterator AddedInstIter =
                      std::next(BlockBeg);
                  AddedInstIter != Iter; ++AddedInstIter) {
    dbgs() << "  adding: ";
    AddedInstIter->dump();
  });

  // Generate the initial BlockMask
  ARM::PredBlockMask BlockMask = getARMVPTBlockMask(BlockSize);

  // Remove VPNOTs while there's still room in the block, so we can make the
  // largest block possible.
  ARMVCC::VPTCodes CurrentPredicate = ARMVCC::Then;
  while (BlockSize < 4 && Iter != EndIter &&
         Iter->getOpcode() == ARM::MVE_VPNOT) {

    // Try to skip all of the predicated instructions after the VPNOT, stopping
    // after (4 - BlockSize). If we can't skip them all, stop.
    unsigned ElseInstCnt = 0;
    MachineBasicBlock::instr_iterator VPNOTBlockEndIter = std::next(Iter);
    if (!StepOverPredicatedInstrs(VPNOTBlockEndIter, EndIter, (4 - BlockSize),
                                  ElseInstCnt))
      break;

    // Check if this VPNOT can be removed or not: It can only be removed if at
    // least one of the predicated instruction that follows it kills or sets
    // VPR.
    if (!IsVPRDefinedOrKilledByBlock(Iter, VPNOTBlockEndIter))
      break;

    LLVM_DEBUG(dbgs() << "  removing VPNOT: "; Iter->dump(););

    // Record the new size of the block
    BlockSize += ElseInstCnt;
    assert(BlockSize <= 4 && "Block is too large!");

    // Record the VPNot to remove it later.
    DeadInstructions.push_back(&*Iter);
    ++Iter;

    // Replace "then" by "elses" in the block until we find an instruction that
    // defines VPR, then after that leave everything to "t".
    // Note that we are using "Iter" to iterate over the block so we can update
    // it at the same time.
    bool ChangeToElse = (CurrentPredicate == ARMVCC::Then);
    for (; Iter != VPNOTBlockEndIter; ++Iter) {
      // Find the register in which the predicate is
      int OpIdx = findFirstVPTPredOperandIdx(*Iter);
      assert(OpIdx != -1);

      // Update the mask + change the predicate to an else if needed.
      if (ChangeToElse) {
        // Change the predicate and update the mask
        Iter->getOperand(OpIdx).setImm(ARMVCC::Else);
        BlockMask = ExpandBlockMask(BlockMask, ARMVCC::Else);
        // Reset back to a "then" predicate if this instruction defines VPR.
        if (Iter->definesRegister(ARM::VPR))
          ChangeToElse = false;
      } else
        BlockMask = ExpandBlockMask(BlockMask, ARMVCC::Then);

      LLVM_DEBUG(dbgs() << "  adding: "; Iter->dump());
    }

    CurrentPredicate =
        (CurrentPredicate == ARMVCC::Then ? ARMVCC::Else : ARMVCC::Then);
  }
  return BlockMask;
}

bool MVEVPTBlock::InsertVPTBlocks(MachineBasicBlock &Block) {
  bool Modified = false;
  MachineBasicBlock::instr_iterator MBIter = Block.instr_begin();
  MachineBasicBlock::instr_iterator EndIter = Block.instr_end();

  SmallVector<MachineInstr *, 4> DeadInstructions;

  while (MBIter != EndIter) {
    MachineInstr *MI = &*MBIter;
    unsigned PredReg = 0;
    DebugLoc DL = MI->getDebugLoc();

    ARMVCC::VPTCodes Pred = getVPTInstrPredicate(*MI, PredReg);

    // The idea of the predicate is that None, Then and Else are for use when
    // handling assembly language: they correspond to the three possible
    // suffixes "", "t" and "e" on the mnemonic. So when instructions are read
    // from assembly source or disassembled from object code, you expect to
    // see a mixture whenever there's a long VPT block. But in code
    // generation, we hope we'll never generate an Else as input to this pass.
    assert(Pred != ARMVCC::Else && "VPT block pass does not expect Else preds");

    if (Pred == ARMVCC::None) {
      ++MBIter;
      continue;
    }

    ARM::PredBlockMask BlockMask =
        CreateVPTBlock(MBIter, EndIter, DeadInstructions);

    // Search back for a VCMP that can be folded to create a VPT, or else
    // create a VPST directly
    MachineInstrBuilder MIBuilder;
    unsigned NewOpcode;
    LLVM_DEBUG(dbgs() << "  final block mask: " << (unsigned)BlockMask << "\n");
    if (MachineInstr *VCMP = findVCMPToFoldIntoVPST(MI, TRI, NewOpcode)) {
      LLVM_DEBUG(dbgs() << "  folding VCMP into VPST: "; VCMP->dump());
      MIBuilder = BuildMI(Block, MI, DL, TII->get(NewOpcode));
      MIBuilder.addImm((uint64_t)BlockMask);
      MIBuilder.add(VCMP->getOperand(1));
      MIBuilder.add(VCMP->getOperand(2));
      MIBuilder.add(VCMP->getOperand(3));
      VCMP->eraseFromParent();
    } else {
      MIBuilder = BuildMI(Block, MI, DL, TII->get(ARM::MVE_VPST));
      MIBuilder.addImm((uint64_t)BlockMask);
    }

    finalizeBundle(
        Block, MachineBasicBlock::instr_iterator(MIBuilder.getInstr()), MBIter);

    Modified = true;
  }

  // Erase all dead instructions
  for (MachineInstr *DeadMI : DeadInstructions) {
    if (DeadMI->isInsideBundle())
      DeadMI->eraseFromBundle();
    else
      DeadMI->eraseFromParent();
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
