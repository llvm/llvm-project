//===------ LeonPasses.cpp - Define passes specific to LEON ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "LeonPasses.h"
#include "SparcSubtarget.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

char ErrataWorkaround::ID = 0;

ErrataWorkaround::ErrataWorkaround() : MachineFunctionPass(ID) {
  initializeErrataWorkaroundPass(*PassRegistry::getPassRegistry());
}

INITIALIZE_PASS(ErrataWorkaround, "errata-workaround", "Errata workaround pass",
                false, false)

// Move iterator to the next instruction in the function, ignoring
// meta instructions and inline assembly. Returns false when reaching
// the end of the function.
bool ErrataWorkaround::moveNext(MachineBasicBlock::iterator &I) {

  MachineBasicBlock *MBB = I->getParent();

  do {
    I++;

    while (I == MBB->end()) {
      if (MBB->getFallThrough() == nullptr)
        return false;
      MBB = MBB->getFallThrough();
      I = MBB->begin();
    }
  } while (I->isMetaInstruction() || I->isInlineAsm());

  return true;
}

void ErrataWorkaround::insertNop(MachineBasicBlock::iterator I) {
  BuildMI(*I->getParent(), I, I->getDebugLoc(), TII->get(SP::NOP));
}

bool ErrataWorkaround::isFloat(MachineBasicBlock::iterator I) {
  if (I->getNumOperands() == 0)
    return false;

  if (!I->getOperand(0).isReg())
    return false;

  unsigned reg = I->getOperand(0).getReg();

  if (!SP::FPRegsRegClass.contains(reg) && !SP::DFPRegsRegClass.contains(reg))
    return false;

  return true;
}

bool ErrataWorkaround::isDivSqrt(MachineBasicBlock::iterator I) {
  switch (I->getOpcode()) {
  case SP::FDIVS:
  case SP::FDIVD:
  case SP::FSQRTS:
  case SP::FSQRTD:
    return true;
  }
  return false;
}

// Prevents the following code sequence from being generated:
// (stb/sth/st/stf) -> (single non-store/load instruction) -> (any store)
// If the sequence is detected a NOP instruction is inserted after
// the first store instruction.
bool ErrataWorkaround::checkSeqTN0009A(MachineBasicBlock::iterator I) {
  switch (I->getOpcode()) {
  case SP::STrr:
  case SP::STri:
  case SP::STBrr:
  case SP::STBri:
  case SP::STHrr:
  case SP::STHri:
  case SP::STFrr:
  case SP::STFri:
    break;
  default:
    return false;
  }

  MachineBasicBlock::iterator MI = I;
  if (!moveNext(MI))
    return false;

  if (MI->mayStore() || MI->mayLoad())
    return false;

  MachineBasicBlock::iterator PatchHere = MI;

  if (!moveNext(MI))
    return false;

  if (!MI->mayStore())
    return false;

  insertNop(PatchHere);
  return true;
}

// Prevents the following code sequence from being generated:
// (std/stdf) -> (any store)
// If the sequence is detected a NOP instruction is inserted after
// the first store instruction.
bool ErrataWorkaround::checkSeqTN0009B(MachineBasicBlock::iterator I) {

  switch (I->getOpcode()) {
  case SP::STDrr:
  case SP::STDri:
  case SP::STDFrr:
  case SP::STDFri:
    break;
  default:
    return false;
  }

  MachineBasicBlock::iterator MI = I;

  if (!moveNext(MI))
    return false;

  if (!MI->mayStore())
    return false;

  insertNop(MI);
  return true;
}

// Insert a NOP at branch target if load in delay slot and atomic
// instruction at branch target. Also insert a NOP between load
// instruction and atomic instruction (swap or casa).
bool ErrataWorkaround::checkSeqTN0010(MachineBasicBlock::iterator I) {

  // Check for load instruction or branch bundled with load instruction
  if (!I->mayLoad())
    return false;

  // Check for branch to atomic instruction with load in delay slot
  if (I->isBranch()) {
    MachineBasicBlock *TargetMBB = I->getOperand(0).getMBB();
    MachineBasicBlock::iterator MI = TargetMBB->begin();

    while (MI != TargetMBB->end() && MI->isMetaInstruction())
      MI++;

    if (MI == TargetMBB->end())
      return false;

    switch (MI->getOpcode()) {
    case SP::SWAPrr:
    case SP::SWAPri:
    case SP::CASArr:
      insertNop(MI);
      break;
    default:
      break;
    }
  }

  // Check for load followed by atomic instruction
  MachineBasicBlock::iterator MI = I;
  if (!moveNext(MI))
    return false;

  switch (MI->getOpcode()) {
  case SP::SWAPrr:
  case SP::SWAPri:
  case SP::CASArr:
    break;
  default:
    return false;
  }
  insertNop(MI);
  return true;
}

// Do not allow functions to begin with an atomic instruction
bool ErrataWorkaround::checkSeqTN0010First(MachineBasicBlock &MBB) {
  MachineBasicBlock::iterator I = MBB.begin();
  while (I != MBB.end() && I->isMetaInstruction())
    I++;
  switch (I->getOpcode()) {
  case SP::SWAPrr:
  case SP::SWAPri:
  case SP::CASArr:
    break;
  default:
    return false;
  }
  insertNop(I);
  return true;
}

// Inserts a NOP instruction at the target of an integer branch if the
// target is a floating-point instruction or floating-point branch.
bool ErrataWorkaround::checkSeqTN0012(MachineBasicBlock::iterator I) {

  if (I->getOpcode() != SP::BCOND && I->getOpcode() != SP::BCONDA)
    return false;

  MachineBasicBlock *TargetMBB = I->getOperand(0).getMBB();
  MachineBasicBlock::iterator MI = TargetMBB->begin();

  while (MI != TargetMBB->end() && MI->isMetaInstruction())
    MI++;

  if (MI == TargetMBB->end())
    return false;

  if (!isFloat(MI) && MI->getOpcode() != SP::FBCOND)
    return false;

  insertNop(MI);
  return true;
}

// Prevents the following code sequence from being generated:
// (div/sqrt) -> (2 to 3 floating-point operations or loads) -> (div/sqrt)
// If the sequence is detected one or two NOP instruction are inserted after
// the first div/sqrt instruction. No NOPs are inserted if one of the floating-
// point instructions in the middle of the sequence is a (div/sqrt), or if
// they have dependency on the destination register of the first (div/sqrt).
//
// The function also prevents the following code sequence from being generated,
// (div/sqrt) -> (branch), by inserting a NOP instruction after the (div/sqrt).
bool ErrataWorkaround::checkSeqTN0013(MachineBasicBlock::iterator I) {

  if (!isDivSqrt(I))
    return false;

  unsigned dstReg = I->getOperand(0).getReg();

  MachineBasicBlock::iterator MI = I;
  if (!moveNext(MI))
    return false;

  if (MI->isBranch()) {
    insertNop(MI);
    return true;
  }

  MachineBasicBlock::iterator PatchHere = MI;

  unsigned fpFound = 0;
  for (unsigned i = 0; i < 4; i++) {

    if (!isFloat(MI)) {
      if (!moveNext(MI))
        return false;
      continue;
    }

    if (MI->readsRegister(dstReg, TRI))
      return false;

    if (isDivSqrt(MI)) {
      if (i < 2)
        return false;
      if (fpFound < 2)
        return false;

      insertNop(PatchHere);
      if (i == 2)
        insertNop(PatchHere);
      return true;
    }

    fpFound++;
    if (!moveNext(MI))
      return false;
  }

  return false;
}

bool ErrataWorkaround::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;
  ST = &MF.getSubtarget<SparcSubtarget>();

  if (!(ST->fixTN0009() || ST->fixTN0010() || ST->fixTN0012() ||
        ST->fixTN0013()))
    return false;

  TII = ST->getInstrInfo();
  TRI = ST->getRegisterInfo();

  if (ST->fixTN0010())
    Changed |= checkSeqTN0010First(MF.front());

  for (auto &MBB : MF) {
    for (auto &I : MBB) {
      if (ST->fixTN0009()) {
        Changed |= checkSeqTN0009A(I);
        Changed |= checkSeqTN0009B(I);
      }
      if (ST->fixTN0010())
        Changed |= checkSeqTN0010(I);
      if (ST->fixTN0012())
        Changed |= checkSeqTN0012(I);
      if (ST->fixTN0013())
        Changed |= checkSeqTN0013(I);
    }
  }
  return Changed;
}

LEONMachineFunctionPass::LEONMachineFunctionPass(char &ID)
    : MachineFunctionPass(ID) {}

//*****************************************************************************
//**** InsertNOPLoad pass
//*****************************************************************************
// This pass fixes the incorrectly working Load instructions that exists for
// some earlier versions of the LEON processor line. NOP instructions must
// be inserted after the load instruction to ensure that the Load instruction
// behaves as expected for these processors.
//
// This pass inserts a NOP after any LD or LDF instruction.
//
char InsertNOPLoad::ID = 0;

InsertNOPLoad::InsertNOPLoad() : LEONMachineFunctionPass(ID) {}

bool InsertNOPLoad::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<SparcSubtarget>();
  if (!Subtarget->insertNOPLoad())
    return false;

  const TargetInstrInfo &TII = *Subtarget->getInstrInfo();
  DebugLoc DL = DebugLoc();

  bool Modified = false;
  for (MachineBasicBlock &MBB : MF) {
    for (auto MBBI = MBB.begin(), E = MBB.end(); MBBI != E; ++MBBI) {
      MachineInstr &MI = *MBBI;
      unsigned Opcode = MI.getOpcode();
      if (Opcode >= SP::LDDArr && Opcode <= SP::LDrr) {
        MachineBasicBlock::iterator NMBBI = std::next(MBBI);
        BuildMI(MBB, NMBBI, DL, TII.get(SP::NOP));
        Modified = true;
      }
    }
  }

  return Modified;
}



//*****************************************************************************
//**** DetectRoundChange pass
//*****************************************************************************
// To prevent any explicit change of the default rounding mode, this pass
// detects any call of the fesetround function.
// A warning is generated to ensure the user knows this has happened.
//
// Detects an erratum in UT699 LEON 3 processor

char DetectRoundChange::ID = 0;

DetectRoundChange::DetectRoundChange() : LEONMachineFunctionPass(ID) {}

bool DetectRoundChange::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<SparcSubtarget>();
  if (!Subtarget->detectRoundChange())
    return false;

  bool Modified = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      unsigned Opcode = MI.getOpcode();
      if (Opcode == SP::CALL && MI.getNumOperands() > 0) {
        MachineOperand &MO = MI.getOperand(0);

        if (MO.isGlobal()) {
          StringRef FuncName = MO.getGlobal()->getName();
          if (FuncName.compare_insensitive("fesetround") == 0) {
            errs() << "Error: You are using the detectroundchange "
                      "option to detect rounding changes that will "
                      "cause LEON errata. The only way to fix this "
                      "is to remove the call to fesetround from "
                      "the source code.\n";
          }
        }
      }
    }
  }

  return Modified;
}

//*****************************************************************************
//**** FixAllFDIVSQRT pass
//*****************************************************************************
// This pass fixes the incorrectly working FDIVx and FSQRTx instructions that
// exist for some earlier versions of the LEON processor line. Five NOP
// instructions need to be inserted after these instructions to ensure the
// correct result is placed in the destination registers before they are used.
//
// This pass implements two fixes:
//  1) fixing the FSQRTS and FSQRTD instructions.
//  2) fixing the FDIVS and FDIVD instructions.
//
// FSQRTS and FDIVS are converted to FDIVD and FSQRTD respectively earlier in
// the pipeline when this option is enabled, so this pass needs only to deal
// with the changes that still need implementing for the "double" versions
// of these instructions.
//
char FixAllFDIVSQRT::ID = 0;

FixAllFDIVSQRT::FixAllFDIVSQRT() : LEONMachineFunctionPass(ID) {}

bool FixAllFDIVSQRT::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<SparcSubtarget>();
  if (!Subtarget->fixAllFDIVSQRT())
    return false;

  const TargetInstrInfo &TII = *Subtarget->getInstrInfo();
  DebugLoc DL = DebugLoc();

  bool Modified = false;
  for (MachineBasicBlock &MBB : MF) {
    for (auto MBBI = MBB.begin(), E = MBB.end(); MBBI != E; ++MBBI) {
      MachineInstr &MI = *MBBI;
      unsigned Opcode = MI.getOpcode();

      // Note: FDIVS and FSQRTS cannot be generated when this erratum fix is
      // switched on so we don't need to check for them here. They will
      // already have been converted to FSQRTD or FDIVD earlier in the
      // pipeline.
      if (Opcode == SP::FSQRTD || Opcode == SP::FDIVD) {
        for (int InsertedCount = 0; InsertedCount < 5; InsertedCount++)
          BuildMI(MBB, MBBI, DL, TII.get(SP::NOP));

        MachineBasicBlock::iterator NMBBI = std::next(MBBI);
        for (int InsertedCount = 0; InsertedCount < 28; InsertedCount++)
          BuildMI(MBB, NMBBI, DL, TII.get(SP::NOP));

        Modified = true;
      }
    }
  }

  return Modified;
}
