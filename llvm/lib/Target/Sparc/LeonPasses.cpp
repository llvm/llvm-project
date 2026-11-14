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
#include "Sparc.h"
#include "SparcSubtarget.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/Analysis.h"

using namespace llvm;

char SparcErrataWorkaroundLegacy::ID = 0;

SparcErrataWorkaroundLegacy::SparcErrataWorkaroundLegacy()
    : MachineFunctionPass(ID) {
  initializeSparcErrataWorkaroundLegacyPass(*PassRegistry::getPassRegistry());
}

INITIALIZE_PASS(SparcErrataWorkaroundLegacy, "errata-workaround",
                "Errata workaround pass", false, false)

// Move iterator to the next instruction in the function, ignoring
// meta instructions and inline assembly. Returns false when reaching
// the end of the function.
static bool moveNext(MachineBasicBlock::iterator &I) {

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

static void insertNop(MachineBasicBlock::iterator I,
                      const TargetInstrInfo *TII) {
  BuildMI(*I->getParent(), I, I->getDebugLoc(), TII->get(SP::NOP));
}

static bool isFloat(MachineBasicBlock::iterator I) {
  if (I->getNumOperands() == 0)
    return false;

  if (!I->getOperand(0).isReg())
    return false;

  unsigned reg = I->getOperand(0).getReg();

  if (!SP::FPRegsRegClass.contains(reg) && !SP::DFPRegsRegClass.contains(reg))
    return false;

  return true;
}

static bool isDivSqrt(MachineBasicBlock::iterator I) {
  switch (I->getOpcode()) {
  case SP::FDIVS:
  case SP::FDIVD:
  case SP::FSQRTS:
  case SP::FSQRTD:
    return true;
  }
  return false;
}

static bool isAtomic(MachineBasicBlock::iterator I) {
  switch (I->getOpcode()) {
  case SP::SWAPrr:
  case SP::SWAPri:
  case SP::CASArr:
  case SP::LDSTUBrr:
  case SP::LDSTUBri:
  case SP::LDSTUBArr:
    return true;
  }
  return false;
}

// Prevents the following code sequence from being generated:
// (stb/sth/st/stf) -> (single non-store/load instruction) -> (any store)
// If the sequence is detected a NOP instruction is inserted after
// the first store instruction.
static bool checkSeqTN0009A(MachineBasicBlock::iterator I,
                            const TargetInstrInfo *TII) {
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

  insertNop(PatchHere, TII);
  return true;
}

// Prevents the following code sequence from being generated:
// (std/stdf) -> (any store)
// If the sequence is detected a NOP instruction is inserted after
// the first store instruction.
static bool checkSeqTN0009B(MachineBasicBlock::iterator I,
                            const TargetInstrInfo *TII) {

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

  insertNop(MI, TII);
  return true;
}

// Insert a NOP at branch target if load in delay slot and atomic
// instruction at branch target. Also insert a NOP between load
// instruction and atomic instruction (swap or casa).
static bool checkSeqTN0010(MachineBasicBlock::iterator I,
                           const TargetInstrInfo *TII) {

  // Check for load instruction or branch bundled with load instruction
  if (!I->mayLoad())
    return false;

  // Check for branch to atomic instruction with load in delay slot
  if (I->isBranch() && I->getOperand(0).isMBB()) {
    MachineBasicBlock *TargetMBB = I->getOperand(0).getMBB();
    MachineBasicBlock::iterator MI = TargetMBB->begin();

    while (MI != TargetMBB->end() && MI->isMetaInstruction())
      MI++;

    if (MI == TargetMBB->end())
      return false;

    if (isAtomic(MI))
      insertNop(MI, TII);
  }

  // Check for load followed by atomic instruction
  MachineBasicBlock::iterator MI = I;
  if (!moveNext(MI))
    return false;

  if (!isAtomic(MI))
    return false;
  insertNop(MI, TII);
  return true;
}

// Do not allow functions to begin with an atomic instruction
static bool checkSeqTN0010First(MachineBasicBlock &MBB,
                                const TargetInstrInfo *TII) {
  MachineBasicBlock::iterator I = MBB.begin();
  while (I != MBB.end() && I->isMetaInstruction())
    I++;
  if (I == MBB.end())
    return false;
  if (!isAtomic(I))
    return false;
  insertNop(I, TII);
  return true;
}

// Inserts a NOP instruction at the target of an integer branch if the
// target is a floating-point instruction or floating-point branch.
static bool checkSeqTN0012(MachineBasicBlock::iterator I,
                           const TargetInstrInfo *TII) {

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

  insertNop(MI, TII);
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
static bool checkSeqTN0013(MachineBasicBlock::iterator I,
                           const TargetInstrInfo *TII,
                           const TargetRegisterInfo *TRI) {

  if (!isDivSqrt(I))
    return false;

  unsigned dstReg = I->getOperand(0).getReg();

  MachineBasicBlock::iterator MI = I;
  if (!moveNext(MI))
    return false;

  if (MI->isBranch()) {
    insertNop(MI, TII);
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

      insertNop(PatchHere, TII);
      if (i == 2)
        insertNop(PatchHere, TII);
      return true;
    }

    fpFound++;
    if (!moveNext(MI))
      return false;
  }

  return false;
}

// On the AT697E and AT697F, a single-word load into an odd-numbered FP register
// followed by a double-precision operation on the enclosing register pair can
// give a wrong result (AT697E erratum 13, AT697F erratum 1). On the UT699, a
// single-cycle load followed by another load or store can fail.
//
// Inserts a NOP after any LD or LDF instruction.
static bool insertNOPLoad(MachineBasicBlock &MBB, const TargetInstrInfo *TII) {
  bool Modified = false;
  for (auto MBBI = MBB.begin(), E = MBB.end(); MBBI != E; ++MBBI) {
    unsigned Opcode = MBBI->getOpcode();
    if (Opcode >= SP::LDDArr && Opcode <= SP::LDrr) {
      BuildMI(MBB, std::next(MBBI), DebugLoc(), TII->get(SP::NOP));
      Modified = true;
    }
  }
  return Modified;
}

static bool runErrataWorkaroundOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;
  const SparcSubtarget *ST = &MF.getSubtarget<SparcSubtarget>();

  if (!(ST->insertNOPLoad() || ST->fixTN0009() || ST->fixTN0010() ||
        ST->fixTN0012() || ST->fixTN0013()))
    return false;

  const TargetInstrInfo *TII = ST->getInstrInfo();
  const TargetRegisterInfo *TRI = ST->getRegisterInfo();

  if (ST->insertNOPLoad())
    for (auto &MBB : MF)
      Changed |= insertNOPLoad(MBB, TII);

  if (ST->fixTN0010())
    Changed |= checkSeqTN0010First(MF.front(), TII);

  for (auto &MBB : MF) {
    for (auto &I : MBB) {
      if (ST->fixTN0009()) {
        Changed |= checkSeqTN0009A(I, TII);
        Changed |= checkSeqTN0009B(I, TII);
      }
      if (ST->fixTN0010())
        Changed |= checkSeqTN0010(I, TII);
      if (ST->fixTN0012())
        Changed |= checkSeqTN0012(I, TII);
      if (ST->fixTN0013())
        Changed |= checkSeqTN0013(I, TII, TRI);
    }
  }
  return Changed;
}

bool SparcErrataWorkaroundLegacy::runOnMachineFunction(MachineFunction &MF) {
  return runErrataWorkaroundOnMachineFunction(MF);
}

PreservedAnalyses
SparcErrataWorkaroundPass::run(MachineFunction &MF,
                               MachineFunctionAnalysisManager &MFAM) {
  return runErrataWorkaroundOnMachineFunction(MF)
             ? getMachineFunctionPassPreservedAnalyses()
                   .preserveSet<CFGAnalyses>()
             : PreservedAnalyses::all();
}
