//===- ARMLoadStoreOptimizer.cpp - nanoMIPS load / store opt. pass --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a pass that performs load / store related peephole
/// optimizations. This pass should be run after register allocation.
//
//===----------------------------------------------------------------------===//

#include "Mips.h"
#include "MipsSubtarget.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include <cmath>

using namespace llvm;

#define NM_LOAD_STORE_OPT_NAME "nanoMIPS load/store optimization pass"

namespace {
struct NMLoadStoreOpt : public MachineFunctionPass {
  using InstrList = SmallVector<MachineInstr *, 11>;
  using MBBIter = MachineBasicBlock::iterator;
  static char ID;
  const MipsSubtarget *STI;
  const TargetInstrInfo *TII;
  const std::unordered_map<unsigned, unsigned> CalleeSaves{
      {Mips::GP_NM, 0}, {Mips::FP_NM, 1}, {Mips::RA_NM, 2},  {Mips::S0_NM, 3},
      {Mips::S1_NM, 4}, {Mips::S2_NM, 5}, {Mips::S3_NM, 6},  {Mips::S4_NM, 7},
      {Mips::S5_NM, 8}, {Mips::S6_NM, 9}, {Mips::S7_NM, 10},
  };

  NMLoadStoreOpt() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return NM_LOAD_STORE_OPT_NAME; }
  bool runOnMachineFunction(MachineFunction &Fn) override;
  bool isReturn(MachineInstr &MI);
  bool isStackPointerAdjustment(MachineInstr &MI, bool IsRestore);
  bool isCalleeSavedLoadStore(MachineInstr &MI, bool IsRestore);
  void sortCalleeSavedLoadStoreList(InstrList &LoadStoreList);
  bool isSequenceValid(InstrList &StoreSequence);
  bool isValidSaveRestore16Offset(int64_t Offset);
  bool generateSaveOrRestore(MachineBasicBlock &MBB, bool IsRestore);
};
} // namespace

char NMLoadStoreOpt::ID = 0;

bool NMLoadStoreOpt::runOnMachineFunction(MachineFunction &Fn) {
  STI = &static_cast<const MipsSubtarget &>(Fn.getSubtarget());
  TII = STI->getInstrInfo();
  bool Modified = false;
  for (MachineFunction::iterator MFI = Fn.begin(), E = Fn.end(); MFI != E;
       ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    if (MBB.isEntryBlock())
      Modified |= generateSaveOrRestore(MBB, /*IsRestore=*/false);
    if (MBB.isReturnBlock())
      Modified |= generateSaveOrRestore(MBB, /*IsRestore=*/true);
  }

  return Modified;
}

bool NMLoadStoreOpt::isStackPointerAdjustment(MachineInstr &MI,
                                              bool IsRestore) {
  if (MI.getOpcode() != Mips::ADDiu_NM)
    return false;
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  if (DstReg != Mips::SP_NM || SrcReg != Mips::SP_NM)
    return false;
  int64_t Imm = MI.getOperand(2).getImm();
  if (IsRestore && Imm <= 0)
    return false;
  if (!IsRestore && Imm >= 0)
    return false;
  Imm = std::abs(Imm);
  // Adjustment has to be doubleword aligned.
  if (Imm & 0x7)
    return false;
  return true;
}

bool NMLoadStoreOpt::isCalleeSavedLoadStore(MachineInstr &MI, bool IsRestore) {
  unsigned Opcode = MI.getOpcode();
  if (IsRestore) {
    if (Opcode != Mips::LW_NM && Opcode != Mips::LWs9_NM)
      return false;
  } else {
    if (Opcode != Mips::SW_NM && Opcode != Mips::SWs9_NM)
      return false;
  }

  Register BaseReg = MI.getOperand(1).getReg();
  if (BaseReg != Mips::SP_NM)
    return false;

  // We care only for callee-saved registers.
  Register SrcReg = MI.getOperand(0).getReg();
  if (CalleeSaves.find(SrcReg) == CalleeSaves.end())
    return false;

  return true;
}

bool NMLoadStoreOpt::isReturn(MachineInstr &MI) {
  unsigned Opcode = MI.getOpcode();
  if (Opcode != Mips::PseudoReturnNM)
    return false;

  Register BaseReg = MI.getOperand(0).getReg();
  if (BaseReg != Mips::RA_NM)
    return false;

  return true;
}

void NMLoadStoreOpt::sortCalleeSavedLoadStoreList(InstrList &LoadStoreList) {
  // nanoMIPS save and restore instructions require callee-saved registers to be
  // saved in particular order on the stack. This sorts the list so that
  // registers are in correct order. But it is still necessary to check if the
  // offsets are valid (this requires sorted list).
  auto CompareInstructions = [this](MachineInstr *First, MachineInstr *Second) {
    Register FirstReg = First->getOperand(0).getReg();
    Register SecondReg = Second->getOperand(0).getReg();
    return CalleeSaves.at(FirstReg) < CalleeSaves.at(SecondReg);
  };
  std::sort(LoadStoreList.begin(), LoadStoreList.end(), CompareInstructions);
}

bool NMLoadStoreOpt::isSequenceValid(InstrList &LoadStoreList) {
  int InsNo = 1;
  for (auto *MII = LoadStoreList.begin(); MII != LoadStoreList.end() - 1;
       MII++, InsNo++) {
    // Make sure that all offsets are 4 bytes apart.
    int64_t CurrOffset = (*MII)->getOperand(2).getImm();
    int64_t NextOffset = (*std::next(MII))->getOperand(2).getImm();
    if (CurrOffset != NextOffset + 4)
      return false;

    // Make sure that there is no gaps between registers.
    int64_t CurrReg = (*MII)->getOperand(0).getReg();
    int64_t NextReg = (*std::next(MII))->getOperand(0).getReg();
    if (CalleeSaves.at(CurrReg) != CalleeSaves.at(NextReg) - 1)
      return false;
  }
  return true;
}

bool NMLoadStoreOpt::isValidSaveRestore16Offset(int64_t Offset) {
  return (Offset <= 240) && !(Offset & 0xf);
}

// Generates save or restore instruction.
//
// addiu $sp, $sp, -16  ->  save 16, $s1-$s4 // sw $s0, 12($sp)      ->
// sw $s1, 8($sp)       ->
// sw $s2, 4($sp)       ->
// sw $s3, 0($sp)       ->
//
// or:
//
// lw $s3, 0($sp)       -> restore.jrc 16, $s0-$s3
// lw $s2, 4($sp)       ->
// lw $s1, 8($sp)       ->
// lw $s0, 12($sp)      ->
// addiu $sp, $sp, 16   ->
// jrc $ra              ->
//
bool NMLoadStoreOpt::generateSaveOrRestore(MachineBasicBlock &MBB,
                                           bool IsRestore) {
  // Look for contiguous list of loads/stores.
  InstrList LoadStoreList;
  MachineInstr *AdjustStack = nullptr;
  MachineInstr *Return = nullptr;
  bool SequenceStarted = false;

  if (IsRestore) {
    // Iterate bacbwards over BB in case were looking to generate restore,
    // because those instructions are at the end of the BB.
    for (auto &MI : make_range(MBB.rbegin(), MBB.rend())) {
      if (isReturn(MI)) {
        Return = &MI;
        SequenceStarted = true;
        continue;
      }
      if (isStackPointerAdjustment(MI, IsRestore)) {
        assert(SequenceStarted && Return);
        AdjustStack = &MI;
        continue;
      }
      // Since we are looking for a contguous list, we should stop searching for
      // more loads once the end of the list has been reached. Both return and
      // stack adjustment should be found by now, since we're iterating from the
      // end.
      if (isCalleeSavedLoadStore(MI, IsRestore)) {
        assert(SequenceStarted && Return && AdjustStack);
        LoadStoreList.emplace_back(&MI);
        continue;
      }

      // CFI instructions don't break the sequence.
      if (MI.isCFIInstruction())
        continue;

      // Sequence has been broken, no need to continue. We either reached the
      // end or found nothing.
      if (SequenceStarted)
        break;
    }
  } else {
    for (auto &MI : MBB) {
      if (isStackPointerAdjustment(MI, IsRestore)) {
        SequenceStarted = true;
        AdjustStack = &MI;
        continue;
      }
      // Since we are looking for a contguous list, we should stop searching for
      // more stores once the end of the list has been reached.
      if (isCalleeSavedLoadStore(MI, IsRestore)) {
        assert(SequenceStarted && AdjustStack);
        LoadStoreList.emplace_back(&MI);
        continue;
      }

      // CFI instructions don't break the sequence.
      if (MI.isCFIInstruction())
        continue;

      // Sequence has been broken, no need to continue. We either reached the
      // end or found nothing.
      if (SequenceStarted)
        break;
    }
  }

  if (AdjustStack) {
    int64_t StackOffset = std::abs(AdjustStack->getOperand(2).getImm());
    // Stack offset has to be doubleword aligned and cannot be larger than 4092.
    // TODO: In case it is larger than 4092, we could emit addiu + save.
    if (StackOffset > 4092 || StackOffset & 0x7)
      return false;

    sortCalleeSavedLoadStoreList(LoadStoreList);

    bool IsValidList = !LoadStoreList.empty() && isSequenceValid(LoadStoreList);

    // Save/restore instructions operate on the beginning of SP, but sometimes
    // that is allocated for function arguments. In that case, it is
    // neccessary to emit additional addiu (or save/restore) which adjusts stack
    // pointer to the place where callee-saves should be.
    //
    // addiu $sp, $sp, -32  ->  save 16
    // sw $s0, 12($sp)      ->  save 16, $s0-$s3
    // sw $s1, 8($sp)       ->
    // sw $s2, 4($sp)       ->
    // sw $s3, 0($sp)       ->
    //
    int64_t NewStackOffset = 0;
    if (IsValidList) {
      auto LastOffset = LoadStoreList.front()->getOperand(2).getImm();
      assert(StackOffset >= LastOffset + 4);
      if (StackOffset > LastOffset + 4) {
        // LastOffset + 4 will be the new stack offset and needs to be
        // doubleword aligned. In case it is not, we need to remove an element
        // from the list in order to ensure the alignment.
        //
        // addiu $sp, $sp, -32  ->  save 16
        // sw $s0, 16($sp)      ->  save 16, $s1-$s4
        // sw $s1, 12($sp)      ->  sw $s0, 16($sp)
        // sw $s2, 8($sp)       ->
        // sw $s3, 4($sp)       ->
        // sw $s4, 0($sp)       ->
        //
        if ((LastOffset + 4) & 0x7) {
          LoadStoreList.erase(LoadStoreList.begin());
          // Since an element has been removed, it is neccessary to check
          // validity again.
          IsValidList = !LoadStoreList.empty();
          if (IsValidList) {
            LastOffset = LoadStoreList.front()->getOperand(2).getImm();
            assert(!((LastOffset + 4) & 0x7));
            NewStackOffset = StackOffset - LastOffset - 4;
            StackOffset = LastOffset + 4;
          }
        } else {
          NewStackOffset = StackOffset - LastOffset - 4;
          StackOffset = LastOffset + 4;
        }
      }
    }

    if (!IsValidList) {
      // Generate 16-bit save/restore if there are no register arguments or
      // register arguments are invalid. They have 8-bit offset with quadword
      // alignment.
      if (!isValidSaveRestore16Offset(StackOffset))
        return false;
      LoadStoreList.clear();
    }

    // We cannot generate restore.jrc if NewStackOffset is set. This is because
    // we first need to emit restore and after an additional addiu.
    unsigned Opcode = IsRestore
                          ? ((Return && !NewStackOffset) ? Mips::RESTOREJRC_NM
                                                         : Mips::RESTORE_NM)
                          : Mips::SAVE_NM;
    auto InsertBefore = std::next(MBBIter(AdjustStack));
    auto DL = Return ? Return->getDebugLoc() : AdjustStack->getDebugLoc();
    auto MII = BuildMI(MBB, InsertBefore, DL, TII->get(Opcode));
    MII.addImm(StackOffset);
    MBB.erase(AdjustStack);
    if (Return && !NewStackOffset)
      MBB.erase(Return);

    for (size_t InsNo = 0; InsNo < LoadStoreList.size(); InsNo++) {
      MachineInstr *MI = LoadStoreList[InsNo];
      MII.addReg(MI->getOperand(0).getReg(),
                 IsRestore ? RegState::Define : RegState::Kill);
      MBB.erase(MI);
    }

    if (NewStackOffset) {
      // NewStackOffset needs to be added before save, but after restore. This
      // is why iterator needs to go to the next iteration in case of restore.
      InsertBefore = MBBIter(MII.getInstr());
      if (IsRestore)
        InsertBefore = std::next(InsertBefore);
      // Favorable case is to generate save/restore (16-bit), but we have to
      // make sure the offset fits. Otherwise, we fall back to addiu (32-bit).
      if (isValidSaveRestore16Offset(NewStackOffset)) {
        if (Return) {
          BuildMI(MBB, InsertBefore, DL, TII->get(Mips::RESTOREJRC_NM))
              .addImm(NewStackOffset);
          MBB.erase(Return);
        } else {
          // Opcode can be safely reused.
          BuildMI(MBB, InsertBefore, DL, TII->get(Opcode))
              .addImm(NewStackOffset);
        }
      } else {
        // In case of save, the offset is subtracted from SP.
        if (!IsRestore)
          NewStackOffset = -NewStackOffset;
        BuildMI(MBB, InsertBefore, DL, TII->get(Mips::ADDiu_NM), Mips::SP_NM)
            .addReg(Mips::SP_NM)
            .addImm(NewStackOffset);
      }
    }
    return true;
  }
  return false;
}

namespace llvm {
FunctionPass *createNanoMipsLoadStoreOptimizerPass() {
  return new NMLoadStoreOpt();
}
} // namespace llvm
