//===- Localizer.cpp ---------------------- Localize some instrs -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the Localizer class.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/Localizer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "localizer"

using namespace llvm;

char Localizer::ID = 0;
INITIALIZE_PASS_BEGIN(Localizer, DEBUG_TYPE,
                      "Move/duplicate certain instructions close to their use",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(Localizer, DEBUG_TYPE,
                    "Move/duplicate certain instructions close to their use",
                    false, false)

Localizer::Localizer(std::function<bool(const MachineFunction &)> F)
    : MachineFunctionPass(ID), DoNotRunPass(F) {}

Localizer::Localizer()
    : Localizer([](const MachineFunction &) { return false; }) {}

void Localizer::init(MachineFunction &MF) {
  MRI = &MF.getRegInfo();
  TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(MF.getFunction());
}

bool Localizer::shouldLocalize(const MachineInstr &MI) {
  // Assuming a spill and reload of a value has a cost of 1 instruction each,
  // this helper function computes the maximum number of uses we should consider
  // for remat. E.g. on arm64 global addresses take 2 insts to materialize. We
  // break even in terms of code size when the original MI has 2 users vs
  // choosing to potentially spill. Any more than 2 users we we have a net code
  // size increase. This doesn't take into account register pressure though.
  auto maxUses = [](unsigned RematCost) {
    // A cost of 1 means remats are basically free.
    if (RematCost == 1)
      return UINT_MAX;
    if (RematCost == 2)
      return 2U;

    // Remat is too expensive, only sink if there's one user.
    if (RematCost > 2)
      return 1U;
    llvm_unreachable("Unexpected remat cost");
  };

  // Helper to walk through uses and terminate if we've reached a limit. Saves
  // us spending time traversing uses if all we want to know is if it's >= min.
  auto isUsesAtMost = [&](unsigned Reg, unsigned MaxUses) {
    unsigned NumUses = 0;
    auto UI = MRI->use_instr_nodbg_begin(Reg), UE = MRI->use_instr_nodbg_end();
    for (; UI != UE && NumUses < MaxUses; ++UI) {
      NumUses++;
    }
    // If we haven't reached the end yet then there are more than MaxUses users.
    return UI == UE;
  };

  switch (MI.getOpcode()) {
  default:
    return false;
  // Constants-like instructions should be close to their users.
  // We don't want long live-ranges for them.
  case TargetOpcode::G_CONSTANT:
  case TargetOpcode::G_FCONSTANT:
  case TargetOpcode::G_FRAME_INDEX:
  case TargetOpcode::G_INTTOPTR:
    return true;
  case TargetOpcode::G_GLOBAL_VALUE: {
    unsigned RematCost = TTI->getGISelRematGlobalCost();
    Register Reg = MI.getOperand(0).getReg();
    unsigned MaxUses = maxUses(RematCost);
    if (MaxUses == UINT_MAX)
      return true; // Remats are "free" so always localize.
    bool B = isUsesAtMost(Reg, MaxUses);
    return B;
  }
  }
}

void Localizer::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetTransformInfoWrapperPass>();
  getSelectionDAGFallbackAnalysisUsage(AU);
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool Localizer::isLocalUse(MachineOperand &MOUse, const MachineInstr &Def,
                           MachineBasicBlock *&InsertMBB) {
  MachineInstr &MIUse = *MOUse.getParent();
  InsertMBB = MIUse.getParent();
  if (MIUse.isPHI())
    InsertMBB = MIUse.getOperand(MIUse.getOperandNo(&MOUse) + 1).getMBB();
  return InsertMBB == Def.getParent();
}

bool Localizer::localizeInterBlock(MachineFunction &MF,
                                   LocalizedSetVecT &LocalizedInstrs) {
  bool Changed = false;
  DenseMap<std::pair<MachineBasicBlock *, unsigned>, unsigned> MBBWithLocalDef;

  // Since the IRTranslator only emits constants into the entry block, and the
  // rest of the GISel pipeline generally emits constants close to their users,
  // we only localize instructions in the entry block here. This might change if
  // we start doing CSE across blocks.
  auto &MBB = MF.front();
  for (auto RI = MBB.rbegin(), RE = MBB.rend(); RI != RE; ++RI) {
    MachineInstr &MI = *RI;
    if (!shouldLocalize(MI))
      continue;
    LLVM_DEBUG(dbgs() << "Should localize: " << MI);
    assert(MI.getDesc().getNumDefs() == 1 &&
           "More than one definition not supported yet");
    Register Reg = MI.getOperand(0).getReg();
    // Check if all the users of MI are local.
    // We are going to invalidation the list of use operands, so we
    // can't use range iterator.
    for (auto MOIt = MRI->use_begin(Reg), MOItEnd = MRI->use_end();
         MOIt != MOItEnd;) {
      MachineOperand &MOUse = *MOIt++;
      // Check if the use is already local.
      MachineBasicBlock *InsertMBB;
      LLVM_DEBUG(MachineInstr &MIUse = *MOUse.getParent();
                 dbgs() << "Checking use: " << MIUse
                        << " #Opd: " << MIUse.getOperandNo(&MOUse) << '\n');
      if (isLocalUse(MOUse, MI, InsertMBB))
        continue;
      LLVM_DEBUG(dbgs() << "Fixing non-local use\n");
      Changed = true;
      auto MBBAndReg = std::make_pair(InsertMBB, Reg);
      auto NewVRegIt = MBBWithLocalDef.find(MBBAndReg);
      if (NewVRegIt == MBBWithLocalDef.end()) {
        // Create the localized instruction.
        MachineInstr *LocalizedMI = MF.CloneMachineInstr(&MI);
        LocalizedInstrs.insert(LocalizedMI);
        MachineInstr &UseMI = *MOUse.getParent();
        if (MRI->hasOneUse(Reg) && !UseMI.isPHI())
          InsertMBB->insert(InsertMBB->SkipPHIsAndLabels(UseMI), LocalizedMI);
        else
          InsertMBB->insert(InsertMBB->SkipPHIsAndLabels(InsertMBB->begin()),
                            LocalizedMI);

        // Set a new register for the definition.
        Register NewReg = MRI->createGenericVirtualRegister(MRI->getType(Reg));
        MRI->setRegClassOrRegBank(NewReg, MRI->getRegClassOrRegBank(Reg));
        LocalizedMI->getOperand(0).setReg(NewReg);
        NewVRegIt =
            MBBWithLocalDef.insert(std::make_pair(MBBAndReg, NewReg)).first;
        LLVM_DEBUG(dbgs() << "Inserted: " << *LocalizedMI);
      }
      LLVM_DEBUG(dbgs() << "Update use with: " << printReg(NewVRegIt->second)
                        << '\n');
      // Update the user reg.
      MOUse.setReg(NewVRegIt->second);
    }
  }
  return Changed;
}

bool Localizer::localizeIntraBlock(LocalizedSetVecT &LocalizedInstrs) {
  bool Changed = false;

  // For each already-localized instruction which has multiple users, then we
  // scan the block top down from the current position until we hit one of them.

  // FIXME: Consider doing inst duplication if live ranges are very long due to
  // many users, but this case may be better served by regalloc improvements.

  for (MachineInstr *MI : LocalizedInstrs) {
    Register Reg = MI->getOperand(0).getReg();
    MachineBasicBlock &MBB = *MI->getParent();
    // All of the user MIs of this reg.
    SmallPtrSet<MachineInstr *, 32> Users;
    for (MachineInstr &UseMI : MRI->use_nodbg_instructions(Reg)) {
      if (!UseMI.isPHI())
        Users.insert(&UseMI);
    }
    // If all the users were PHIs then they're not going to be in our block,
    // don't try to move this instruction.
    if (Users.empty())
      continue;

    MachineBasicBlock::iterator II(MI);
    ++II;
    while (II != MBB.end() && !Users.count(&*II))
      ++II;

    LLVM_DEBUG(dbgs() << "Intra-block: moving " << *MI << " before " << *&*II
                      << "\n");
    assert(II != MBB.end() && "Didn't find the user in the MBB");
    MI->removeFromParent();
    MBB.insert(II, MI);
    Changed = true;
  }
  return Changed;
}

bool Localizer::runOnMachineFunction(MachineFunction &MF) {
  // If the ISel pipeline failed, do not bother running that pass.
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;

  // Don't run the pass if the target asked so.
  if (DoNotRunPass(MF))
    return false;

  LLVM_DEBUG(dbgs() << "Localize instructions for: " << MF.getName() << '\n');

  init(MF);

  // Keep track of the instructions we localized. We'll do a second pass of
  // intra-block localization to further reduce live ranges.
  LocalizedSetVecT LocalizedInstrs;

  bool Changed = localizeInterBlock(MF, LocalizedInstrs);
  Changed |= localizeIntraBlock(LocalizedInstrs);
  return Changed;
}
