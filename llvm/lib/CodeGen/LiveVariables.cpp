//===-- LiveVariables.cpp - Live Variable Analysis for Machine Code -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveVariable pass. For each machine instruction in
// the function, this pass marks the registers that are immediately dead after
// the instruction (i.e., the instruction calculates the value, but it is never
// used). It does not compute kill flags or any queryable liveness information.
//
// A virtual register def is dead if the register has no reading uses. Physical
// registers are assumed to only be live within a single basic block, and are
// resolved with a local analysis of each block. This also adds implicit defs of
// sub-registers and super-registers to model partially dead physical register
// definitions. Reserved physical registers are not tracked.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

AnalysisKey LiveVariablesAnalysis::Key;

LiveVariablesAnalysis::Result
LiveVariablesAnalysis::run(MachineFunction &MF,
                           MachineFunctionAnalysisManager &) {
  return Result(MF);
}

char LiveVariablesWrapperPass::ID = 0;
char &llvm::LiveVariablesID = LiveVariablesWrapperPass::ID;
INITIALIZE_PASS_BEGIN(LiveVariablesWrapperPass, "livevars",
                      "Live Variable Analysis", false, false)
INITIALIZE_PASS_DEPENDENCY(UnreachableMachineBlockElimLegacy)
INITIALIZE_PASS_END(LiveVariablesWrapperPass, "livevars",
                    "Live Variable Analysis", false, false)

void LiveVariablesWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequiredID(UnreachableMachineBlockElimID);
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

LiveVariables::LiveVariables(MachineFunction &MF) { analyze(MF); }

/// FindLastPartialDef - Return the last partial def of the specified register.
MachineInstr *LiveVariables::FindLastPartialDef(Register Reg) {
  unsigned LastDefDist = 0;
  MachineInstr *LastDef = nullptr;
  for (MCPhysReg SubReg : TRI->subregs(Reg)) {
    MachineInstr *Def = PhysRegDef[SubReg];
    if (!Def)
      continue;
    unsigned Dist = DistanceMap[Def];
    if (Dist > LastDefDist) {
      LastDef     = Def;
      LastDefDist = Dist;
    }
  }

  return LastDef;
}

/// HandlePhysRegUse - Turn previous partial def's into read/mod/writes. Add
/// implicit defs to a machine instruction if there was an earlier def of its
/// super-register.
void LiveVariables::HandlePhysRegUse(Register Reg, MachineInstr &MI) {
  MachineInstr *LastDef = PhysRegDef[Reg.id()];
  // If there was a previous use or a "full" def all is well.
  if (!LastDef && !PhysRegUse[Reg.id()]) {
    // Otherwise, the last sub-register def implicitly defines this register.
    // e.g.
    // AH =
    // AL = ... implicit-def EAX, implicit killed AH
    //    = AH
    // ...
    //    = EAX
    // All of the sub-registers must have been defined before the use of Reg!
    MachineInstr *LastPartialDef = FindLastPartialDef(Reg);
    // If LastPartialDef is NULL, it must be using a livein register.
    if (LastPartialDef) {
      LastPartialDef->addOperand(
          MachineOperand::CreateReg(Reg, /*IsDef=*/true, /*IsImp=*/true));
    }
  } else if (LastDef && !PhysRegUse[Reg.id()] &&
             !LastDef->findRegisterDefOperand(Reg, /*TRI=*/nullptr))
    // Last def defines the super register, add an implicit def of reg.
    LastDef->addOperand(MachineOperand::CreateReg(Reg, true/*IsDef*/,
                                                  true/*IsImp*/));

  // Remember this use.
  for (MCPhysReg SubReg : TRI->subregs_inclusive(Reg))
    PhysRegUse[SubReg] = &MI;
}

/// FindLastRefOrPartRef - Return the last reference or partial reference of
/// the specified register.
MachineInstr *LiveVariables::FindLastRefOrPartRef(Register Reg) {
  MachineInstr *LastDef = PhysRegDef[Reg.id()];
  MachineInstr *LastUse = PhysRegUse[Reg.id()];
  if (!LastDef && !LastUse)
    return nullptr;

  MachineInstr *LastRefOrPartRef = LastUse ? LastUse : LastDef;
  unsigned LastRefOrPartRefDist = DistanceMap[LastRefOrPartRef];
  for (MCPhysReg SubReg : TRI->subregs(Reg)) {
    MachineInstr *Def = PhysRegDef[SubReg];
    if (Def && Def != LastDef)
      continue;
    if (MachineInstr *Use = PhysRegUse[SubReg]) {
      unsigned Dist = DistanceMap[Use];
      if (Dist > LastRefOrPartRefDist) {
        LastRefOrPartRefDist = Dist;
        LastRefOrPartRef = Use;
      }
    }
  }

  return LastRefOrPartRef;
}

void LiveVariables::HandlePhysRegKill(Register Reg, MachineInstr *MI) {
  MachineInstr *LastDef = PhysRegDef[Reg.id()];
  MachineInstr *LastUse = PhysRegUse[Reg.id()];
  if (!LastDef && !LastUse)
    return;

  MachineInstr *LastRefOrPartRef = LastUse ? LastUse : LastDef;
  unsigned LastRefOrPartRefDist = DistanceMap[LastRefOrPartRef];
  // The whole register is used.
  // AL =
  // AH =
  //
  //    = AX
  //    = AL, implicit killed AX
  // AX =
  //
  // Or whole register is defined, but not used at all.
  // dead AX =
  // ...
  // AX =
  //
  // Or whole register is defined, but only partly used.
  // dead AX = implicit-def AL
  //    = killed AL
  // AX =
  MachineInstr *LastPartDef = nullptr;
  unsigned LastPartDefDist = 0;
  SmallSet<unsigned, 8> PartUses;
  for (MCPhysReg SubReg : TRI->subregs(Reg)) {
    MachineInstr *Def = PhysRegDef[SubReg];
    if (Def && Def != LastDef) {
      // There was a def of this sub-register in between. This is a partial
      // def, keep track of the last one.
      unsigned Dist = DistanceMap[Def];
      if (Dist > LastPartDefDist) {
        LastPartDefDist = Dist;
        LastPartDef = Def;
      }
      continue;
    }
    if (MachineInstr *Use = PhysRegUse[SubReg]) {
      PartUses.insert_range(TRI->subregs_inclusive(SubReg));
      unsigned Dist = DistanceMap[Use];
      if (Dist > LastRefOrPartRefDist) {
        LastRefOrPartRefDist = Dist;
        LastRefOrPartRef = Use;
      }
    }
  }

  if (!PhysRegUse[Reg.id()]) {
    // Partial uses. Mark register def dead and add implicit def of
    // sub-registers which are used.
    // dead EAX  = op  implicit-def AL
    // That is, EAX def is dead but AL def extends pass it.
    PhysRegDef[Reg.id()]->addRegisterDead(Reg, TRI, true);
    for (MCPhysReg SubReg : TRI->subregs(Reg)) {
      if (!PartUses.count(SubReg))
        continue;
      bool NeedDef = true;
      if (PhysRegDef[Reg.id()] == PhysRegDef[SubReg]) {
        MachineOperand *MO = PhysRegDef[Reg.id()]->findRegisterDefOperand(
            SubReg, /*TRI=*/nullptr);
        if (MO) {
          NeedDef = false;
          assert(!MO->isDead());
        }
      }
      if (NeedDef)
        PhysRegDef[Reg.id()]->addOperand(
            MachineOperand::CreateReg(SubReg, true /*IsDef*/, true /*IsImp*/));
      if (!FindLastRefOrPartRef(SubReg)) {
        for (MCPhysReg SS : TRI->subregs_inclusive(SubReg))
          PhysRegUse[SS] = LastRefOrPartRef;
      }
      for (MCPhysReg SS : TRI->subregs(SubReg))
        PartUses.erase(SS);
    }
  } else if (LastRefOrPartRef == PhysRegDef[Reg.id()] &&
             LastRefOrPartRef != MI && !LastPartDef) {
    MachineOperand *MO =
        LastRefOrPartRef->findRegisterDefOperand(Reg, TRI, false, false);
    bool NeedEC = MO->isEarlyClobber() && MO->getReg() != Reg;
    // If the last reference is the last def, then it's not used at all.
    // That is, unless we are currently processing the last reference itself.
    LastRefOrPartRef->addRegisterDead(Reg, TRI, true);
    if (NeedEC) {
      // If we are adding a subreg def and the superreg def is marked early
      // clobber, add an early clobber marker to the subreg def.
      MO = LastRefOrPartRef->findRegisterDefOperand(Reg, /*TRI=*/nullptr);
      if (MO)
        MO->setIsEarlyClobber();
    }
  }
}

void LiveVariables::HandleRegMask(const MachineOperand &MO, unsigned NumRegs) {
  // Call HandlePhysRegKill() for all live registers clobbered by Mask.
  // Clobbered registers are always dead, sp there is no need to use
  // HandlePhysRegDef().
  for (unsigned Reg = 1; Reg != NumRegs; ++Reg) {
    // Skip dead regs.
    if (!PhysRegDef[Reg] && !PhysRegUse[Reg])
      continue;
    // Skip mask-preserved regs.
    if (!MO.clobbersPhysReg(Reg))
      continue;
    // Kill the largest clobbered super-register.
    // This avoids needless implicit operands.
    unsigned Super = Reg;
    for (MCPhysReg SR : TRI->superregs(Reg))
      if (SR < NumRegs && (PhysRegDef[SR] || PhysRegUse[SR]) &&
          MO.clobbersPhysReg(SR))
        Super = SR;
    HandlePhysRegKill(Super, nullptr);
  }
}

void LiveVariables::HandlePhysRegDef(Register Reg, MachineInstr *MI,
                                     SmallVectorImpl<Register> &Defs) {
  // What parts of the register are previously defined?
  SmallSet<unsigned, 32> Live;
  if (PhysRegDef[Reg.id()] || PhysRegUse[Reg.id()]) {
    Live.insert_range(TRI->subregs_inclusive(Reg));
  } else {
    for (MCPhysReg SubReg : TRI->subregs(Reg)) {
      // If a register isn't itself defined, but all parts that make up of it
      // are defined, then consider it also defined.
      // e.g.
      // AL =
      // AH =
      //    = AX
      if (Live.count(SubReg))
        continue;
      if (PhysRegDef[SubReg] || PhysRegUse[SubReg])
        Live.insert_range(TRI->subregs_inclusive(SubReg));
    }
  }

  // Start from the largest piece, find the last time any part of the register
  // is referenced.
  HandlePhysRegKill(Reg, MI);
  // Only some of the sub-registers are used.
  for (MCPhysReg SubReg : TRI->subregs(Reg)) {
    if (!Live.count(SubReg))
      // Skip if this sub-register isn't defined.
      continue;
    HandlePhysRegKill(SubReg, MI);
  }

  if (MI)
    Defs.push_back(Reg);  // Remember this def.
}

void LiveVariables::UpdatePhysRegDefs(MachineInstr &MI,
                                      SmallVectorImpl<Register> &Defs) {
  while (!Defs.empty()) {
    Register Reg = Defs.pop_back_val();
    for (MCPhysReg SubReg : TRI->subregs_inclusive(Reg)) {
      PhysRegDef[SubReg] = &MI;
      PhysRegUse[SubReg]  = nullptr;
    }
  }
}

void LiveVariables::runOnInstr(MachineInstr &MI,
                               SmallVectorImpl<Register> &Defs,
                               unsigned NumRegs) {
  assert(!MI.isDebugOrPseudoInstr());

  // Clear dead markers. LV will recompute them.
  SmallVector<Register, 4> UseRegs;
  SmallVector<Register, 4> DefRegs;
  SmallVector<unsigned, 1> RegMasks;
  for (auto [I, MO] : enumerate(MI.operands())) {
    if (MO.isRegMask()) {
      RegMasks.push_back(I);
      continue;
    }
    if (!MO.isReg())
      continue;
    Register MOReg = MO.getReg();
    if (!MOReg.isPhysical() || MRI->isReserved(MOReg))
      continue;
    if (MO.isUse()) {
      if (MO.readsReg())
        UseRegs.push_back(MOReg);
    } else {
      // FIXME: We should not remove any dead flags. However the MIPS RDDSP
      // instruction needs it at the moment: http://llvm.org/PR27116.
      MO.setIsDead(false);
      DefRegs.push_back(MOReg);
    }
  }

  // Process all uses.
  for (Register MOReg : UseRegs)
    HandlePhysRegUse(MOReg, MI);

  // Process all masked registers. (Call clobbers).
  for (unsigned Mask : RegMasks)
    HandleRegMask(MI.getOperand(Mask), NumRegs);

  // Process all defs.
  for (Register MOReg : DefRegs)
    HandlePhysRegDef(MOReg, &MI, Defs);
  UpdatePhysRegDefs(MI, Defs);
}

void LiveVariables::runOnBlock(MachineBasicBlock *MBB, unsigned NumRegs) {
  // Mark live-in registers as live-in.
  SmallVector<Register, 4> Defs;
  for (const auto &LI : MBB->liveins()) {
    assert(LI.PhysReg.isPhysical() &&
           "Cannot have a live-in virtual register!");
    HandlePhysRegDef(LI.PhysReg, nullptr, Defs);
  }

  // Loop over all of the instructions, processing them.
  DistanceMap.clear();
  unsigned Dist = 0;
  for (MachineInstr &MI : *MBB) {
    if (MI.isDebugOrPseudoInstr())
      continue;
    DistanceMap.insert(std::make_pair(&MI, Dist++));

    runOnInstr(MI, Defs, NumRegs);
  }

  // MachineCSE may CSE instructions which write to non-allocatable physical
  // registers across MBBs. Remember if any reserved register is liveout.
  SmallSet<MCRegister, 4> LiveOuts;
  for (const MachineBasicBlock *SuccMBB : MBB->successors()) {
    if (SuccMBB->isEHPad())
      continue;
    for (const auto &LI : SuccMBB->liveins()) {
      if (!TRI->isInAllocatableClass(LI.PhysReg))
        // Ignore other live-ins, e.g. those that are live into landing pads.
        LiveOuts.insert(LI.PhysReg);
    }
  }

  // Loop over PhysRegDef / PhysRegUse, killing any registers that are
  // available at the end of the basic block.
  for (unsigned i = 0; i != NumRegs; ++i)
    if ((PhysRegDef[i] || PhysRegUse[i]) && !LiveOuts.count(i))
      HandlePhysRegDef(i, nullptr, Defs);
}

void LiveVariables::analyze(MachineFunction &mf) {
  MRI = &mf.getRegInfo();
  TRI = mf.getSubtarget().getRegisterInfo();

  // FIXME: LiveIntervals will be updated to remove its dependence on
  // LiveVariables to improve compilation time and eliminate bizarre pass
  // dependencies. Until then, we can't change much in -O0.
  if (!MRI->isSSA())
    reportFatalUsageError("regalloc=... not currently supported with -O0");

  const unsigned NumRegs = TRI->getNumSupportedRegs(mf);
  PhysRegDef.assign(NumRegs, nullptr);
  PhysRegUse.assign(NumRegs, nullptr);

  for (MachineBasicBlock &MBB : mf) {
    runOnBlock(&MBB, NumRegs);

    PhysRegDef.assign(NumRegs, nullptr);
    PhysRegUse.assign(NumRegs, nullptr);
  }

  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    MachineInstr *Def = MRI->getVRegDef(Reg);
    if (Def && none_of(MRI->use_nodbg_operands(Reg),
                       [](const MachineOperand &MO) { return MO.readsReg(); }))
      Def->addRegisterDead(Reg, TRI);
  }

  PhysRegDef.clear();
  PhysRegUse.clear();
}
