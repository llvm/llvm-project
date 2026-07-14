//===-- SSASIFormMemoryClauses.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass is a clone of SIFormMemoryClauses intended to run in SSA
/// form, before PHI elimination. It extends the live ranges of registers used
/// as pointers in sequences of adjacent SMEM and VMEM instructions when XNACK
/// is enabled, preventing a load from overwriting a pointer and requiring a
/// soft clause break.
///
/// TODO: Once PR #161054 (SSAMachineScheduler) is merged this pass should be
/// placed immediately after SSAMachineScheduler in the pipeline.
///
//===----------------------------------------------------------------------===//

#include "SSASIFormMemoryClauses.h"
#include "AMDGPU.h"
#include "GCNRegPressure.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "ssa-si-form-memory-clauses"

// Clauses longer then 15 instructions would overflow one of the counters
// and stall. They can stall even earlier if there are outstanding counters.
static cl::opt<unsigned> SSAMaxClause(
    "amdgpu-ssa-max-memory-clause", cl::Hidden, cl::init(15),
    cl::desc("Maximum length of a memory clause for SSA form pass, "
             "instructions"));

namespace {

class SSASIFormMemoryClausesImpl {
  using RegUse = DenseMap<unsigned, std::pair<RegState, LaneBitmask>>;

  bool canBundle(const MachineInstr &MI, const RegUse &Defs,
                 const RegUse &Uses) const;
  bool checkPressure(const MachineInstr &MI, GCNRegPressure &CurPressure);
  void collectRegUses(const MachineInstr &MI, RegUse &Defs, RegUse &Uses) const;
  bool processRegUses(const MachineInstr &MI, RegUse &Defs, RegUse &Uses,
                      GCNRegPressure &CurPressure);

  const GCNSubtarget *ST;
  const SIRegisterInfo *TRI;
  const MachineRegisterInfo *MRI;
  SIMachineFunctionInfo *MFI;
  LiveVariables *LV;

  unsigned LastRecordedOccupancy;
  unsigned MaxVGPRs;
  unsigned MaxSGPRs;

public:
  bool run(MachineFunction &MF, LiveVariables &LV);
};

class SSASIFormMemoryClausesLegacy : public MachineFunctionPass {
public:
  static char ID;

  SSASIFormMemoryClausesLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SSA SI Form memory clauses";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveVariablesWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  // Unlike SIFormMemoryClauses, we do NOT clear the IsSSA property because
  // this pass is designed to run while the function is still in SSA form.
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SSASIFormMemoryClausesLegacy, DEBUG_TYPE,
                      "SSA SI Form memory clauses", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveVariablesWrapperPass)
INITIALIZE_PASS_END(SSASIFormMemoryClausesLegacy, DEBUG_TYPE,
                    "SSA SI Form memory clauses", false, false)

char SSASIFormMemoryClausesLegacy::ID = 0;

char &llvm::SSASIFormMemoryClausesID = SSASIFormMemoryClausesLegacy::ID;

FunctionPass *llvm::createSSASIFormMemoryClausesLegacyPass() {
  return new SSASIFormMemoryClausesLegacy();
}

static bool isVMEMClauseInst(const MachineInstr &MI) {
  return SIInstrInfo::isVMEM(MI);
}

static bool isSMEMClauseInst(const MachineInstr &MI) {
  return SIInstrInfo::isSMRD(MI);
}

// There no sense to create store clauses, they do not define anything,
// thus there is nothing to set early-clobber.
static bool isValidClauseInst(const MachineInstr &MI, bool IsVMEMClause) {
  assert(!MI.isDebugInstr() && "debug instructions should not reach here");
  if (MI.isBundled())
    return false;
  if (!MI.mayLoad() || MI.mayStore())
    return false;
  if (SIInstrInfo::isAtomic(MI))
    return false;
  if (IsVMEMClause && !isVMEMClauseInst(MI))
    return false;
  if (!IsVMEMClause && !isSMEMClauseInst(MI))
    return false;
  // If this is a load instruction where the result has been coalesced with an
  // operand, then we cannot clause it.
  for (const MachineOperand &ResMO : MI.defs()) {
    Register ResReg = ResMO.getReg();
    for (const MachineOperand &MO : MI.all_uses()) {
      if (MO.getReg() == ResReg)
        return false;
    }
    break; // Only check the first def.
  }
  return true;
}

static RegState getMopState(const MachineOperand &MO) {
  RegState S = {};
  if (MO.isImplicit())
    S |= RegState::Implicit;
  if (MO.isDead())
    S |= RegState::Dead;
  if (MO.isUndef())
    S |= RegState::Undef;
  if (MO.isKill())
    S |= RegState::Kill;
  if (MO.isEarlyClobber())
    S |= RegState::EarlyClobber;
  if (MO.getReg().isPhysical() && MO.isRenamable())
    S |= RegState::Renamable;
  return S;
}

// Returns false if there is a use of a def already in the map.
// In this case we must break the clause.
bool SSASIFormMemoryClausesImpl::canBundle(const MachineInstr &MI,
                                           const RegUse &Defs,
                                           const RegUse &Uses) const {
  // Check interference with defs.
  for (const MachineOperand &MO : MI.operands()) {
    // TODO: Prologue/Epilogue Insertion pass does not process bundled
    //       instructions.
    if (MO.isFI())
      return false;

    if (!MO.isReg())
      continue;

    Register Reg = MO.getReg();

    // If it is tied we will need to write same register as we read.
    if (MO.isTied())
      return false;

    const RegUse &Map = MO.isDef() ? Uses : Defs;
    auto Conflict = Map.find(Reg);
    if (Conflict == Map.end())
      continue;

    if (Reg.isPhysical())
      return false;

    LaneBitmask Mask = TRI->getSubRegIndexLaneMask(MO.getSubReg());
    if ((Conflict->second.second & Mask).any())
      return false;
  }

  return true;
}

// Since all defs in the clause are early clobber we can run out of registers.
// Function returns false if pressure would hit the limit if instruction is
// bundled into a memory clause.
//
// We accumulate pressure monotonically across the clause: because all defs are
// marked early-clobber they remain live until the clause end, so we never
// subtract pressure for uses that die mid-clause. This is conservative and
// avoids the need for LiveIntervals.
bool SSASIFormMemoryClausesImpl::checkPressure(const MachineInstr &MI,
                                               GCNRegPressure &CurPressure) {
  // Speculatively add this instruction's virtual defs to the running pressure.
  // Physical register defs are skipped: they are not allocatable slots and
  // GCNRegPressure::inc() requires a virtual register.
  GCNRegPressure NewPressure = CurPressure;
  for (const MachineOperand &MO : MI.defs()) {
    if (!MO.isReg() || !MO.getReg().isVirtual())
      continue;
    Register Reg = MO.getReg();
    LaneBitmask Mask = MO.getSubReg()
                           ? TRI->getSubRegIndexLaneMask(MO.getSubReg())
                           : MRI->getMaxLaneMaskForVReg(Reg);
    NewPressure.inc(Reg, LaneBitmask::getNone(), Mask, *MRI);
  }

  unsigned Occupancy = NewPressure.getOccupancy(
      *ST,
      MI.getMF()->getInfo<SIMachineFunctionInfo>()->getDynamicVGPRBlockSize());

  // Don't push over half the register budget. We don't want to introduce
  // spilling just to form a soft clause.
  //
  // FIXME: This pressure check is fundamentally broken. First, this is checking
  // the global pressure, not the pressure at this specific point in the
  // program. Second, it's not accounting for the increased liveness of the use
  // operands due to the early clobber we will introduce. Third, the pressure
  // tracking does not account for the alignment requirements for SGPRs, or the
  // fragmentation of registers the allocator will need to satisfy.
  if (Occupancy >= MFI->getMinAllowedOccupancy() &&
      NewPressure.getVGPRNum(ST->hasGFX90AInsts()) <= MaxVGPRs / 2 &&
      NewPressure.getSGPRNum() <= MaxSGPRs / 2) {
    LastRecordedOccupancy = Occupancy;
    CurPressure = NewPressure;
    return true;
  }
  return false;
}

// Collect register defs and uses along with their lane masks and states.
void SSASIFormMemoryClausesImpl::collectRegUses(const MachineInstr &MI,
                                                RegUse &Defs,
                                                RegUse &Uses) const {
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg())
      continue;
    Register Reg = MO.getReg();
    if (!Reg)
      continue;

    LaneBitmask Mask = Reg.isVirtual()
                           ? TRI->getSubRegIndexLaneMask(MO.getSubReg())
                           : LaneBitmask::getAll();
    RegUse &Map = MO.isDef() ? Defs : Uses;

    RegState State = getMopState(MO);
    auto [Loc, Inserted] = Map.try_emplace(Reg, State, Mask);
    if (!Inserted) {
      Loc->second.first |= State;
      Loc->second.second |= Mask;
    }
  }
}

// Check register def/use conflicts, occupancy limits and collect def/use maps.
// Return true if instruction can be bundled with previous. If it cannot
// def/use maps are not updated.
bool SSASIFormMemoryClausesImpl::processRegUses(const MachineInstr &MI,
                                                RegUse &Defs, RegUse &Uses,
                                                GCNRegPressure &CurPressure) {
  if (!canBundle(MI, Defs, Uses))
    return false;

  if (!checkPressure(MI, CurPressure))
    return false;

  collectRegUses(MI, Defs, Uses);
  return true;
}

bool SSASIFormMemoryClausesImpl::run(MachineFunction &MF, LiveVariables &LVIn) {
  ST = &MF.getSubtarget<GCNSubtarget>();
  if (!ST->isXNACKEnabled())
    return false;

  const SIInstrInfo *TII = ST->getInstrInfo();
  TRI = ST->getRegisterInfo();
  MRI = &MF.getRegInfo();
  MFI = MF.getInfo<SIMachineFunctionInfo>();
  LV = &LVIn;
  bool Changed = false;

  MaxVGPRs = TRI->getAllocatableSet(MF, &AMDGPU::VGPR_32RegClass).count();
  MaxSGPRs = TRI->getAllocatableSet(MF, &AMDGPU::SGPR_32RegClass).count();
  unsigned FuncMaxClause = MF.getFunction().getFnAttributeAsParsedInteger(
      "amdgpu-max-memory-clause", SSAMaxClause);

  for (MachineBasicBlock &MBB : MF) {
    // BlockPressure tracks the register pressure at the current scan position
    // within MBB. It is seeded with virtual registers live-in to this block
    // (as computed by LiveVariables), then updated instruction-by-instruction:
    // virtual register defs increase pressure; uses with kill flags decrease
    // it. In SSA form, kill flags are reliable (each vreg has exactly one
    // def), so this gives accurate intra-block liveness.
    GCNRegPressure BlockPressure;
    for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
      Register Reg = Register::index2VirtReg(I);
      if (LV->isLiveIn(Reg, MBB)) {
        LaneBitmask Mask = MRI->getMaxLaneMaskForVReg(Reg);
        BlockPressure.inc(Reg, LaneBitmask::getNone(), Mask, *MRI);
      }
    }

    // PressurePos is the next instruction to be consumed into BlockPressure.
    // It may lag behind the outer loop iterator when the inner clause-extension
    // loop advances Next past instructions not admitted to a clause.
    // advanceBlockPressure() catches it up before each clause attempt.
    auto PressurePos = MBB.instr_begin();

    auto advanceBlockPressure = [&](MachineBasicBlock::instr_iterator Target) {
      while (PressurePos != Target) {
        const MachineInstr &CurMI = *PressurePos++;
        if (CurMI.isMetaInstruction())
          continue;
        for (const MachineOperand &MO : CurMI.operands()) {
          if (!MO.isReg() || !MO.getReg().isVirtual())
            continue;
          Register Reg = MO.getReg();
          LaneBitmask Mask = TRI->getSubRegIndexLaneMask(MO.getSubReg());
          if (MO.isDef())
            BlockPressure.inc(Reg, LaneBitmask::getNone(), Mask, *MRI);
          else if (MO.isKill())
            BlockPressure.inc(Reg, Mask, LaneBitmask::getNone(), *MRI);
        }
      }
    };

    MachineBasicBlock::instr_iterator Next;
    for (auto I = MBB.instr_begin(), E = MBB.instr_end(); I != E; I = Next) {
      MachineInstr &MI = *I;
      Next = std::next(I);

      if (MI.isMetaInstruction())
        continue;

      bool IsVMEM = isVMEMClauseInst(MI);

      if (!isValidClauseInst(MI, IsVMEM)) {
        advanceBlockPressure(Next);
        continue;
      }

      // Bring BlockPressure up to (but not including) MI, then snapshot it as
      // the baseline pressure entering this potential clause.
      advanceBlockPressure(I);
      GCNRegPressure CurPressure = BlockPressure;

      RegUse Defs, Uses;
      // Kills: virtual registers with isKill() on any use inside the clause.
      // These registers die within the clause and need a whole-register KILL
      // pseudo after the last load to extend their live range past the
      // early-clobber defs. The specific subreg that LV flagged does not
      // matter; we always emit a whole-register KILL.
      DenseSet<Register> Kills;

      auto collectKills = [&](const MachineInstr &Instr) {
        for (const MachineOperand &MO : Instr.operands()) {
          if (!MO.isReg() || MO.isDef() || !MO.isKill() ||
              !MO.getReg().isVirtual())
            continue;
          Kills.insert(MO.getReg());
        }
      };

      if (!processRegUses(MI, Defs, Uses, CurPressure)) {
        advanceBlockPressure(Next);
        continue;
      }
      collectKills(MI);

      MachineBasicBlock::instr_iterator LastClauseInst = Next;
      unsigned Length = 1;
      for (; Next != E && Length < FuncMaxClause; ++Next) {
        // Debug instructions should not change the kill insertion.
        if (Next->isMetaInstruction())
          continue;

        if (!isValidClauseInst(*Next, IsVMEM))
          break;

        // A load from pointer which was loaded inside the same bundle is an
        // impossible clause because we will need to write and read the same
        // register inside. In this case processRegUses will return false.
        if (!processRegUses(*Next, Defs, Uses, CurPressure))
          break;

        collectKills(*Next);
        LastClauseInst = Next;
        ++Length;
      }
      if (Length < 2) {
        // Clause did not form; process MI normally. Instructions examined by
        // the inner loop but not admitted will be caught up by
        // advanceBlockPressure() at the start of the next outer iteration.
        advanceBlockPressure(std::next(I));
        continue;
      }

      Changed = true;
      MFI->limitOccupancy(LastRecordedOccupancy);

      assert(!LastClauseInst->isMetaInstruction());

      // For each register killed within the clause, insert a whole-register
      // KILL pseudo after the clause to extend its liveness through the
      // early-clobber defs. Registers not in Kills are live past the clause
      // and need nothing.
      for (Register Reg : Kills) {
        auto UseIt = Uses.find(Reg);
        assert(UseIt != Uses.end());
        RegState UseState = UseIt->second.first & ~RegState::Kill;

        MachineInstrBuilder Kill =
            BuildMI(*MI.getParent(), std::next(LastClauseInst), DebugLoc(),
                    TII->get(AMDGPU::KILL));
        Kill.addUse(Reg, UseState | RegState::Kill, AMDGPU::NoSubRegister);

        // Move the kill record from within the clause to the KILL instruction,
        // keeping LiveVariables consistent with the modified MIR.
        // findKill is guaranteed non-null: a kill flag within the clause
        // implies LV recorded a kill for this register in this block.
        MachineInstr *OldKill = LV->getVarInfo(Reg).findKill(&MBB);
        assert(OldKill &&
               "Kill flag in clause but no LV kill record in block?");
        // replaceKillInstruction only updates the VarInfo::Kills list; clear
        // the kill flag on the old instruction manually.
        OldKill->clearRegisterKills(Reg, TRI);
        LV->replaceKillInstruction(Reg, *OldKill, *Kill);
      }

      // Update BlockPressure: CurPressure already has all clause defs
      // accumulated; subtract the registers that died within the clause.
      BlockPressure = CurPressure;
      for (Register Reg : Kills)
        BlockPressure.inc(Reg, MRI->getMaxLaneMaskForVReg(Reg),
                          LaneBitmask::getNone(), *MRI);
      PressurePos = Next;
    }
  }

  return Changed;
}

bool SSASIFormMemoryClausesLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  LiveVariables &LV = getAnalysis<LiveVariablesWrapperPass>().getLV();
  return SSASIFormMemoryClausesImpl().run(MF, LV);
}

PreservedAnalyses
SSASIFormMemoryClausesPass::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  LiveVariables &LV = MFAM.getResult<LiveVariablesAnalysis>(MF);
  SSASIFormMemoryClausesImpl().run(MF, LV);
  return PreservedAnalyses::all();
}
