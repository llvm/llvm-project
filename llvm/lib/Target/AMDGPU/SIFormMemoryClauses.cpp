//===-- SIFormMemoryClauses.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass creates bundles of SMEM and VMEM instructions forming memory
/// clauses if XNACK is enabled. Def operands of clauses are marked as early
/// clobber to make sure we will not override any source within a clause.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNRegPressure.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-form-memory-clauses"

// Clauses longer then 15 instructions would overflow one of the counters
// and stall. They can stall even earlier if there are outstanding counters.
static cl::opt<unsigned>
MaxClause("amdgpu-max-memory-clause", cl::Hidden, cl::init(15),
          cl::desc("Maximum length of a memory clause, instructions"));

namespace {

class SIFormMemoryClauses : public MachineFunctionPass {
  typedef DenseMap<unsigned, std::pair<unsigned, LaneBitmask>> RegUse;

public:
  static char ID;

public:
  SIFormMemoryClauses() : MachineFunctionPass(ID) {
    initializeSIFormMemoryClausesPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI Form memory clauses";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervals>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }

private:
  template <typename Callable>
  void forAllLanes(Register Reg, LaneBitmask LaneMask, Callable Func) const;

  bool canBundle(const MachineInstr &MI, const RegUse &Defs,
                 const RegUse &Uses) const;
  bool checkPressure(const MachineInstr &MI, GCNDownwardRPTracker &RPT);
  void collectRegUses(const MachineInstr &MI, RegUse &Defs, RegUse &Uses) const;
  bool processRegUses(const MachineInstr &MI, RegUse &Defs, RegUse &Uses,
                      GCNDownwardRPTracker &RPT);

  const GCNSubtarget *ST;
  const SIRegisterInfo *TRI;
  const MachineRegisterInfo *MRI;
  SIMachineFunctionInfo *MFI;

  unsigned LastRecordedOccupancy;
  unsigned MaxVGPRs;
  unsigned MaxSGPRs;
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SIFormMemoryClauses, DEBUG_TYPE,
                      "SI Form memory clauses", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(SIFormMemoryClauses, DEBUG_TYPE,
                    "SI Form memory clauses", false, false)


char SIFormMemoryClauses::ID = 0;

char &llvm::SIFormMemoryClausesID = SIFormMemoryClauses::ID;

FunctionPass *llvm::createSIFormMemoryClausesPass() {
  return new SIFormMemoryClauses();
}

static bool isVMEMClauseInst(const MachineInstr &MI) {
  return SIInstrInfo::isFLAT(MI) || SIInstrInfo::isVMEM(MI);
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
  // If this is a load instruction where the result has been coalesced with an operand, then we cannot clause it.
  for (const MachineOperand &ResMO : MI.defs()) {
    Register ResReg = ResMO.getReg();
    for (const MachineOperand &MO : MI.uses()) {
      if (!MO.isReg() || MO.isDef())
        continue;
      if (MO.getReg() == ResReg)
        return false;
    }
    break; // Only check the first def.
  }
  return true;
}

static unsigned getMopState(const MachineOperand &MO) {
  unsigned S = 0;
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

template <typename Callable>
void SIFormMemoryClauses::forAllLanes(Register Reg, LaneBitmask LaneMask,
                                      Callable Func) const {
  if (LaneMask.all() || Reg.isPhysical() ||
      LaneMask == MRI->getMaxLaneMaskForVReg(Reg)) {
    Func(0);
    return;
  }

  const TargetRegisterClass *RC = MRI->getRegClass(Reg);
  unsigned E = TRI->getNumSubRegIndices();
  SmallVector<unsigned, AMDGPU::NUM_TARGET_SUBREGS> CoveringSubregs;
  for (unsigned Idx = 1; Idx < E; ++Idx) {
    // Is this index even compatible with the given class?
    if (TRI->getSubClassWithSubReg(RC, Idx) != RC)
      continue;
    LaneBitmask SubRegMask = TRI->getSubRegIndexLaneMask(Idx);
    // Early exit if we found a perfect match.
    if (SubRegMask == LaneMask) {
      Func(Idx);
      return;
    }

    if ((SubRegMask & ~LaneMask).any() || (SubRegMask & LaneMask).none())
      continue;

    CoveringSubregs.push_back(Idx);
  }

  llvm::sort(CoveringSubregs, [this](unsigned A, unsigned B) {
    LaneBitmask MaskA = TRI->getSubRegIndexLaneMask(A);
    LaneBitmask MaskB = TRI->getSubRegIndexLaneMask(B);
    unsigned NA = MaskA.getNumLanes();
    unsigned NB = MaskB.getNumLanes();
    if (NA != NB)
      return NA > NB;
    return MaskA.getHighestLane() > MaskB.getHighestLane();
  });

  for (unsigned Idx : CoveringSubregs) {
    LaneBitmask SubRegMask = TRI->getSubRegIndexLaneMask(Idx);
    if ((SubRegMask & ~LaneMask).any() || (SubRegMask & LaneMask).none())
      continue;

    Func(Idx);
    LaneMask &= ~SubRegMask;
    if (LaneMask.none())
      return;
  }

  llvm_unreachable("Failed to find all subregs to cover lane mask");
}

// Returns false if there is a use of a def already in the map.
// In this case we must break the clause.
bool SIFormMemoryClauses::canBundle(const MachineInstr &MI, const RegUse &Defs,
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
bool SIFormMemoryClauses::checkPressure(const MachineInstr &MI,
                                        GCNDownwardRPTracker &RPT) {
  // NB: skip advanceBeforeNext() call. Since all defs will be marked
  // early-clobber they will all stay alive at least to the end of the
  // clause. Therefor we should not decrease pressure even if load
  // pointer becomes dead and could otherwise be reused for destination.
  RPT.advanceToNext();
  GCNRegPressure MaxPressure = RPT.moveMaxPressure();
  unsigned Occupancy = MaxPressure.getOccupancy(*ST);

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
      MaxPressure.getVGPRNum() <= MaxVGPRs / 2 &&
      MaxPressure.getSGPRNum() <= MaxSGPRs / 2) {
    LastRecordedOccupancy = Occupancy;
    return true;
  }
  return false;
}

// Collect register defs and uses along with their lane masks and states.
void SIFormMemoryClauses::collectRegUses(const MachineInstr &MI,
                                         RegUse &Defs, RegUse &Uses) const {
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

    auto Loc = Map.find(Reg);
    unsigned State = getMopState(MO);
    if (Loc == Map.end()) {
      Map[Reg] = std::make_pair(State, Mask);
    } else {
      Loc->second.first |= State;
      Loc->second.second |= Mask;
    }
  }
}

// Check register def/use conflicts, occupancy limits and collect def/use maps.
// Return true if instruction can be bundled with previous. It it cannot
// def/use maps are not updated.
bool SIFormMemoryClauses::processRegUses(const MachineInstr &MI,
                                         RegUse &Defs, RegUse &Uses,
                                         GCNDownwardRPTracker &RPT) {
  if (!canBundle(MI, Defs, Uses))
    return false;

  if (!checkPressure(MI, RPT))
    return false;

  collectRegUses(MI, Defs, Uses);
  return true;
}

bool SIFormMemoryClauses::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  ST = &MF.getSubtarget<GCNSubtarget>();
  if (!ST->isXNACKEnabled())
    return false;

  const SIInstrInfo *TII = ST->getInstrInfo();
  TRI = ST->getRegisterInfo();
  MRI = &MF.getRegInfo();
  MFI = MF.getInfo<SIMachineFunctionInfo>();
  LiveIntervals *LIS = &getAnalysis<LiveIntervals>();
  SlotIndexes *Ind = LIS->getSlotIndexes();
  bool Changed = false;

  MaxVGPRs = TRI->getAllocatableSet(MF, &AMDGPU::VGPR_32RegClass).count();
  MaxSGPRs = TRI->getAllocatableSet(MF, &AMDGPU::SGPR_32RegClass).count();
  unsigned FuncMaxClause = AMDGPU::getIntegerAttribute(
      MF.getFunction(), "amdgpu-max-memory-clause", MaxClause);

  SmallVector<MachineInstr *> DbgInstrs;

  for (MachineBasicBlock &MBB : MF) {
    GCNDownwardRPTracker RPT(*LIS);
    MachineBasicBlock::instr_iterator Next;
    for (auto I = MBB.instr_begin(), E = MBB.instr_end(); I != E; I = Next) {
      MachineInstr &MI = *I;
      Next = std::next(I);

      if (MI.isDebugInstr())
        continue;

      bool IsVMEM = isVMEMClauseInst(MI);

      if (!isValidClauseInst(MI, IsVMEM))
        continue;

      if (!RPT.getNext().isValid())
        RPT.reset(MI);
      else { // Advance the state to the current MI.
        RPT.advance(MachineBasicBlock::const_iterator(MI));
        RPT.advanceBeforeNext();
      }

      const GCNRPTracker::LiveRegSet LiveRegsCopy(RPT.getLiveRegs());
      RegUse Defs, Uses;
      if (!processRegUses(MI, Defs, Uses, RPT)) {
        RPT.reset(MI, &LiveRegsCopy);
        continue;
      }

      unsigned Length = 1;
      for ( ; Next != E && Length < FuncMaxClause; ++Next) {
        // Debug instructions should not change the bundling. We need to move
        // these after the bundle
        if (Next->isDebugInstr())
          continue;

        if (!isValidClauseInst(*Next, IsVMEM))
          break;

        // A load from pointer which was loaded inside the same bundle is an
        // impossible clause because we will need to write and read the same
        // register inside. In this case processRegUses will return false.
        if (!processRegUses(*Next, Defs, Uses, RPT))
          break;

        ++Length;
      }
      if (Length < 2) {
        RPT.reset(MI, &LiveRegsCopy);
        continue;
      }

      Changed = true;
      MFI->limitOccupancy(LastRecordedOccupancy);

      auto B = BuildMI(MBB, I, DebugLoc(), TII->get(TargetOpcode::BUNDLE));
      Ind->insertMachineInstrInMaps(*B);

      // Restore the state after processing the bundle.
      RPT.reset(*B, &LiveRegsCopy);
      DbgInstrs.clear();

      auto BundleNext = I;
      for (auto BI = I; BI != Next; BI = BundleNext) {
        BundleNext = std::next(BI);

        if (BI->isDebugValue()) {
          DbgInstrs.push_back(BI->removeFromParent());
          continue;
        }

        BI->bundleWithPred();
        Ind->removeSingleMachineInstrFromMaps(*BI);

        for (MachineOperand &MO : BI->defs())
          if (MO.readsReg())
            MO.setIsInternalRead(true);
      }

      // Replace any debug instructions after the new bundle.
      for (MachineInstr *DbgInst : DbgInstrs)
        MBB.insert(Next, DbgInst);

      for (auto &&R : Defs) {
        forAllLanes(R.first, R.second.second, [&R, &B](unsigned SubReg) {
          unsigned S = R.second.first | RegState::EarlyClobber;
          if (!SubReg)
            S &= ~(RegState::Undef | RegState::Dead);
          B.addDef(R.first, S, SubReg);
        });
      }

      for (auto &&R : Uses) {
        forAllLanes(R.first, R.second.second, [&R, &B](unsigned SubReg) {
          B.addUse(R.first, R.second.first & ~RegState::Kill, SubReg);
        });
      }

      for (auto &&R : Defs) {
        Register Reg = R.first;
        Uses.erase(Reg);
        if (Reg.isPhysical())
          continue;
        LIS->removeInterval(Reg);
        LIS->createAndComputeVirtRegInterval(Reg);
      }

      for (auto &&R : Uses) {
        Register Reg = R.first;
        if (Reg.isPhysical())
          continue;
        LIS->removeInterval(Reg);
        LIS->createAndComputeVirtRegInterval(Reg);
      }
    }
  }

  return Changed;
}
