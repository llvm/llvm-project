//=====-- Rematerializer.cpp - MIR rematerialization support ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==-----------------------------------------------------------------------===//
//
/// \file
/// Implements helpers for target-independent rematerialization at the MIR
/// level.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Rematerializer.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "rematerializer"

using namespace llvm;
using RegisterIdx = Rematerializer::RegisterIdx;

// Pin the vtable to this file.
void Rematerializer::Listener::anchor() {}

/// Checks whether the value in \p LI at \p UseIdx is identical to \p OVNI (this
/// implies it is also live there). When \p LI has sub-ranges, checks that
/// all sub-ranges intersecting with \p Mask are also live at \p UseIdx.
static bool isIdenticalAtUse(const VNInfo &OVNI, LaneBitmask Mask,
                             SlotIndex UseIdx, const LiveInterval &LI) {
  if (&OVNI != LI.getVNInfoAt(UseIdx))
    return false;

  if (LI.hasSubRanges()) {
    // Check that intersecting subranges are live at user.
    for (const LiveInterval::SubRange &SR : LI.subranges()) {
      if ((SR.LaneMask & Mask).none())
        continue;
      if (!SR.liveAt(UseIdx))
        return false;

      // Early exit if all used lanes are checked. No need to continue.
      Mask &= ~SR.LaneMask;
      if (Mask.none())
        break;
    }
  }
  return true;
}

/// If \p MO is a virtual read register, returns it. Otherwise returns the
/// sentinel register.
static Register getRegDependency(const MachineOperand &MO) {
  if (!MO.isReg() || !MO.readsReg())
    return Register();
  Register Reg = MO.getReg();
  if (Reg.isPhysical()) {
    // By the requirements on trivially rematerializable instructions, a
    // physical register use is either constant or ignorable.
    return Register();
  }
  return Reg;
}

RegisterIdx Rematerializer::rematerializeToRegion(RegisterIdx RootIdx,
                                                  unsigned UseRegion,
                                                  DependencyReuseInfo &DRI) {
  MachineInstr *FirstMI =
      getReg(RootIdx).getRegionUseBounds(UseRegion, LIS).first;
  // If there are no users in the region, rematerialize the register at the very
  // end of the region.
  MachineBasicBlock::iterator InsertPos =
      FirstMI ? FirstMI : Regions[UseRegion].second;
  RegisterIdx NewRegIdx =
      rematerializeToPos(RootIdx, UseRegion, InsertPos, DRI);
  transferRegionUsers(RootIdx, NewRegIdx, UseRegion);
  return NewRegIdx;
}

RegisterIdx
Rematerializer::rematerializeToPos(RegisterIdx RootIdx, unsigned UseRegion,
                                   MachineBasicBlock::iterator InsertPos,
                                   DependencyReuseInfo &DRI) {
  assert(!DRI.DependencyMap.contains(RootIdx));
  LLVM_DEBUG(dbgs() << "Rematerializing " << printID(RootIdx) << '\n');

  // Traverse the root's dependency DAG depth-first to find the set of
  // registers we must rematerialize along with it and a legal order to
  // rematerialize them in.
  SmallVector<RegisterIdx, 4> DepDAG{RootIdx};
  SmallSetVector<RegisterIdx, 8> RematOrder;
  RematOrder.insert(RootIdx);
  do {
    RegisterIdx RegIdx = DepDAG.pop_back_val();
    for (const Reg::Dependency &Dep : getReg(RegIdx).Dependencies) {
      // The dependency may already have a rematerialization ready to use.
      if (DRI.DependencyMap.contains(Dep.RegIdx))
        continue;
      // We may have already seen the dependency in the dependency DAG.
      if (RematOrder.contains(Dep.RegIdx))
        continue;
      DepDAG.push_back(Dep.RegIdx);
      RematOrder.insert(Dep.RegIdx);
    }
  } while (!DepDAG.empty());

  // Rematerialize all necessary registers in the root's dependency DAG. At each
  // rematerialization, dependencies should already be available.
  RegisterIdx LastNewIdx;
  for (RegisterIdx RegIdx : reverse(RematOrder)) {
    assert(!DRI.DependencyMap.contains(RegIdx) && "useless remat");
    SmallVector<Reg::Dependency, 2> Dependencies;
    for (const Reg::Dependency &Dep : getReg(RegIdx).Dependencies)
      Dependencies.emplace_back(Dep.MOIdx, DRI.DependencyMap.at(Dep.RegIdx));
    LastNewIdx =
        rematerializeReg(RegIdx, UseRegion, InsertPos, std::move(Dependencies));
    DRI.DependencyMap.insert({RegIdx, LastNewIdx});
  }

  return LastNewIdx;
}

void Rematerializer::transferUser(RegisterIdx FromRegIdx, RegisterIdx ToRegIdx,
                                  unsigned UserRegion, MachineInstr &UserMI) {
  transferUserImpl(FromRegIdx, ToRegIdx, UserMI);
  Regs[FromRegIdx].eraseUser(&UserMI, UserRegion);
  Regs[ToRegIdx].addUser(&UserMI, UserRegion);
  deleteRegIfUnused(FromRegIdx);
}

void Rematerializer::transferRegionUsers(RegisterIdx FromRegIdx,
                                         RegisterIdx ToRegIdx,
                                         unsigned UseRegion) {
  auto &FromRegUsers = Regs[FromRegIdx].Uses;
  auto UsesIt = FromRegUsers.find(UseRegion);
  if (UsesIt == FromRegUsers.end())
    return;

  const SmallDenseSet<MachineInstr *, 4> &RegionUsers = UsesIt->getSecond();
  for (MachineInstr *UserMI : RegionUsers)
    transferUserImpl(FromRegIdx, ToRegIdx, *UserMI);
  Regs[ToRegIdx].addUsers(RegionUsers, UseRegion);
  FromRegUsers.erase(UseRegion);
  deleteRegIfUnused(FromRegIdx);
}

void Rematerializer::transferAllUsers(RegisterIdx FromRegIdx,
                                      RegisterIdx ToRegIdx) {
  Reg &FromReg = Regs[FromRegIdx], &ToReg = Regs[ToRegIdx];
  for (const auto &[UseRegion, RegionUsers] : FromReg.Uses) {
    for (MachineInstr *UserMI : RegionUsers)
      transferUserImpl(FromRegIdx, ToRegIdx, *UserMI);
    ToReg.addUsers(RegionUsers, UseRegion);
  }
  FromReg.Uses.clear();
  deleteRegIfUnused(FromRegIdx);
}

void Rematerializer::transferUserImpl(RegisterIdx FromRegIdx,
                                      RegisterIdx ToRegIdx,
                                      MachineInstr &UserMI) {
  assert(FromRegIdx != ToRegIdx && "identical registers");
  assert(getOriginOrSelf(FromRegIdx) == getOriginOrSelf(ToRegIdx) &&
         "unrelated registers");

  LLVM_DEBUG(dbgs() << "User transfer from " << printID(FromRegIdx) << " to "
                    << printID(ToRegIdx) << ": " << printUser(&UserMI) << '\n');

  UserMI.substituteRegister(getReg(FromRegIdx).getDefReg(),
                            getReg(ToRegIdx).getDefReg(), 0, TRI);
  LISUpdates.insert(FromRegIdx);
  LISUpdates.insert(ToRegIdx);

  // If the user is rematerializable, we must change its dependency to the
  // new register.
  if (RegisterIdx UserRegIdx = getDefRegIdx(UserMI); UserRegIdx != NoReg) {
    // Look for the user's dependency that matches the register.
    for (Reg::Dependency &Dep : Regs[UserRegIdx].Dependencies) {
      if (Dep.RegIdx == FromRegIdx) {
        Dep.RegIdx = ToRegIdx;
        return;
      }
    }
    llvm_unreachable("broken dependency");
  }
}

void Rematerializer::updateLiveIntervals() {
  DenseSet<Register> SeenUnrematRegs;
  for (RegisterIdx RegIdx : LISUpdates) {
    const Reg &UpdateReg = getReg(RegIdx);
    assert(UpdateReg.isAlive() && "dead register");

    Register DefReg = UpdateReg.getDefReg();
    if (LIS.hasInterval(DefReg))
      LIS.removeInterval(DefReg);
    // Rematerializable registers have a single definition by construction so
    // re-creating their interval cannot yield a live interval with multiple
    // connected components.
    LIS.createAndComputeVirtRegInterval(DefReg);

    LLVM_DEBUG({
      dbgs() << "Re-computed interval for " << printID(RegIdx) << ": ";
      LIS.getInterval(DefReg).print(dbgs());
      dbgs() << '\n' << printRegUsers(RegIdx);
    });

    // Update intervals for unrematerializable operands.
    for (unsigned MOIdx : getUnrematableOprds(RegIdx)) {
      Register UnrematReg = UpdateReg.DefMI->getOperand(MOIdx).getReg();
      if (!SeenUnrematRegs.insert(UnrematReg).second)
        continue;
      LIS.removeInterval(UnrematReg);
      bool NeedSplit = false;

      // Unrematerializable registers may end up with multiple connected
      // components in their live interval after it is re-created. It needs to
      // be split in such cases. We don't track unrematerializable registers by
      // their actual register index (just by operand index) so we do not need
      // to update any state in the rematerializer.
      LiveInterval &LI =
          LIS.createAndComputeVirtRegInterval(UnrematReg, NeedSplit);
      if (NeedSplit) {
        SmallVector<LiveInterval *> SplitLIs;
        LIS.splitSeparateComponents(LI, SplitLIs);
      }
      LLVM_DEBUG(
          dbgs() << "  Re-computed interval for register "
                 << printReg(UnrematReg, &TRI,
                             UpdateReg.DefMI->getOperand(MOIdx).getSubReg(),
                             &MRI)
                 << '\n');
    }
  }
  LISUpdates.clear();
}

bool Rematerializer::isMOIdenticalAtUses(MachineOperand &MO,
                                         ArrayRef<SlotIndex> Uses) const {
  if (Uses.empty())
    return true;
  Register Reg = MO.getReg();
  unsigned SubIdx = MO.getSubReg();
  LaneBitmask Mask = SubIdx ? TRI.getSubRegIndexLaneMask(SubIdx)
                            : MRI.getMaxLaneMaskForVReg(Reg);
  const LiveInterval &LI = LIS.getInterval(Reg);
  const VNInfo *DefVN =
      LI.getVNInfoAt(LIS.getInstructionIndex(*MO.getParent()).getRegSlot(true));
  for (SlotIndex Use : Uses) {
    if (!isIdenticalAtUse(*DefVN, Mask, Use, LI))
      return false;
  }
  return true;
}

RegisterIdx Rematerializer::findRematInRegion(RegisterIdx RegIdx,
                                              unsigned Region,
                                              SlotIndex Before) const {
  auto It = Rematerializations.find(getOriginOrSelf(RegIdx));
  if (It == Rematerializations.end())
    return NoReg;
  const RematsOf &Remats = It->getSecond();

  SlotIndex BestSlot;
  RegisterIdx BestRegIdx = NoReg;
  for (RegisterIdx RematRegIdx : Remats) {
    const Reg &RematReg = getReg(RematRegIdx);
    if (RematReg.DefRegion != Region || RematReg.Uses.empty())
      continue;
    SlotIndex RematRegSlot =
        LIS.getInstructionIndex(*RematReg.DefMI).getRegSlot();
    if (RematRegSlot < Before &&
        (BestRegIdx == NoReg || RematRegSlot > BestSlot)) {
      BestSlot = RematRegSlot;
      BestRegIdx = RematRegIdx;
    }
  }
  return BestRegIdx;
}

void Rematerializer::deleteRegIfUnused(RegisterIdx RootIdx) {
  if (!getReg(RootIdx).Uses.empty())
    return;

  // Traverse the root's dependency DAG depth-first to find the set of registers
  // we can delete and a legal order to delete them in.
  SmallVector<RegisterIdx, 4> DepDAG{RootIdx};
  SmallSetVector<RegisterIdx, 8> DeleteOrder;
  DeleteOrder.insert(RootIdx);
  do {
    // A deleted register's dependencies may be deletable too.
    const Reg &DeleteReg = getReg(DepDAG.pop_back_val());
    for (const Reg::Dependency &Dep : DeleteReg.Dependencies) {
      // All dependencies loose a user (the deleted register).
      Reg &DepReg = Regs[Dep.RegIdx];
      DepReg.eraseUser(DeleteReg.DefMI, DeleteReg.DefRegion);
      if (DepReg.Uses.empty()) {
        DeleteOrder.insert(Dep.RegIdx);
        DepDAG.push_back(Dep.RegIdx);
      }
    }
  } while (!DepDAG.empty());

  for (RegisterIdx RegIdx : reverse(DeleteOrder)) {
    Reg &DeleteReg = Regs[RegIdx];

    // It is possible that the defined register we are deleting doesn't have an
    // interval yet if the LIS hasn't been updated since it was created.
    Register DefReg = DeleteReg.getDefReg();
    if (LIS.hasInterval(DefReg))
      LIS.removeInterval(DefReg);
    LISUpdates.erase(RegIdx);

    deleteReg(RegIdx);
    if (isRematerializedRegister(RegIdx)) {
      // Delete rematerialized register from its origin's rematerializations.
      RematsOf &OriginRemats = Rematerializations.at(getOriginOf(RegIdx));
      assert(OriginRemats.contains(RegIdx) && "broken remat<->origin link");
      OriginRemats.erase(RegIdx);
      if (OriginRemats.empty())
        Rematerializations.erase(RegIdx);
    }
    LLVM_DEBUG(dbgs() << "** Deleted " << printID(RegIdx) << "\n");
  }
}

void Rematerializer::deleteReg(RegisterIdx RegIdx) {
  noteRegDeleted(RegIdx);

  Reg &DeleteReg = Regs[RegIdx];
  assert(DeleteReg.DefMI && "register was already deleted");
  // It is not possible for the deleted instruction to be the upper region
  // boundary since we don't ever consider them rematerializable.
  MachineBasicBlock::iterator &RegionBegin = Regions[DeleteReg.DefRegion].first;
  if (RegionBegin == DeleteReg.DefMI)
    RegionBegin = std::next(MachineBasicBlock::iterator(DeleteReg.DefMI));
  LIS.RemoveMachineInstrFromMaps(*DeleteReg.DefMI);
  DeleteReg.DefMI->eraseFromParent();
  DeleteReg.DefMI = nullptr;
}

Rematerializer::Rematerializer(MachineFunction &MF,
                               SmallVectorImpl<RegionBoundaries> &Regions,
                               LiveIntervals &LIS)
    : Regions(Regions), MRI(MF.getRegInfo()), LIS(LIS),
      TII(*MF.getSubtarget().getInstrInfo()), TRI(TII.getRegisterInfo()) {
#ifdef EXPENSIVE_CHECKS
  // Check that regions are valid.
  DenseSet<MachineInstr *> SeenMIs;
  for (const auto &[RegionBegin, RegionEnd] : Regions) {
    assert(RegionBegin != RegionEnd && "empty region");
    for (auto MI = RegionBegin; MI != RegionEnd; ++MI) {
      bool IsNewMI = SeenMIs.insert(&*MI).second;
      assert(IsNewMI && "overlapping regions");
      assert(!MI->isTerminator() && "terminator in region");
    }
    if (RegionEnd != RegionBegin->getParent()->end()) {
      bool IsNewMI = SeenMIs.insert(&*RegionEnd).second;
      assert(IsNewMI && "overlapping regions (upper bound)");
    }
  }
#endif
}

bool Rematerializer::analyze() {
  Regs.clear();
  UnrematableOprds.clear();
  Origins.clear();
  Rematerializations.clear();
  RegionMBB.clear();
  RegToIdx.clear();
  LISUpdates.clear();
  if (Regions.empty())
    return false;

  /// Maps all MIs to their parent region. Region terminators are considered
  /// part of the region they terminate.
  DenseMap<MachineInstr *, unsigned> MIRegion;

  // Initialize MI to containing region mapping.
  RegionMBB.reserve(Regions.size());
  for (unsigned I = 0, E = Regions.size(); I < E; ++I) {
    RegionBoundaries Region = Regions[I];
    assert(Region.first != Region.second && "empty cannot be region");
    for (auto MI = Region.first; MI != Region.second; ++MI) {
      assert(!MIRegion.contains(&*MI) && "regions should not intersect");
      MIRegion.insert({&*MI, I});
    }
    MachineBasicBlock &MBB = *Region.first->getParent();
    RegionMBB.push_back(&MBB);

    // A terminator instruction is considered part of the region it terminates.
    if (Region.second != MBB.end()) {
      MachineInstr *RegionTerm = &*Region.second;
      assert(!MIRegion.contains(RegionTerm) && "regions should not intersect");
      MIRegion.insert({RegionTerm, I});
    }
  }

  const unsigned NumVirtRegs = MRI.getNumVirtRegs();
  BitVector SeenRegs(NumVirtRegs);
  for (unsigned I = 0, E = NumVirtRegs; I != E; ++I) {
    if (!SeenRegs[I])
      addRegIfRematerializable(I, MIRegion, SeenRegs);
  }
  assert(Regs.size() == UnrematableOprds.size());

  LLVM_DEBUG({
    for (RegisterIdx I = 0, E = getNumRegs(); I < E; ++I)
      dbgs() << printDependencyDAG(I) << '\n';
  });
  return !Regs.empty();
}

void Rematerializer::addRegIfRematerializable(
    unsigned VirtRegIdx, const DenseMap<MachineInstr *, unsigned> &MIRegion,
    BitVector &SeenRegs) {
  assert(!SeenRegs[VirtRegIdx] && "register already seen");
  Register DefReg = Register::index2VirtReg(VirtRegIdx);
  SeenRegs.set(VirtRegIdx);

  MachineOperand *MO = MRI.getOneDef(DefReg);
  if (!MO)
    return;
  MachineInstr &DefMI = *MO->getParent();
  if (!isMIRematerializable(DefMI))
    return;
  auto DefRegion = MIRegion.find(&DefMI);
  if (DefRegion == MIRegion.end())
    return;

  Reg RematReg;
  RematReg.DefMI = &DefMI;
  RematReg.DefRegion = DefRegion->second;
  unsigned SubIdx = DefMI.getOperand(0).getSubReg();
  RematReg.Mask = SubIdx ? TRI.getSubRegIndexLaneMask(SubIdx)
                         : MRI.getMaxLaneMaskForVReg(DefReg);

  // Collect the candidate's direct users, both rematerializable and
  // unrematerializable. MIs outside provided regions cannot be tracked so the
  // registers they use are not safely rematerializable.
  for (MachineInstr &UseMI : MRI.use_nodbg_instructions(DefReg)) {
    if (auto UseRegion = MIRegion.find(&UseMI); UseRegion != MIRegion.end())
      RematReg.addUser(&UseMI, UseRegion->second);
    else
      return;
  }
  if (RematReg.Uses.empty())
    return;

  // Collect the candidate's dependencies. If the same register is used
  // multiple times we just need to consider it once.
  SmallDenseSet<Register, 4> AllDepRegs;
  SmallVector<unsigned, 2> UnrematDeps;
  for (const auto &[MOIdx, MO] : enumerate(RematReg.DefMI->operands())) {
    Register DepReg = getRegDependency(MO);
    if (!DepReg || !AllDepRegs.insert(DepReg).second)
      continue;
    unsigned DepRegIdx = DepReg.virtRegIndex();
    if (!SeenRegs[DepRegIdx])
      addRegIfRematerializable(DepRegIdx, MIRegion, SeenRegs);
    if (auto DepIt = RegToIdx.find(DepReg); DepIt != RegToIdx.end())
      RematReg.Dependencies.push_back(Reg::Dependency(MOIdx, DepIt->second));
    else
      UnrematDeps.push_back(MOIdx);
  }

  // The register is rematerializable.
  RegToIdx.insert({DefReg, Regs.size()});
  Regs.push_back(RematReg);
  UnrematableOprds.push_back(UnrematDeps);
}

bool Rematerializer::isMIRematerializable(const MachineInstr &MI) const {
  if (!TII.isReMaterializable(MI))
    return false;

  assert(MI.getOperand(0).getReg().isVirtual() && "should be virtual");
  assert(MRI.hasOneDef(MI.getOperand(0).getReg()) && "should have single def");

  for (const MachineOperand &MO : MI.all_uses()) {
    // We can't remat physreg uses, unless it is a constant or an ignorable
    // use (e.g. implicit exec use on VALU instructions)
    if (MO.getReg().isPhysical()) {
      if (MRI.isConstantPhysReg(MO.getReg()) || TII.isIgnorableUse(MO))
        continue;
      return false;
    }
  }

  return true;
}

RegisterIdx Rematerializer::getDefRegIdx(const MachineInstr &MI) const {
  if (!MI.getNumOperands() || !MI.getOperand(0).isReg() ||
      MI.getOperand(0).readsReg())
    return NoReg;
  Register Reg = MI.getOperand(0).getReg();
  auto UserRegIt = RegToIdx.find(Reg);
  if (UserRegIt == RegToIdx.end())
    return NoReg;
  return UserRegIt->second;
}

RegisterIdx Rematerializer::rematerializeReg(
    RegisterIdx RegIdx, unsigned UseRegion,
    MachineBasicBlock::iterator InsertPos,
    SmallVectorImpl<Reg::Dependency> &&Dependencies) {
  RegisterIdx NewRegIdx = Regs.size();

  Reg &NewReg = Regs.emplace_back();
  Reg &FromReg = Regs[RegIdx];
  NewReg.Mask = FromReg.Mask;
  NewReg.DefRegion = UseRegion;
  NewReg.Dependencies = std::move(Dependencies);

  // Track rematerialization link between registers. Origins are always
  // registers that existed originally, and rematerializations are always
  // attached to them.
  const RegisterIdx OriginIdx = getOriginOrSelf(RegIdx);
  Origins.push_back(OriginIdx);
  Rematerializations[OriginIdx].insert(NewRegIdx);

  // Use the TII to rematerialize the defining instruction with a new defined
  // register.
  Register NewDefReg = MRI.cloneVirtualRegister(FromReg.getDefReg());
  TII.reMaterialize(*RegionMBB[UseRegion], InsertPos, NewDefReg, 0,
                    *FromReg.DefMI);
  NewReg.DefMI = &*std::prev(InsertPos);
  RegToIdx.insert({NewDefReg, NewRegIdx});
  postRematerialization(RegIdx, NewRegIdx, InsertPos);

  noteRegCreated(NewRegIdx);
  LLVM_DEBUG(dbgs() << "** Rematerialized " << printID(RegIdx) << " as "
                    << printRematReg(NewRegIdx) << '\n');
  return NewRegIdx;
}

void Rematerializer::recreateReg(
    RegisterIdx RegIdx, unsigned DefRegion,
    MachineBasicBlock::iterator InsertPos, Register DefReg,
    SmallVectorImpl<Reg::Dependency> &&Dependencies) {
  assert(RegToIdx.contains(DefReg) && "unknown defined register");
  assert(RegToIdx.at(DefReg) == RegIdx && "incorrect defined register");
  assert(!getReg(RegIdx).DefMI && "register is still alive");

  Reg &OriginReg = Regs[RegIdx];
  OriginReg.DefRegion = DefRegion;
  OriginReg.Dependencies = std::move(Dependencies);

  // Re-establish the link between origin and rematerialization if necessary.
  const bool RecreateOriginalReg = isOriginalRegister(RegIdx);
  if (!RecreateOriginalReg)
    Rematerializations[getOriginOf(RegIdx)].insert(RegIdx);

  // Rematerialize from one of the existing rematerializations or from the
  // origin. We expect at least one to exist, otherwise it would mean the value
  // held by the original register is no longer available anywhere in the MF.
  RegisterIdx ModelRegIdx;
  if (RecreateOriginalReg) {
    assert(Rematerializations.contains(RegIdx) && "expected remats");
    ModelRegIdx = *Rematerializations.at(RegIdx).begin();
  } else {
    assert(getReg(getOriginOf(RegIdx)).DefMI && "expected alive origin");
    ModelRegIdx = getOriginOf(RegIdx);
  }
  const MachineInstr &ModelDefMI = *getReg(ModelRegIdx).DefMI;

  TII.reMaterialize(*RegionMBB[DefRegion], InsertPos, DefReg, 0, ModelDefMI);
  OriginReg.DefMI = &*std::prev(InsertPos);
  postRematerialization(ModelRegIdx, RegIdx, InsertPos);
  LLVM_DEBUG(dbgs() << "** Recreated " << printID(RegIdx) << " as "
                    << printRematReg(RegIdx) << '\n');
}

void Rematerializer::postRematerialization(
    RegisterIdx ModelRegIdx, RegisterIdx RematRegIdx,
    MachineBasicBlock::iterator InsertPos) {

  // The start of the new register's region may have changed.
  Reg &ModelReg = Regs[ModelRegIdx], &RematReg = Regs[RematRegIdx];
  LIS.InsertMachineInstrInMaps(*RematReg.DefMI);
  MachineBasicBlock::iterator &RegionBegin = Regions[RematReg.DefRegion].first;
  if (RegionBegin == std::next(MachineBasicBlock::iterator(RematReg.DefMI)))
    RegionBegin = RematReg.DefMI;

  // Replace dependencies as needed in the rematerialized MI. All dependencies
  // of the latter gain a new user.
  auto ZipedDeps = zip_equal(ModelReg.Dependencies, RematReg.Dependencies);
  for (const auto &[OldDep, NewDep] : ZipedDeps) {
    assert(OldDep.MOIdx == NewDep.MOIdx && "operand mismatch");
    LLVM_DEBUG(dbgs() << "  Operand #" << OldDep.MOIdx << ": "
                      << printID(OldDep.RegIdx) << " -> "
                      << printID(NewDep.RegIdx) << '\n');

    Reg &NewDepReg = Regs[NewDep.RegIdx];
    if (OldDep.RegIdx != NewDep.RegIdx) {
      Register OldDefReg = ModelReg.DefMI->getOperand(OldDep.MOIdx).getReg();
      RematReg.DefMI->substituteRegister(OldDefReg, NewDepReg.getDefReg(), 0,
                                         TRI);
      LISUpdates.insert(OldDep.RegIdx);
    }
    NewDepReg.addUser(RematReg.DefMI, RematReg.DefRegion);
    LISUpdates.insert(NewDep.RegIdx);
  }
}

std::pair<MachineInstr *, MachineInstr *>
Rematerializer::Reg::getRegionUseBounds(unsigned UseRegion,
                                        const LiveIntervals &LIS) const {
  auto It = Uses.find(UseRegion);
  if (It == Uses.end())
    return {nullptr, nullptr};
  const RegionUsers &RegionUsers = It->getSecond();
  assert(!RegionUsers.empty() && "empty userset in region");

  auto User = RegionUsers.begin(), UserEnd = RegionUsers.end();
  MachineInstr *FirstMI = *User, *LastMI = FirstMI;
  SlotIndex FirstIndex = LIS.getInstructionIndex(*FirstMI),
            LastIndex = FirstIndex;

  while (++User != UserEnd) {
    SlotIndex UserIndex = LIS.getInstructionIndex(**User);
    if (UserIndex < FirstIndex) {
      FirstIndex = UserIndex;
      FirstMI = *User;
    } else if (UserIndex > LastIndex) {
      LastIndex = UserIndex;
      LastMI = *User;
    }
  }

  return {FirstMI, LastMI};
}

void Rematerializer::Reg::addUser(MachineInstr *MI, unsigned Region) {
  Uses[Region].insert(MI);
}

void Rematerializer::Reg::addUsers(const RegionUsers &NewUsers,
                                   unsigned Region) {
  Uses[Region].insert_range(NewUsers);
}

void Rematerializer::Reg::eraseUser(MachineInstr *MI, unsigned Region) {
  RegionUsers &RUsers = Uses.at(Region);
  assert(RUsers.contains(MI) && "user not in region");
  if (RUsers.size() == 1)
    Uses.erase(Region);
  else
    RUsers.erase(MI);
}

Printable Rematerializer::printDependencyDAG(RegisterIdx RootIdx) const {
  return Printable([&, RootIdx](raw_ostream &OS) {
    DenseMap<RegisterIdx, unsigned> RegDepths;
    std::function<void(RegisterIdx, unsigned)> WalkTree =
        [&](RegisterIdx RegIdx, unsigned Depth) -> void {
      unsigned MaxDepth = std::max(RegDepths.lookup_or(RegIdx, Depth), Depth);
      RegDepths.emplace_or_assign(RegIdx, MaxDepth);
      for (const Reg::Dependency &Dep : getReg(RegIdx).Dependencies)
        WalkTree(Dep.RegIdx, Depth + 1);
    };
    WalkTree(RootIdx, 0);

    // Sort in decreasing depth order to print root at the bottom.
    SmallVector<std::pair<RegisterIdx, unsigned>> Regs(RegDepths.begin(),
                                                       RegDepths.end());
    sort(Regs, [](const auto &LHS, const auto &RHS) {
      return LHS.second > RHS.second;
    });

    OS << printID(RootIdx) << " has " << Regs.size() - 1 << " dependencies\n";
    for (const auto &[RegIdx, Depth] : Regs) {
      OS << indent(Depth, 2) << (Depth ? '|' : '*') << ' '
         << printRematReg(RegIdx, /*SkipRegions=*/Depth) << '\n';
    }
    OS << printRegUsers(RootIdx);
  });
}

Printable Rematerializer::printID(RegisterIdx RegIdx) const {
  return Printable([&, RegIdx](raw_ostream &OS) {
    const Reg &PrintReg = getReg(RegIdx);
    OS << '(' << RegIdx << '/';
    if (!PrintReg.DefMI) {
      OS << "<dead>";
    } else {
      OS << printReg(PrintReg.getDefReg(), &TRI,
                     PrintReg.DefMI->getOperand(0).getSubReg(), &MRI);
    }
    OS << ")[" << PrintReg.DefRegion << "]";
  });
}

Printable Rematerializer::printRematReg(RegisterIdx RegIdx,
                                        bool SkipRegions) const {
  return Printable([&, RegIdx, SkipRegions](raw_ostream &OS) {
    const Reg &PrintReg = getReg(RegIdx);
    if (!SkipRegions) {
      OS << printID(RegIdx) << " [" << PrintReg.DefRegion;
      if (!PrintReg.Uses.empty()) {
        assert(PrintReg.DefMI && "dead register cannot have uses");
        const LiveInterval &LI = LIS.getInterval(PrintReg.getDefReg());
        // First display all regions in which the register is live-through and
        // not used.
        bool First = true;
        for (const auto [I, Bounds] : enumerate(Regions)) {
          if (Bounds.first == Bounds.second)
            continue;
          if (!PrintReg.Uses.contains(I) &&
              LI.liveAt(LIS.getInstructionIndex(*Bounds.first)) &&
              LI.liveAt(LIS.getInstructionIndex(*std::prev(Bounds.second))
                            .getRegSlot())) {
            OS << (First ? " - " : ",") << I;
            First = false;
          }
        }
        OS << (First ? " --> " : " -> ");

        // Then display regions in which the register is used.
        auto It = PrintReg.Uses.begin();
        OS << It->first;
        while (++It != PrintReg.Uses.end())
          OS << "," << It->first;
      }
      OS << "] ";
    }
    OS << printID(RegIdx) << ' ';
    PrintReg.DefMI->print(OS, /*IsStandalone=*/true, /*SkipOpers=*/false,
                          /*SkipDebugLoc=*/false, /*AddNewLine=*/false);
    OS << " @ ";
    LIS.getInstructionIndex(*PrintReg.DefMI).print(OS);
  });
}

Printable Rematerializer::printRegUsers(RegisterIdx RegIdx) const {
  return Printable([&, RegIdx](raw_ostream &OS) {
    for (const auto &[UseRegion, Users] : getReg(RegIdx).Uses) {
      for (MachineInstr *MI : Users)
        OS << "  User " << printUser(MI, UseRegion) << '\n';
    }
  });
}

Printable Rematerializer::printUser(const MachineInstr *MI,
                                    std::optional<unsigned> UseRegion) const {
  return Printable([&, MI, UseRegion](raw_ostream &OS) {
    RegisterIdx RegIdx = getDefRegIdx(*MI);
    if (RegIdx != NoReg) {
      OS << printID(RegIdx);
    } else {
      OS << "(-/-)[";
      if (UseRegion)
        OS << *UseRegion;
      else
        OS << '?';
      OS << ']';
    }
    OS << ' ';
    MI->print(OS, /*IsStandalone=*/true, /*SkipOpers=*/false,
              /*SkipDebugLoc=*/false, /*AddNewLine=*/false);
    OS << " @ ";
    LIS.getInstructionIndex(*MI).print(OS);
  });
}

Rollbacker::RollbackInfo::RollbackInfo(const Rematerializer &Remater,
                                       RegisterIdx RegIdx) {
  const Rematerializer::Reg &Reg = Remater.getReg(RegIdx);
  DefReg = Reg.getDefReg();
  DefRegion = Reg.DefRegion;
  Dependencies = Reg.Dependencies;

  InsertPos = std::next(Reg.DefMI->getIterator());
  if (InsertPos != Reg.DefMI->getParent()->end())
    NextRegIdx = Remater.getDefRegIdx(*InsertPos);
  else
    NextRegIdx = Rematerializer::NoReg;
}

void Rollbacker::rematerializerNoteRegCreated(const Rematerializer &Remater,
                                              RegisterIdx RegIdx) {
  if (RollingBack)
    return;
  Rematerializations[Remater.getOriginOf(RegIdx)].insert(RegIdx);
}

void Rollbacker::rematerializerNoteRegDeleted(const Rematerializer &Remater,
                                              RegisterIdx RegIdx) {
  if (RollingBack || Remater.isRematerializedRegister(RegIdx))
    return;
  DeadRegs.try_emplace(RegIdx, Remater, RegIdx);
}

void Rollbacker::rollback(Rematerializer &Remater) {
  RollingBack = true;

  // Re-create deleted registers.
  for (auto &[RegIdx, Info] : DeadRegs) {
    assert(!Remater.getReg(RegIdx).isAlive() && "register should be dead");

    // The MI that was originally just after the MI defining the register we
    // are trying to re-create may have been deleted. In such cases, we can
    // re-create at that MI's own insert position (and apply the same logic
    // recursively).
    MachineBasicBlock::iterator InsertPos = Info.InsertPos;
    RegisterIdx NextRegIdx = Info.NextRegIdx;
    while (NextRegIdx != Rematerializer::NoReg) {
      const auto *NextRegRollback = DeadRegs.find(NextRegIdx);
      if (NextRegRollback == DeadRegs.end())
        break;
      InsertPos = NextRegRollback->second.InsertPos;
      NextRegIdx = NextRegRollback->second.NextRegIdx;
    }
    Remater.recreateReg(RegIdx, Info.DefRegion, InsertPos, Info.DefReg,
                        std::move(Info.Dependencies));
  }

  // Rollback rematerializations.
  for (const auto &[RegIdx, RematsOf] : Rematerializations) {
    for (RegisterIdx RematRegIdx : RematsOf) {
      // It is possible that rematerializations were deleted. Their users would
      // have been transfered to some other rematerialization so we can safely
      // ignore them. Original registers that were deleted were just re-created
      // so we do not need to check for that.
      if (Remater.getReg(RematRegIdx).isAlive())
        Remater.transferAllUsers(RematRegIdx, RegIdx);
    }
  }

  Remater.updateLiveIntervals();
  DeadRegs.clear();
  Rematerializations.clear();
  RollingBack = false;
}
