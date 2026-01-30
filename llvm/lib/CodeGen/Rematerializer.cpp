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
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "rematerializer"

using namespace llvm;

static bool isAvailableAtUse(const VNInfo *OVNI, LaneBitmask Mask,
                             SlotIndex UseIdx, const LiveInterval &LI) {
  assert(OVNI);
  if (OVNI != LI.getVNInfoAt(UseIdx))
    return false;

  // Check that subrange is live at user.
  if (LI.hasSubRanges()) {
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

static Register isRegDependency(const MachineOperand &MO) {
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

unsigned Rematerializer::rematerializeToRegion(unsigned RootIdx,
                                               unsigned UseRegion,
                                               bool SupportRollback,
                                               DependencyReuseInfo &DRI) {

  MachineInstr *FirstMI =
      getReg(RootIdx).getRegionUseBounds(UseRegion, LIS).first;
  unsigned NewRegIdx = rematerializeToPos(RootIdx, FirstMI, DRI);
  transferRegionUsers(RootIdx, NewRegIdx, UseRegion, SupportRollback);
  return NewRegIdx;
}

unsigned
Rematerializer::rematerializeToPos(unsigned RootIdx,
                                   MachineBasicBlock::iterator InsertPos,
                                   DependencyReuseInfo &DRI) {
  LLVM_DEBUG({
    rdbgs() << "Rematerializing " << printID(RootIdx) << " to "
            << printUser(&*InsertPos) << '\n';
    ++CallDepth;
  });

  // Create/identify dependencies for the new register. Copy the dependencies
  // vector because underlying updates to the backing vector of registers may
  // invalidate references.
  SmallVector<Reg::Dependency, 2> NewDeps, Deps(Regs[RootIdx].Dependencies);
  for (const Reg::Dependency &Dep : Deps) {
    if (auto NewDep = DRI.DependencyMap.find(Dep.RegIdx);
        NewDep != DRI.DependencyMap.end()) {
      // We already have the version of the dependency we want to use.
      NewDeps.emplace_back(Dep.MOIdx, NewDep->second);
    } else {
      // Dependencies must be rematerialized in def-use order.
      unsigned NewDepIdx = rematerializeToPos(Dep.RegIdx, InsertPos, DRI);
      DRI.DependencyMap.insert({Dep.RegIdx, NewDepIdx});
      NewDeps.emplace_back(Dep.MOIdx, NewDepIdx);
    }
  }

  LLVM_DEBUG(--CallDepth);
  return rematerializeReg(RootIdx, InsertPos, std::move(NewDeps));
}

void Rematerializer::rollbackRematsOf(unsigned RootIdx) {
  auto Remats = Rematerializations.find(RootIdx);
  if (Remats == Rematerializations.end())
    return;

  LLVM_DEBUG({
    rdbgs() << "Rolling back rematerializations of " << printID(RootIdx)
            << '\n';
    ++CallDepth;
  });

  reviveRegIfDead(RootIdx);
  // All of the rematerialization's users must use the revived register.
  for (unsigned RematRegIdx : Remats->getSecond()) {
    for (const auto &[UseRegion, RegionUsers] : Regs[RematRegIdx].Uses) {
      transferRegionUsers(RematRegIdx, RootIdx, UseRegion,
                          /*SupportRollback=*/false);
    }
  }
  Rematerializations.erase(RootIdx);

  LLVM_DEBUG({
    rdbgs() << "** Rolled back rematerializations of " << printID(RootIdx)
            << '\n';
    --CallDepth;
  });
}

void Rematerializer::rollback(unsigned RematIdx) {
  assert(getReg(RematIdx).DefMI && !Rollbackable.contains(RematIdx) &&
         "cannot rollback dead register");
  const unsigned OriginRegIdx = getOriginOf(RematIdx);
  reviveRegIfDead(OriginRegIdx);
  for (const auto &[UseRegion, RegionUsers] : Regs[RematIdx].Uses) {
    transferRegionUsers(RematIdx, OriginRegIdx, UseRegion,
                        /*SupportRollback=*/false);
  }
}

void Rematerializer::reviveRegIfDead(unsigned RootIdx) {
  assert(!isRematerializedRegister(RootIdx) &&
         "cannot revive rematerialization");

  Reg &Root = Regs[RootIdx];
  if (!Root.Uses.empty()) {
    // The register still exists, nothing to do.
    LLVM_DEBUG(rdbgs() << printID(RootIdx) << " still exists\n");
    return;
  }

  assert(Rollbackable.contains(RootIdx) && "not marked rollbackable");
  assert(Root.DefMI && Root.DefMI->getOpcode() == TargetOpcode::DBG_VALUE &&
         "not the right opcode");
  assert(Rematerializations.contains(RootIdx) && "no remats");

  LLVM_DEBUG({
    rdbgs() << "Partially rolling back " << printID(RootIdx) << '\n';
    ++CallDepth;
  });

  // Fully rematerialized dependencies need to be revived. All dependencies gain
  // a new user.
  for (const Reg::Dependency &Dep : Root.Dependencies) {
    reviveRegIfDead(Dep.RegIdx);
    Regs[Dep.RegIdx].addUser(Root.DefMI, Root.DefRegion);
    LISUpdates.insert(Dep.RegIdx);
  }

  // Pick any rematerialization to retrieve the original opcode from.
  unsigned RematIdx = *Rematerializations.at(RootIdx).begin();
  Root.DefMI->setDesc(getReg(RematIdx).DefMI->getDesc());
  for (const auto &[MOIdx, Reg] : Rollbackable.at(RootIdx))
    Root.DefMI->getOperand(MOIdx).setReg(Reg);
  Rollbackable.erase(RootIdx);
  LISUpdates.insert(RootIdx);

  LLVM_DEBUG({
    rdbgs() << "** Partially rolled back " << printID(RootIdx) << " @ ";
    LIS.getInstructionIndex(*Root.DefMI).print(dbgs());
    dbgs() << '\n';
    --CallDepth;
  });
}

void Rematerializer::transferUser(unsigned FromRegIdx, unsigned ToRegIdx,
                                  MachineInstr &UserMI, bool SupportRollback) {
  transferUserInternal(FromRegIdx, ToRegIdx, UserMI);
  unsigned UserRegion = MIRegion[&UserMI];
  Regs[FromRegIdx].eraseUser(&UserMI, UserRegion);
  Regs[ToRegIdx].addUser(&UserMI, UserRegion);
  deleteRegIfUnused(FromRegIdx, SupportRollback);
}

void Rematerializer::transferRegionUsers(unsigned FromRegIdx, unsigned ToRegIdx,
                                         unsigned UseRegion,
                                         bool SupportRollback) {
  auto &FromRegUsers = Regs[FromRegIdx].Uses;
  auto UsesIt = FromRegUsers.find(UseRegion);
  if (UsesIt == FromRegUsers.end())
    return;

  const SmallDenseSet<MachineInstr *, 4> &RegionUsers = UsesIt->getSecond();
  for (MachineInstr *UserMI : RegionUsers)
    transferUserInternal(FromRegIdx, ToRegIdx, *UserMI);
  Regs[ToRegIdx].addUsers(RegionUsers, UseRegion);
  FromRegUsers.erase(UseRegion);
  deleteRegIfUnused(FromRegIdx, SupportRollback);
}

void Rematerializer::transferUserInternal(unsigned FromRegIdx,
                                          unsigned ToRegIdx,
                                          MachineInstr &UserMI) {
  assert(MIRegion.contains(&UserMI) && "unknown user");
  assert(getReg(FromRegIdx).Uses.at(MIRegion.at(&UserMI)).contains(&UserMI) &&
         "not a user");
  assert(FromRegIdx != ToRegIdx && "identical registers");
  assert(getOriginOrSelf(FromRegIdx) == getOriginOrSelf(ToRegIdx) &&
         "unrelated registers");

  LLVM_DEBUG(rdbgs() << "User transfer from " << printID(FromRegIdx) << " to "
                     << printID(ToRegIdx) << ": " << printUser(&UserMI)
                     << '\n');

  UserMI.substituteRegister(getReg(FromRegIdx).getDefReg(),
                            getReg(ToRegIdx).getDefReg(), 0, TRI);
  LISUpdates.insert(FromRegIdx);
  LISUpdates.insert(ToRegIdx);

  // If the user is rematerializable, we must change its dependency to the
  // new register.
  if (unsigned UserRegIdx = getDefRegIdx(UserMI); UserRegIdx != NoReg) {
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
  for (unsigned RegIdx : LISUpdates) {
    const Reg &UpdateReg = getReg(RegIdx);
    assert(UpdateReg.DefMI || Rollbackable.contains(RegIdx) && "dead register");

    Register DefReg = UpdateReg.getDefReg();
    if (LIS.hasInterval(DefReg))
      LIS.removeInterval(DefReg);
    LIS.createAndComputeVirtRegInterval(DefReg);

    LLVM_DEBUG({
      rdbgs() << "Re-computed interval for " << printID(RegIdx) << ": ";
      LIS.getInterval(DefReg).print(dbgs());
      rdbgs() << '\n' << printRegUsers(RegIdx);
    });

    // Update intervals for unrematerializable operands.
    for (unsigned MOIdx : getUnrematableOprds(RegIdx)) {
      Register UnrematReg = UpdateReg.DefMI->getOperand(MOIdx).getReg();
      if (!SeenUnrematRegs.insert(UnrematReg).second)
        continue;
      LIS.removeInterval(UnrematReg);
      LIS.createAndComputeVirtRegInterval(UnrematReg);
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

void Rematerializer::commitRematerializations() {
  for (auto &[RegIdx, _] : Rollbackable)
    deleteReg(RegIdx);
  Rollbackable.clear();
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
    if (!isAvailableAtUse(DefVN, Mask, Use, LI))
      return false;
  }
  return true;
}

unsigned Rematerializer::findRematInRegion(unsigned RegIdx, unsigned Region,
                                           SlotIndex Before) const {
  auto It = Rematerializations.find(getOriginOrSelf(RegIdx));
  if (It == Rematerializations.end())
    return NoReg;
  const SmallDenseSet<unsigned, 4> &Remats = It->getSecond();

  SlotIndex BestSlot;
  unsigned BestRegIdx = NoReg;
  for (unsigned RematRegIdx : Remats) {
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

bool Rematerializer::deleteRegIfUnused(unsigned RootIdx, bool SupportRollback) {
  Reg &Root = Regs[RootIdx];
  if (!Root.Uses.empty())
    return false;
  LLVM_DEBUG({
    rdbgs() << "Deleting " << printID(RootIdx) << " with no users\n";
    ++CallDepth;
  });

  Register DefReg = Root.getDefReg();
  for (const Reg::Dependency &Dep : Root.Dependencies) {
    LLVM_DEBUG(rdbgs() << "Deleting user from " << printID(Dep.RegIdx) << "\n");
    Regs[Dep.RegIdx].eraseUser(Root.DefMI, Root.DefRegion);
    deleteRegIfUnused(Dep.RegIdx, SupportRollback);
  }

  LIS.removeInterval(DefReg);
  LISUpdates.erase(RootIdx);
  if (SupportRollback) {
    // Replace all read registers with the null one to prevent issues in live
    // interval calculations. Store mappings between operand indices and
    // original registers for potential rolqlback.
    DenseMap<unsigned, Register> &RegMap =
        Rollbackable.try_emplace(RootIdx).first->getSecond();
    for (auto [Idx, MO] : enumerate(Root.DefMI->operands())) {
      if (MO.isReg() && MO.readsReg()) {
        RegMap.insert({Idx, MO.getReg()});
        MO.setReg(Register());
      }
    }
    Root.DefMI->setDesc(TII.get(TargetOpcode::DBG_VALUE));
  } else {
    deleteReg(RootIdx);
  }
  if (isRematerializedRegister(RootIdx)) {
    SmallDenseSet<unsigned, 4> &Remats =
        Rematerializations.at(getOriginOf(RootIdx));
    assert(Remats.contains(RootIdx) && "broken link between remat and origin");
    Remats.erase(RootIdx);
    if (Remats.empty())
      Rematerializations.erase(RootIdx);
  }
  LLVM_DEBUG(--CallDepth);
  return true;
}

void Rematerializer::deleteReg(unsigned RegIdx) {
  Reg &DeleteReg = Regs[RegIdx];
  assert(DeleteReg.DefMI && "register was already deleted");
  // It is not possible for the deleted instruction to be the upper region
  // boundary since we don't ever consider them rematerializable.
  if (Regions[DeleteReg.DefRegion].first == DeleteReg.DefMI)
    Regions[DeleteReg.DefRegion].first =
        std::next(MachineBasicBlock::iterator(DeleteReg.DefMI));
  LIS.RemoveMachineInstrFromMaps(*DeleteReg.DefMI);
  DeleteReg.DefMI->eraseFromParent();
  MIRegion.erase(DeleteReg.DefMI);
  DeleteReg.DefMI = nullptr;
}

bool Rematerializer::analyze() {
  MIRegion.clear();
  Regs.clear();
  RegToIdx.clear();
  LISUpdates.clear();
  Rollbackable.clear();
  if (Regions.empty())
    return false;

  // Maps each basic block number to regions that are part of the BB.
  DenseMap<unsigned, SmallVector<unsigned, 4>> RegionsPerBlock;

  const unsigned NumRegions = Regions.size();
  for (unsigned I = 0; I < NumRegions; ++I) {
    RegionBoundaries Region = Regions[I];
    for (auto MI = Region.first; MI != Region.second; ++MI)
      MIRegion.insert({&*MI, I});
    MachineBasicBlock *MBB = Region.first->getParent();
    if (Region.second != MBB->end())
      MIRegion.insert({&*Region.second, I});
    RegionsPerBlock[MBB->getNumber()].push_back(I);
  }

  // Visit regions in dominator tree pre-order to ensure that regions defining
  // registers come before regions using them.
  MachineDominatorTree MDT(MF);
  for (MachineDomTreeNode *MBB : depth_first(&MDT)) {
    auto MBBRegions = RegionsPerBlock.find(MBB->getBlock()->getNumber());
    if (MBBRegions == RegionsPerBlock.end())
      continue;
    auto MBBRegionsIt = RegionsTopDown ? MBBRegions->getSecond()
                                       : reverse(MBBRegions->getSecond());
    for (unsigned I : MBBRegionsIt)
      collectRegs(I);
  }

  LLVM_DEBUG({
    for (unsigned I = 0, E = getNumRegs(); I < E; ++I)
      dbgs() << printDependencyDAG(I) << '\n';
  });
  return !Regs.empty();
}

void Rematerializer::collectRegs(unsigned DefRegion) {
  // Collect partially rematerializable registers in instruction order within
  // each region. This guarantees that, within a single region, partially
  // rematerializable registers used in instructions defining other partially
  // rematerializable registers are visited first. This is important to
  // guarantee that all of a register's dependencies are visited before the
  // register itself.
  RegionBoundaries Bounds = Regions[DefRegion];
  for (auto MI = Bounds.first; MI != Bounds.second; ++MI) {
    MachineInstr &DefMI = *MI;
    if (!isMIRematerializable(DefMI))
      continue;

    Reg &CurrentReg = Regs.emplace_back();
    CurrentReg.DefMI = &DefMI;
    CurrentReg.DefRegion = DefRegion;
    Register DefReg = CurrentReg.getDefReg();
    unsigned SubIdx = DefMI.getOperand(0).getSubReg();
    CurrentReg.Mask = SubIdx ? TRI.getSubRegIndexLaneMask(SubIdx)
                             : MRI.getMaxLaneMaskForVReg(DefReg);

    // Collect the candidate's direct users, both rematerializable and
    // unrematerializable.
    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(DefReg)) {
      auto UseRegion = MIRegion.find(&UseMI);
      if (UseRegion == MIRegion.end()) {
        // Only lone MI terminators can trigger this condition. They are not
        // part of any region so we cannot rematerialize next to them. Just
        // consider this register unrematerializable.
        CurrentReg.Uses.clear();
        break;
      }
      CurrentReg.addUser(&UseMI, UseRegion->second);
    }
    if (CurrentReg.Uses.empty()) {
      Regs.pop_back();
      continue;
    }

    // Collect the candidate's dependencies. If the same register is used
    // multiple times we just need to store it once.
    SmallDenseSet<Register, 4> AllDepRegs;
    SmallVector<unsigned, 2> &Unrematable = UnrematableOprds.emplace_back();
    for (const auto &[MOIdx, MO] : enumerate(CurrentReg.DefMI->operands())) {
      Register DepReg = isRegDependency(MO);
      if (!DepReg || !AllDepRegs.insert(DepReg).second)
        continue;
      if (auto DepIt = RegToIdx.find(DepReg); DepIt != RegToIdx.end()) {
        Reg::Dependency Dep(MOIdx, DepIt->second);
        CurrentReg.Dependencies.push_back(Dep);
      } else
        Unrematable.push_back(MOIdx);
    }

    // The register is rematerializable.
    RegToIdx.insert({DefReg, Regs.size() - 1});
  }

  assert(Regs.size() == UnrematableOprds.size());
}

bool Rematerializer::isMIRematerializable(const MachineInstr &MI) const {
  if (!TII.isReMaterializable(MI))
    return false;

  for (const MachineOperand &MO : MI.all_uses()) {
    // We can't remat physreg uses, unless it is a constant or an ignorable
    // use (e.g. implicit exec use on VALU instructions)
    if (MO.getReg().isPhysical()) {
      if (MRI.isConstantPhysReg(MO.getReg()) || TII.isIgnorableUse(MO))
        continue;
      return false;
    }
  }

  // We only support rematerializing virtual registers with one definition.
  Register DefReg = MI.getOperand(0).getReg();
  return DefReg.isVirtual() && MRI.hasOneDef(DefReg);
}

unsigned Rematerializer::getDefRegIdx(const MachineInstr &MI) const {
  if (!MI.getNumOperands() || !MI.getOperand(0).isReg() ||
      MI.getOperand(0).readsReg())
    return NoReg;
  Register Reg = MI.getOperand(0).getReg();
  auto UserRegIt = RegToIdx.find(Reg);
  if (UserRegIt == RegToIdx.end())
    return NoReg;
  return UserRegIt->second;
}

unsigned Rematerializer::rematerializeReg(
    unsigned RegIdx, MachineBasicBlock::iterator InsertPos,
    SmallVectorImpl<Reg::Dependency> &&Dependencies) {
  unsigned UseRegion = MIRegion.at(&*InsertPos);
  unsigned NewRegIdx = Regs.size();

  Reg &NewReg = Regs.emplace_back();
  Reg &FromReg = Regs[RegIdx];
  NewReg.Mask = FromReg.Mask;
  NewReg.DefRegion = UseRegion;
  NewReg.Dependencies = std::move(Dependencies);

  // Track rematerialization link between registers. Origins are always
  // registers that existed originally, and rematerializations are always
  // attached to them.
  unsigned OriginIdx =
      isRematerializedRegister(RegIdx) ? getOriginOf(RegIdx) : RegIdx;
  Origins.push_back(OriginIdx);
  Rematerializations[OriginIdx].insert(NewRegIdx);

  // Use the TII to rematerialize the defining instruction with a new defined
  // register.
  Register NewDefReg = MRI.cloneVirtualRegister(FromReg.getDefReg());
  TII.reMaterialize(*InsertPos->getParent(), InsertPos, NewDefReg, 0,
                    *FromReg.DefMI);
  NewReg.DefMI = &*std::prev(InsertPos);
  RegToIdx.insert({NewDefReg, NewRegIdx});

  // Update the DAG.
  RegionBoundaries &Bounds = Regions[UseRegion];
  if (Bounds.first == std::next(MachineBasicBlock::iterator(NewReg.DefMI)))
    Bounds.first = NewReg.DefMI;
  LIS.InsertMachineInstrInMaps(*NewReg.DefMI);
  MIRegion.emplace_or_assign(NewReg.DefMI, UseRegion);
  LISUpdates.insert(NewRegIdx);

  // Replace dependencies as needed in the rematerialized MI. All dependencies
  // of the latter gain a new user.
  auto ZipedDeps = zip_equal(FromReg.Dependencies, NewReg.Dependencies);
  for (const auto &[OldDep, NewDep] : ZipedDeps) {
    assert(OldDep.MOIdx == NewDep.MOIdx && "operand mismatch");
    LLVM_DEBUG(rdbgs() << "  Operand #" << OldDep.MOIdx << ": "
                       << printID(OldDep.RegIdx) << " -> "
                       << printID(NewDep.RegIdx) << '\n');

    Reg &NewDepReg = Regs[NewDep.RegIdx];
    if (OldDep.RegIdx != NewDep.RegIdx) {
      Register OldDefReg = FromReg.DefMI->getOperand(OldDep.MOIdx).getReg();
      NewReg.DefMI->substituteRegister(OldDefReg, NewDepReg.getDefReg(), 0,
                                       TRI);
      LISUpdates.insert(OldDep.RegIdx);
    }
    NewDepReg.addUser(NewReg.DefMI, UseRegion);
    LISUpdates.insert(NewDep.RegIdx);
  }

  LLVM_DEBUG({
    rdbgs() << "** Rematerialized " << printID(RegIdx) << " as "
            << printRematReg(NewRegIdx) << '\n';
  });
  return NewRegIdx;
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
  assert(Uses.contains(Region) && "no user in region");
  assert(Uses.at(Region).contains(MI) && "user not in region");
  RegionUsers &RUsers = Uses[Region];
  if (RUsers.size() == 1)
    Uses.erase(Region);
  else
    RUsers.erase(MI);
}

Printable Rematerializer::printDependencyDAG(unsigned RootIdx) const {
  return Printable([&, RootIdx](raw_ostream &OS) {
    DenseMap<unsigned, unsigned> RegDepths;
    std::function<void(unsigned, unsigned)> WalkTree =
        [&](unsigned RegIdx, unsigned Depth) -> void {
      unsigned MaxDepth = std::max(RegDepths.lookup_or(RegIdx, Depth), Depth);
      RegDepths.emplace_or_assign(RegIdx, MaxDepth);
      for (const Reg::Dependency &Dep : getReg(RegIdx).Dependencies)
        WalkTree(Dep.RegIdx, Depth + 1);
    };
    WalkTree(RootIdx, 0);

    // Sort in decreasing depth order to print root at the bottom.
    SmallVector<std::pair<unsigned, unsigned>> Regs(RegDepths.begin(),
                                                    RegDepths.end());
    sort(Regs, [](const auto &LHS, const auto &RHS) {
      return LHS.second > RHS.second;
    });

    OS << printID(RootIdx) << " has " << Regs.size() - 1 << " dependencies\n";
    for (const auto &[RegIdx, Depth] : Regs) {
      std::string Shift(2 * Depth, ' ');
      OS << Shift << (Depth ? '|' : '*') << ' '
         << printRematReg(RegIdx, /*SkipRegions=*/Depth) << '\n';
    }
    OS << printRegUsers(RootIdx);
  });
}

Printable Rematerializer::printID(unsigned RegIdx) const {
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

Printable Rematerializer::printRematReg(unsigned RegIdx,
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

Printable Rematerializer::printRegUsers(unsigned RegIdx) const {
  return Printable([&, RegIdx](raw_ostream &OS) {
    for (const auto &[_, Users] : getReg(RegIdx).Uses) {
      for (MachineInstr *MI : Users)
        dbgs() << "  User " << printUser(MI) << '\n';
    }
  });
}

Printable Rematerializer::printUser(const MachineInstr *MI) const {
  return Printable([&, MI](raw_ostream &OS) {
    unsigned RegIdx = getDefRegIdx(*MI);
    if (RegIdx != NoReg)
      OS << printID(RegIdx);
    else
      OS << "(-/-)[" << MIRegion.at(MI) << ']';
    OS << ' ';
    MI->print(OS, /*IsStandalone=*/true, /*SkipOpers=*/false,
              /*SkipDebugLoc=*/false, /*AddNewLine=*/false);
    OS << " @ ";
    LIS.getInstructionIndex(*MI).print(dbgs());
  });
}
