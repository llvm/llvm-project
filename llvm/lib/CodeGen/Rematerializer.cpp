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
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
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

  SmallVector<RegisterIdx, 2> NewDeps;
  // Copy all dependencies because recursive rematerialization of dependencies
  // may invalidate references to the backing vector of registers.
  SmallVector<RegisterIdx, 2> OldDeps(getReg(RootIdx).Dependencies);
  for (RegisterIdx DepRegIdx : OldDeps) {
    // Recursively rematerialize required dependencies at the same position as
    // the root. Registers form a DAG so the recursion is guaranteed to
    // terminate.
    auto RematIdx = DRI.DependencyMap.find(DepRegIdx);
    RegisterIdx NewDepRegIdx;
    if (RematIdx == DRI.DependencyMap.end())
      NewDepRegIdx = rematerializeToPos(DepRegIdx, UseRegion, InsertPos, DRI);
    else
      NewDepRegIdx = RematIdx->second;
    NewDeps.push_back(NewDepRegIdx);
  }
  RegisterIdx NewIdx =
      rematerializeReg(RootIdx, UseRegion, InsertPos, std::move(NewDeps));
  DRI.DependencyMap.insert({RootIdx, NewIdx});
  return NewIdx;
}

void Rematerializer::transferUser(RegisterIdx FromRegIdx, RegisterIdx ToRegIdx,
                                  unsigned UserRegion, MachineInstr &UserMI) {
  transferUserImpl(FromRegIdx, ToRegIdx, UserMI);

  Regs[ToRegIdx].addUser(&UserMI, UserRegion);
  extendToNewUsers(ToRegIdx, &UserMI);

  Regs[FromRegIdx].eraseUser(&UserMI, UserRegion);
  shrinkToUses(FromRegIdx);
}

void Rematerializer::transferRegionUsers(RegisterIdx FromRegIdx,
                                         RegisterIdx ToRegIdx,
                                         unsigned UseRegion) {
  Reg &FromReg = Regs[FromRegIdx];
  auto UsesIt = FromReg.Uses.find(UseRegion);
  if (UsesIt == FromReg.Uses.end())
    return;

  const SmallDenseSet<MachineInstr *, 4> &RegionUsers = UsesIt->getSecond();
  SmallVector<MachineInstr *, 4> NewUsers;
  for (MachineInstr *UserMI : RegionUsers) {
    transferUserImpl(FromRegIdx, ToRegIdx, *UserMI);
    NewUsers.push_back(UserMI);
  }

  extendToNewUsers(ToRegIdx, NewUsers);
  Regs[ToRegIdx].addUsers(RegionUsers, UseRegion);

  FromReg.Uses.erase(UseRegion);
  shrinkToUses(FromRegIdx);
}

void Rematerializer::transferAllUsers(RegisterIdx FromRegIdx,
                                      RegisterIdx ToRegIdx) {
  Reg &FromReg = Regs[FromRegIdx];
  SmallVector<MachineInstr *, 4> NewUsers;
  for (const auto &[UseRegion, RegionUsers] : FromReg.Uses) {
    for (MachineInstr *UserMI : RegionUsers) {
      transferUserImpl(FromRegIdx, ToRegIdx, *UserMI);
      NewUsers.push_back(UserMI);
    }
    Regs[ToRegIdx].addUsers(RegionUsers, UseRegion);
  }
  extendToNewUsers(ToRegIdx, NewUsers);

  FromReg.Uses.clear();
  deleteReg(FromRegIdx);
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

  // If the user is rematerializable, we must change its dependency to the
  // new register.
  if (RegisterIdx UserRegIdx = getDefRegIdx(UserMI); UserRegIdx != NoReg) {
    // Look for the user's dependency that matches the register.
    for (RegisterIdx &DepRegIdx : Regs[UserRegIdx].Dependencies) {
      if (DepRegIdx == FromRegIdx) {
        DepRegIdx = ToRegIdx;
        return;
      }
    }
    llvm_unreachable("broken dependency");
  }
}

bool Rematerializer::isMOIdenticalAtUses(MachineOperand &MO,
                                         ArrayRef<SlotIndex> Uses) const {
  unsigned SubIdx = MO.getSubReg();
  LaneBitmask Mask = SubIdx ? TRI.getSubRegIndexLaneMask(SubIdx)
                            : MRI.getMaxLaneMaskForVReg(MO.getReg());
  return isRegIdenticalAtUses(
      MO.getReg(), Mask,
      LIS.getInstructionIndex(*MO.getParent()).getRegSlot(true), Uses);
}

bool Rematerializer::isRegIdenticalAtUses(Register Reg, LaneBitmask Mask,
                                          SlotIndex RefSlot,
                                          ArrayRef<SlotIndex> Uses) const {
  if (Uses.empty())
    return true;
  const LiveInterval &LI = LIS.getInterval(Reg);
  const VNInfo *DefVN = LI.getVNInfoAt(RefSlot);
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

void Rematerializer::deleteReg(RegisterIdx RootIdx) {
  assert(getReg(RootIdx).Uses.empty() && "register still has uses");

  // Traverse the root's dependency DAG depth-first to find the set of registers
  // we can delete and a legal order to delete them in.
  SmallVector<RegisterIdx, 4> DepDAG{RootIdx};
  SmallVector<RegisterIdx, 8> DeleteOrder{RootIdx};
  do {
    // A deleted register's dependencies may be deletable too.
    const Reg &DeleteReg = getReg(DepDAG.pop_back_val());
    for (RegisterIdx DepRegIdx : DeleteReg.Dependencies) {
      // All dependencies lose a user (the deleted register).
      Reg &DepReg = Regs[DepRegIdx];
      DepReg.eraseUser(DeleteReg.DefMI, DeleteReg.DefRegion);
      if (DepReg.Uses.empty()) {
        DeleteOrder.push_back(DepRegIdx);
        DepDAG.push_back(DepRegIdx);
      }
    }
  } while (!DepDAG.empty());

  for (RegisterIdx RegIdx : DeleteOrder) {
    preDeletion(RegIdx);
    Reg &DeleteReg = Regs[RegIdx];
    Register DefReg = DeleteReg.getDefReg();
    LIS.RemoveMachineInstrFromMaps(*DeleteReg.DefMI);
    DeleteReg.DefMI->eraseFromParent();
    DeleteReg.DefMI = nullptr;
    LIS.removeInterval(DefReg);
  }

  SmallSet<RegisterIdx, 8> ShrinkRematRegs;
  SmallSet<Register, 8> ShrinkUnrematRegs;

  // All dependencies lose a user; their live interval could be shrunk.
  for (RegisterIdx DeletedRegIdx : DeleteOrder) {
    for (RegisterIdx DepRegIdx : getReg(DeletedRegIdx).Dependencies) {
      const Reg &DepReg = getReg(DepRegIdx);
      if (DepReg.isAlive() && ShrinkRematRegs.insert(DepRegIdx).second) {
        assert(!DepReg.Uses.empty() && "dep should have uses");
        shrinkToUses(DepRegIdx);
      }
    }
    for (const auto [Reg, Mask] : getUnrematableDeps(DeletedRegIdx)) {
      if (ShrinkUnrematRegs.insert(Reg).second)
        shrinkToUsesUnremat(Reg);
    }
  }
}

void Rematerializer::DeadDefDelegate::LRE_WillEraseInstruction(
    MachineInstr *MI) {
  RegisterIdx RegIdx = Remater.getDefRegIdx(*MI);
  if (RegIdx == Rematerializer::NoReg) {
    // This is an unrematerializable register.
    Remater.noteMIWillBeDeleted(*MI);
    LLVM_DEBUG(dbgs() << "** About to delete dead definition: " << *MI);

    // Do a linear scan through regions to figure out which one the about to be
    // deleted unrematerializable MI is a part of. This is expensive but should
    // happen extremely rarely.
    //
    // FIXME: the rematerializer should stop tracking regions and operate on a
    // machine basic block-basis. This would simplify this and a lot of the
    // tracking elsewhere.
    MachineBasicBlock::iterator It = MI->getIterator();
    const LiveIntervals &LIS = Remater.LIS;
    SlotIndex MISlot = LIS.getInstructionIndex(*MI);
    unsigned MIRegion = ~0U;
    for (auto [RegionIdx, Bounds] : enumerate(Remater.Regions)) {
      auto &[RegionBegin, RegionEnd] = Bounds;
      MachineBasicBlock::iterator FirstMI =
          skipDebugInstructionsForward(RegionBegin, RegionEnd);
      if (FirstMI == RegionEnd) {
        // The MI cannot be in an empty region.
        continue;
      }

      if (LIS.getInstructionIndex(*FirstMI) <= MISlot) {
        // FistMI exists inside the region so this is guaranteed to point to a
        // non-debug MI.
        MachineBasicBlock::iterator LastMI =
            skipDebugInstructionsBackward(std::prev(RegionEnd), RegionBegin);
        if (LIS.getInstructionIndex(*LastMI) < MISlot)
          continue;

        // We have found the region the MI is a part of.
        MIRegion = RegionIdx;
        if (RegionBegin == It)
          ++RegionBegin;
        break;
      }
    }

    // All rematerializable registers that this MI uses must be notified.
    SmallDenseSet<Register, 2> UsedRegs;
    for (const MachineOperand &MO : MI->all_uses()) {
      Register Reg = MO.getReg();
      if (Reg.isVirtual() && !UsedRegs.insert(Reg).second)
        continue;
      auto RematRegUse = Remater.RegToIdx.find(Reg);
      if (RematRegUse == Remater.RegToIdx.end())
        continue;
      assert(MIRegion != ~0U && "remat user cannot be outside regions");
      Remater.Regs[RematRegUse->second].eraseUser(MI, MIRegion);
    }
    return;
  }
  // This is a rematerializable register.

  // All rematerializable dependencies must be notified.
  Reg &DeleteReg = Remater.Regs[RegIdx];
  for (RegisterIdx DepRegIdx : DeleteReg.Dependencies)
    Remater.Regs[DepRegIdx].eraseUser(MI, DeleteReg.DefRegion);

  assert(DeleteReg.isAlive() && "register must be alive");
  assert(DeleteReg.Uses.empty() && "register should no longer have uses");

  // The live-range editor will delete the defining instruction from the MIR
  // as well as the register's live-range, so we just need to nullify the def
  // internally.
  Remater.preDeletion(RegIdx);
  DeleteReg.DefMI = nullptr;
}

void Rematerializer::preDeletion(RegisterIdx DeleteRegIdx) {
  Reg &DeleteReg = Regs[DeleteRegIdx];
  assert(DeleteReg.isAlive() && "register must still be alive");
  noteRegWillBeDeleted(DeleteRegIdx);
  LLVM_DEBUG(dbgs() << "** About to delete " << printID(DeleteRegIdx) << "\n");

  // Update region boundary if necessary. It is not possible for the deleted
  // instruction to be the upper region boundary since we don't ever consider
  // them rematerializable.
  MachineBasicBlock::iterator &RegionBegin = Regions[DeleteReg.DefRegion].first;
  if (RegionBegin == DeleteReg.DefMI)
    ++RegionBegin;

  if (isOriginalRegister(DeleteRegIdx))
    return;

  // Delete rematerialized register from its origin's rematerializations.
  const RegisterIdx OriginIdx = getOriginOf(DeleteRegIdx);
  RematsOf &OriginRemats = Rematerializations.at(OriginIdx);
  assert(OriginRemats.contains(DeleteRegIdx) && "broken remat<->origin link");
  OriginRemats.erase(DeleteRegIdx);
  if (OriginRemats.empty())
    Rematerializations.erase(OriginIdx);
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
  UnrematableDeps.clear();
  Origins.clear();
  Rematerializations.clear();
  RegionMBB.clear();
  RegToIdx.clear();
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
  assert(Regs.size() == UnrematableDeps.size());

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

  // Collect the candidate's dependencies, rematerializable or not. If the same
  // rematerializable register is used multiple times we just need to consider
  // it once.
  SmallSetVector<RegisterIdx, 2> RematDeps;
  SmallMapVector<Register, LaneBitmask, 2> UnrematDeps;
  for (const MachineOperand &MO : DefMI.all_uses()) {
    Register DepReg = getRegDependency(MO);
    if (!DepReg)
      continue;
    unsigned DepRegIdx = DepReg.virtRegIndex();
    if (!SeenRegs[DepRegIdx])
      addRegIfRematerializable(DepRegIdx, MIRegion, SeenRegs);
    if (auto DepIt = RegToIdx.find(DepReg); DepIt != RegToIdx.end()) {
      RematDeps.insert(DepIt->second);
    } else {
      LaneBitmask &CurrentMask =
          UnrematDeps.try_emplace(DepReg, LaneBitmask::getNone()).first->second;
      LaneBitmask Mask = MO.getSubReg()
                             ? TRI.getSubRegIndexLaneMask(MO.getSubReg())
                             : MRI.getMaxLaneMaskForVReg(DepReg);
      CurrentMask |= Mask;
    }
  }

  // The register is rematerializable.
  RematReg.Dependencies = RematDeps.takeVector();
  RegToIdx.insert({DefReg, Regs.size()});
  Regs.push_back(RematReg);
  UnrematableDeps.push_back(UnrematDeps.takeVector());
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

RegisterIdx
Rematerializer::rematerializeReg(RegisterIdx RegIdx, unsigned UseRegion,
                                 MachineBasicBlock::iterator InsertPos,
                                 SmallVectorImpl<RegisterIdx> &&Dependencies) {
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
  postRematerialization(RegIdx, NewRegIdx);

  noteRegCreated(NewRegIdx);
  LLVM_DEBUG(dbgs() << "** Rematerialized " << printID(RegIdx) << " as "
                    << printRematReg(NewRegIdx) << '\n');
  return NewRegIdx;
}

void Rematerializer::recreateReg(RegisterIdx RegIdx,
                                 MachineBasicBlock::iterator InsertPos,
                                 Register DefReg) {
  assert(RegToIdx.contains(DefReg) && "unknown defined register");
  assert(RegToIdx.at(DefReg) == RegIdx && "incorrect defined register");
  assert(!getReg(RegIdx).isAlive() && "register is still alive");

  Reg &OriginReg = Regs[RegIdx];

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
    assert(getReg(getOriginOf(RegIdx)).isAlive() && "expected alive origin");
    ModelRegIdx = getOriginOf(RegIdx);
  }
  const MachineInstr &ModelDefMI = *getReg(ModelRegIdx).DefMI;

  TII.reMaterialize(*RegionMBB[OriginReg.DefRegion], InsertPos, DefReg, 0,
                    ModelDefMI);
  OriginReg.DefMI = &*std::prev(InsertPos);
  postRematerialization(ModelRegIdx, RegIdx);
  LLVM_DEBUG(dbgs() << "** Recreated " << printID(RegIdx) << " as "
                    << printRematReg(RegIdx) << '\n');
}

void Rematerializer::postRematerialization(RegisterIdx ModelRegIdx,
                                           RegisterIdx RematRegIdx) {
  Reg &ModelReg = Regs[ModelRegIdx], &RematReg = Regs[RematRegIdx];

  // The rematerialization has no user at this point so its interval will
  // initially be empty.
  SlotIndex UseIdx = LIS.InsertMachineInstrInMaps(*RematReg.DefMI).getRegSlot();
  LIS.createAndComputeVirtRegInterval(RematReg.getDefReg());

  // The start of the new register's region may have changed.
  MachineBasicBlock::iterator &RegionBegin = Regions[RematReg.DefRegion].first;
  if (RegionBegin == std::next(MachineBasicBlock::iterator(RematReg.DefMI)))
    RegionBegin = RematReg.DefMI;

  // Replace dependencies as needed in the rematerialized MI. All dependencies
  // of the latter gain a new user.
  auto ZipedDeps = zip_equal(ModelReg.Dependencies, RematReg.Dependencies);
  for (const auto &[OldDepRegIdx, NewDepRegIdx] : ZipedDeps) {
    LLVM_DEBUG(dbgs() << "  Dependency: " << printID(OldDepRegIdx) << " -> "
                      << printID(NewDepRegIdx) << '\n');

    Reg &NewDepReg = Regs[NewDepRegIdx];
    if (OldDepRegIdx != NewDepRegIdx) {
      Reg &OldDepReg = Regs[OldDepRegIdx];
      RematReg.DefMI->substituteRegister(OldDepReg.getDefReg(),
                                         NewDepReg.getDefReg(), 0, TRI);
    }
    NewDepReg.addUser(RematReg.DefMI, RematReg.DefRegion);
    extendToNewUsers(NewDepRegIdx, RematReg.DefMI);
  }

  // Unrematerializable dependencies always gain a new user after a
  // rematerialization; their live range may need to be extended.
  for (const auto &[Reg, Mask] : getUnrematableDeps(ModelRegIdx))
    extendInterval(LIS.getInterval(Reg), Mask, UseIdx);
}

void Rematerializer::extendToNewUsers(RegisterIdx RegIdx,
                                      ArrayRef<MachineInstr *> NewUsers) const {
  if (NewUsers.empty())
    return;
  const Reg &ExtendReg = getReg(RegIdx);
  assert(ExtendReg.isAlive() && "register must be alive");

  Register DefReg = ExtendReg.getDefReg();
  LiveInterval &LI = LIS.getInterval(DefReg);
  const LaneBitmask FullLaneMask = MRI.getMaxLaneMaskForVReg(DefReg);
  const bool ShouldTrackSubReg = MRI.shouldTrackSubRegLiveness(DefReg);

  // When subreg liveness tracking is required but no subrange exists yet (e.g.,
  // the interval was computed with only a def of the entire register),
  // initialize subranges from the main range before extending them. This must
  // happen even if every new user reads the full mask, because other existing
  // users of the register may read individual subregs and later passes
  // (VirtRegRewriter) expect subranges to exist.
  if (!LI.hasSubRanges() && ShouldTrackSubReg)
    LI.createSubRangeFrom(LIS.getVNInfoAllocator(), FullLaneMask, LI);

  // Extend all ranges in the register's live interval so that they reach the
  // new users.
  for (MachineInstr *UserMI : NewUsers) {
    SlotIndex UseIdx = LIS.getInstructionIndex(*UserMI).getRegSlot();

    // Derive register lanes read by that user.
    LaneBitmask RegMask;
    for (MachineOperand &MO : UserMI->all_uses()) {
      if (MO.getReg() == DefReg) {
        unsigned SubIdx = MO.getSubReg();
        if (SubIdx == 0) {
          RegMask = FullLaneMask;
          break;
        }
        RegMask |= TRI.getSubRegIndexLaneMask(SubIdx);
      }
    }

    if (RegMask != FullLaneMask) {
      // Refine sub-ranges to be able to track the mask for that user.
      LI.refineSubRanges(
          LIS.getVNInfoAllocator(), RegMask, [](LiveInterval::SubRange &SR) {},
          *LIS.getSlotIndexes(), TRI);
    }
    extendInterval(LI, RegMask, UseIdx);
  }

  LLVM_DEBUG({
    if (ExtendReg.DefMI->getOperand(0).isDead())
      dbgs() << "Clearing dead flag for "
             << printRematReg(RegIdx, /*SkipRegions=*/false) << '\n';
  });
  ExtendReg.DefMI->getOperand(0).setIsDead(false);
}

void Rematerializer::extendInterval(LiveInterval &LI, LaneBitmask Mask,
                                    SlotIndex UseIdx) const {
  if (!LI.hasSubRanges()) {
    if (!LI.liveAt(UseIdx))
      LLVM_DEBUG(dbgs() << "Extending interval of register "
                        << printReg(LI.reg(), &TRI, 0, &MRI) << " to " << UseIdx
                        << '\n');
    LIS.extendToIndices(LI, UseIdx);
    return;
  }

  bool SubRangeExtended = false;
  for (LiveInterval::SubRange &SR : LI.subranges()) {
    if ((SR.LaneMask & Mask).any() && !SR.liveAt(UseIdx)) {
      SubRangeExtended = true;
      LLVM_DEBUG(dbgs() << "Extending subrange " << SR << " of register "
                        << printReg(LI.reg(), &TRI, 0, &MRI) << " to " << UseIdx
                        << '\n');
      LIS.extendToIndices(SR, UseIdx);
    }
  }
  if (!SubRangeExtended)
    return;

  // FIXME: this fully reconstructs the main live range from scratch, but
  // there may be a more targeted way to make the update.
  LI.clear();
  LIS.constructMainRangeFromSubranges(LI);
}

void Rematerializer::shrinkToUses(RegisterIdx RegIdx) {
  Reg &ShrinkReg = Regs[RegIdx];
  assert(ShrinkReg.isAlive() && "register must be alive");
  if (ShrinkReg.Uses.empty()) {
    deleteReg(RegIdx);
    return;
  }

  // By construction, registers should never end up with multiple disconnected
  // components or dead definitions.
  LiveInterval &LI = LIS.getInterval(ShrinkReg.getDefReg());
  LLVM_DEBUG(dbgs() << "Shrinking interval of " << printID(RegIdx) << ": " << LI
                    << '\n');
  LIS.shrinkToUses(&LI);
}

void Rematerializer::shrinkToUsesUnremat(Register Reg) {
  LiveInterval &LI = LIS.getInterval(Reg);
  LLVM_DEBUG(dbgs() << "Shrinking interval of unrematerializable register "
                    << LI << '\n');

  SmallVector<MachineInstr *, 2> DeadDefs;
  if (!LIS.shrinkToUses(&LI, &DeadDefs)) {
    assert(DeadDefs.empty() && "expected no dead def");
    return;
  }

  // This should be a very rare occurence, but shrinking an unrematerializable
  // register could create dead defs.
  if (DeadDefs.empty())
    return;

  // The live-range editor delegate will take care of reflecting the
  // elimination of all dead definitions in the rematerializer.
  SmallVector<Register, 4> NewRegs;
  DeadDefDelegate DeadDefDeleg(*this);
  MachineFunction &MF = *DeadDefs.front()->getParent()->getParent();
  LiveRangeEdit(nullptr, NewRegs, MF, LIS, nullptr, &DeadDefDeleg)
      .eliminateDeadDefs(DeadDefs);
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
      for (RegisterIdx DepRegIdx : getReg(RegIdx).Dependencies)
        WalkTree(DepRegIdx, Depth + 1);
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
    if (!PrintReg.isAlive()) {
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
    OS << printID(RegIdx);
    if (!SkipRegions) {
      OS << " [" << PrintReg.DefRegion;
      if (!PrintReg.Uses.empty()) {
        assert(PrintReg.isAlive() && "dead register cannot have uses");
        const LiveInterval &LI = LIS.getInterval(PrintReg.getDefReg());
        // First display all regions in which the register is live-through and
        // not used.
        bool First = true;
        for (const auto &[I, Bounds] : enumerate(Regions)) {
          if (PrintReg.Uses.contains(I))
            continue;
          // The register must be live at the live-ins and live-outs of the
          // region.
          MachineBasicBlock::iterator LiveIn =
              skipDebugInstructionsForward(Bounds.first, Bounds.second);
          if (LiveIn == Bounds.second) {
            // The region has no non-debug instructions, it's hard to assess
            // whether the register is live across it without an index.
            continue;
          }
          // LiveIn is inside the range and a non-debug instruction so we know
          // this will also point to a non-debug instruction within the region.
          MachineBasicBlock::iterator LiveOut = skipDebugInstructionsBackward(
              std::prev(Bounds.second), Bounds.first);
          if (LI.liveAt(LIS.getInstructionIndex(*LiveIn)) &&
              LI.liveAt(LIS.getInstructionIndex(*LiveOut).getDeadSlot())) {
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
    if (PrintReg.isAlive()) {
      PrintReg.DefMI->print(OS, /*IsStandalone=*/true, /*SkipOpers=*/false,
                            /*SkipDebugLoc=*/false, /*AddNewLine=*/false);
      OS << " @ ";
      LIS.getInstructionIndex(*PrintReg.DefMI).print(OS);
    }
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

void Rollbacker::rematerializerNoteRegCreated(const Rematerializer &Remater,
                                              RegisterIdx RegIdx) {
  if (RollingBack)
    return;
  assert(Remater.isRematerializedRegister(RegIdx) && "only remats are created");
  Rematerializations[Remater.getOriginOf(RegIdx)].insert(RegIdx);
}

void Rollbacker::rematerializerNoteRegWillBeDeleted(
    const Rematerializer &Remater, RegisterIdx RegIdx) {
  if (RollingBack)
    return;

  // Find a valid re-creation position after the register's definition.
  MachineInstr *DefMI = Remater.getReg(RegIdx).DefMI;
  MachineBasicBlock *ParentMBB = DefMI->getParent();
  MachineBasicBlock::iterator ValidPos = std::next(DefMI->getIterator());
  while (ValidPos != ParentMBB->end() && isRollbackableMI(*ValidPos, Remater))
    ValidPos = std::next(ValidPos);

  if (Remater.isRematerializedRegister(RegIdx)) {
    // Rematerializations will not be re-created. Previously deleted registers
    // that reference this register's defining instruction as their re-creation
    // position should instead be re-created at a valid position after the
    // deleted MI.
    invalidatePosition(DefMI, ValidPos);
    return;
  }

  // Original registers can be re-created. Add a re-creation position for the
  // definition of the rematerializable register.
  DeadRegs.push_back(DeadReg(RegIdx, Remater));
  const InsertBeforePos InsertPos = makePos(ValidPos, ParentMBB);
  PosToIdx[InsertPos].insert(Positions.size());
  Positions.push_back(InsertPos);
}

void Rollbacker::rematerializerNoteMIWillBeDeleted(
    const Rematerializer &Remater, MachineInstr &MI) {
  if (RollingBack)
    return;

  // Previously deleted registers that reference this MI as their re-creation
  // position should instead be re-created at a valid position after it.
  MachineBasicBlock *ParentMBB = MI.getParent();
  MachineBasicBlock::iterator ValidPos = std::next(MI.getIterator());
  while (ValidPos != ParentMBB->end() && isRollbackableMI(*ValidPos, Remater))
    ValidPos = std::next(ValidPos);
  invalidatePosition(&MI, ValidPos);
}

void Rollbacker::rollback(Rematerializer &Remater) {
  RollingBack = true;

  // As we re-create registers, map deleted definitions to re-created ones. This
  // allows to replace invalid re-creation positions that reference deleted
  // definitions to valid new positions while restoring original MI order.
  DenseMap<MachineInstr *, MachineInstr *> Replacements;
  unsigned PositionIndex = Positions.size();

  // Re-create deleted registers in reverse order of deletion. Related registers
  // are deleted in reverse def-use order so this ensures we re-create registers
  // in def-use order. This also ensures that re-creation positions that became
  // invalid due to later MI deletions can be corrected as we go.
  for (const DeadReg &Reg : reverse(DeadRegs)) {
    if (Remater.isPermanentlyDead(Reg.Idx)) {
      // It is possible the register was permanently deleted as a consequence of
      // dead-def elimination.
      Rematerializations.erase(Reg.Idx);
      --PositionIndex;
      continue;
    }

    assert(!Remater.getReg(Reg.Idx).isAlive() && "register should be dead");

    // Determine re-creation position for the register's definition.
    MachineBasicBlock::iterator InsertPosition;
    InsertBeforePos Pos = Positions[--PositionIndex];
    if (auto *MBB = dyn_cast<MachineBasicBlock *>(Pos)) {
      InsertPosition = MBB->end();
    } else {
      auto *MI = cast<MachineInstr *>(Pos);
      InsertPosition = Replacements.lookup_or(MI, MI)->getIterator();
    }

    Remater.recreateReg(Reg.Idx, InsertPosition, Reg.DefReg);

    const Rematerializer::Reg &RecreateReg = Remater.getReg(Reg.Idx);
    if (!Replacements.insert({Reg.DefMI, RecreateReg.DefMI}).second)
      llvm_unreachable("duplicate deleted MI");
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

  DeadRegs.clear();
  Positions.clear();
  PosToIdx.clear();
  Rematerializations.clear();
  RollingBack = false;
}

bool Rollbacker::isRollbackableMI(const MachineInstr &MI,
                                  const Rematerializer &Remater) const {
  RegisterIdx RegIdx = Remater.getDefRegIdx(MI);
  if (RegIdx == Rematerializer::NoReg ||
      !Remater.isRematerializedRegister(RegIdx))
    return false;
  // It is possible that the MI defines a rematerializable register that was not
  // recorded if the rollbacker was attached to the rematerializer after the
  // rematerialization happened. In such cases the MI won't be rolled back.
  auto RematsOf = Rematerializations.find(Remater.getOriginOf(RegIdx));
  if (RematsOf == Rematerializations.end())
    return false;
  return RematsOf->getSecond().contains(RegIdx);
}

void Rollbacker::invalidatePosition(MachineInstr *MI,
                                    MachineBasicBlock::iterator It) {
  const InsertBeforePos MIPos = InsertBeforePos(MI),
                        NewPos = makePos(It, MI->getParent());
  auto MIIndices = PosToIdx.find(MIPos);
  if (MIIndices == PosToIdx.end())
    return;
  const SmallDenseSet<unsigned, 1> &InvalIndices = MIIndices->getSecond();
  assert(!InvalIndices.empty() && "no index hold position");
  for (unsigned I : InvalIndices)
    Positions[I] = NewPos;
  PosToIdx.try_emplace(NewPos).first->getSecond().insert_range(InvalIndices);
  PosToIdx.erase(MIPos);
}
