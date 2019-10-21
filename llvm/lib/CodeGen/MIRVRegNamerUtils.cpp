//===---------- MIRVRegNamerUtils.cpp - MIR VReg Renaming Utilities -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MIRVRegNamerUtils.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "mir-vregnamer-utils"

namespace {

// TypedVReg and VRType are used to tell the renamer what to do at points in a
// sequence of values to be renamed. A TypedVReg can either contain
// an actual VReg, a FrameIndex, or it could just be a barrier for the next
// candidate (side-effecting instruction). This tells the renamer to increment
// to the next vreg name, or to skip modulo some skip-gap value.
enum VRType { RSE_Reg = 0, RSE_FrameIndex, RSE_NewCandidate };
class TypedVReg {
  VRType Type;
  Register Reg;

public:
  TypedVReg(Register Reg) : Type(RSE_Reg), Reg(Reg) {}
  TypedVReg(VRType Type) : Type(Type), Reg(~0U) {
    assert(Type != RSE_Reg && "Expected a non-Register Type.");
  }

  bool isReg() const { return Type == RSE_Reg; }
  bool isFrameIndex() const { return Type == RSE_FrameIndex; }
  bool isCandidate() const { return Type == RSE_NewCandidate; }

  VRType getType() const { return Type; }
  Register getReg() const {
    assert(this->isReg() && "Expected a virtual or physical Register.");
    return Reg;
  }
};

/// Here we find our candidates. What makes an interesting candidate?
/// A candidate for a canonicalization tree root is normally any kind of
/// instruction that causes side effects such as a store to memory or a copy to
/// a physical register or a return instruction. We use these as an expression
/// tree root that we walk in order to build a canonical walk which should
/// result in canonical vreg renaming.
std::vector<MachineInstr *> populateCandidates(MachineBasicBlock *MBB) {
  std::vector<MachineInstr *> Candidates;
  MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();

  for (auto II = MBB->begin(), IE = MBB->end(); II != IE; ++II) {
    MachineInstr *MI = &*II;

    bool DoesMISideEffect = false;

    if (MI->getNumOperands() > 0 && MI->getOperand(0).isReg()) {
      const Register Dst = MI->getOperand(0).getReg();
      DoesMISideEffect |= !Register::isVirtualRegister(Dst);

      for (auto UI = MRI.use_begin(Dst); UI != MRI.use_end(); ++UI) {
        if (DoesMISideEffect)
          break;
        DoesMISideEffect |= (UI->getParent()->getParent() != MI->getParent());
      }
    }

    if (!MI->mayStore() && !MI->isBranch() && !DoesMISideEffect)
      continue;

    LLVM_DEBUG(dbgs() << "Found Candidate:  "; MI->dump(););
    Candidates.push_back(MI);
  }

  return Candidates;
}

void doCandidateWalk(std::vector<TypedVReg> &VRegs,
                     std::queue<TypedVReg> &RegQueue,
                     std::vector<MachineInstr *> &VisitedMIs,
                     const MachineBasicBlock *MBB) {

  const MachineFunction &MF = *MBB->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  while (!RegQueue.empty()) {

    auto TReg = RegQueue.front();
    RegQueue.pop();

    if (TReg.isFrameIndex()) {
      LLVM_DEBUG(dbgs() << "Popping frame index.\n";);
      VRegs.push_back(TypedVReg(RSE_FrameIndex));
      continue;
    }

    assert(TReg.isReg() && "Expected vreg or physreg.");
    Register Reg = TReg.getReg();

    if (Register::isVirtualRegister(Reg)) {
      LLVM_DEBUG({
        dbgs() << "Popping vreg ";
        MRI.def_begin(Reg)->dump();
        dbgs() << "\n";
      });

      if (!llvm::any_of(VRegs, [&](const TypedVReg &TR) {
            return TR.isReg() && TR.getReg() == Reg;
          })) {
        VRegs.push_back(TypedVReg(Reg));
      }
    } else {
      LLVM_DEBUG(dbgs() << "Popping physreg.\n";);
      VRegs.push_back(TypedVReg(Reg));
      continue;
    }

    for (auto RI = MRI.def_begin(Reg), RE = MRI.def_end(); RI != RE; ++RI) {
      MachineInstr *Def = RI->getParent();

      if (Def->getParent() != MBB)
        continue;

      if (llvm::any_of(VisitedMIs,
                       [&](const MachineInstr *VMI) { return Def == VMI; })) {
        break;
      }

      LLVM_DEBUG({
        dbgs() << "\n========================\n";
        dbgs() << "Visited MI: ";
        Def->dump();
        dbgs() << "BB Name: " << Def->getParent()->getName() << "\n";
        dbgs() << "\n========================\n";
      });
      VisitedMIs.push_back(Def);
      for (unsigned I = 1, E = Def->getNumOperands(); I != E; ++I) {

        MachineOperand &MO = Def->getOperand(I);
        if (MO.isFI()) {
          LLVM_DEBUG(dbgs() << "Pushing frame index.\n";);
          RegQueue.push(TypedVReg(RSE_FrameIndex));
        }

        if (!MO.isReg())
          continue;
        RegQueue.push(TypedVReg(MO.getReg()));
      }
    }
  }
}

std::map<unsigned, unsigned>
getVRegRenameMap(const std::vector<TypedVReg> &VRegs,
                 const std::vector<Register> &renamedInOtherBB,
                 MachineRegisterInfo &MRI, NamedVRegCursor &NVC) {
  std::map<unsigned, unsigned> VRegRenameMap;
  bool FirstCandidate = true;

  for (auto &vreg : VRegs) {
    if (vreg.isFrameIndex()) {
      // We skip one vreg for any frame index because there is a good chance
      // (especially when comparing SelectionDAG to GlobalISel generated MIR)
      // that in the other file we are just getting an incoming vreg that comes
      // from a copy from a frame index. So it's safe to skip by one.
      unsigned LastRenameReg = NVC.incrementVirtualVReg();
      (void)LastRenameReg;
      LLVM_DEBUG(dbgs() << "Skipping rename for FI " << LastRenameReg << "\n";);
      continue;
    } else if (vreg.isCandidate()) {

      // After the first candidate, for every subsequent candidate, we skip mod
      // 10 registers so that the candidates are more likely to start at the
      // same vreg number making it more likely that the canonical walk from the
      // candidate insruction. We don't need to skip from the first candidate of
      // the BasicBlock because we already skip ahead several vregs for each BB.
      unsigned LastRenameReg = NVC.getVirtualVReg();
      if (FirstCandidate)
        NVC.incrementVirtualVReg(LastRenameReg % 10);
      FirstCandidate = false;
      continue;
    } else if (!Register::isVirtualRegister(vreg.getReg())) {
      unsigned LastRenameReg = NVC.incrementVirtualVReg();
      (void)LastRenameReg;
      LLVM_DEBUG({
        dbgs() << "Skipping rename for Phys Reg " << LastRenameReg << "\n";
      });
      continue;
    }

    auto Reg = vreg.getReg();
    if (llvm::find(renamedInOtherBB, Reg) != renamedInOtherBB.end()) {
      LLVM_DEBUG(dbgs() << "Vreg " << Reg
                        << " already renamed in other BB.\n";);
      continue;
    }

    auto Rename = NVC.createVirtualRegister(Reg);

    if (VRegRenameMap.find(Reg) == VRegRenameMap.end()) {
      LLVM_DEBUG(dbgs() << "Mapping vreg ";);
      if (MRI.reg_begin(Reg) != MRI.reg_end()) {
        LLVM_DEBUG(auto foo = &*MRI.reg_begin(Reg); foo->dump(););
      } else {
        LLVM_DEBUG(dbgs() << Reg;);
      }
      LLVM_DEBUG(dbgs() << " to ";);
      if (MRI.reg_begin(Rename) != MRI.reg_end()) {
        LLVM_DEBUG(auto foo = &*MRI.reg_begin(Rename); foo->dump(););
      } else {
        LLVM_DEBUG(dbgs() << Rename;);
      }
      LLVM_DEBUG(dbgs() << "\n";);

      VRegRenameMap.insert(std::pair<unsigned, unsigned>(Reg, Rename));
    }
  }

  return VRegRenameMap;
}

bool doVRegRenaming(std::vector<Register> &renamedInOtherBB,
                    const std::map<unsigned, unsigned> &VRegRenameMap,
                    MachineRegisterInfo &MRI) {
  bool Changed = false;
  for (auto I = VRegRenameMap.begin(), E = VRegRenameMap.end(); I != E; ++I) {

    auto VReg = I->first;
    auto Rename = I->second;

    renamedInOtherBB.push_back(Rename);

    std::vector<MachineOperand *> RenameMOs;
    for (auto &MO : MRI.reg_operands(VReg)) {
      RenameMOs.push_back(&MO);
    }

    for (auto *MO : RenameMOs) {
      Changed = true;
      MO->setReg(Rename);

      if (!MO->isDef())
        MO->setIsKill(false);
    }
  }

  return Changed;
}

bool renameVRegs(MachineBasicBlock *MBB,
                 std::vector<Register> &renamedInOtherBB,
                 NamedVRegCursor &NVC) {
  bool Changed = false;
  MachineFunction &MF = *MBB->getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  std::vector<MachineInstr *> Candidates = populateCandidates(MBB);
  std::vector<MachineInstr *> VisitedMIs;
  llvm::copy(Candidates, std::back_inserter(VisitedMIs));

  std::vector<TypedVReg> VRegs;
  for (auto candidate : Candidates) {
    VRegs.push_back(TypedVReg(RSE_NewCandidate));

    std::queue<TypedVReg> RegQueue;

    // Here we walk the vreg operands of a non-root node along our walk.
    // The root nodes are the original candidates (stores normally).
    // These are normally not the root nodes (except for the case of copies to
    // physical registers).
    for (unsigned i = 1; i < candidate->getNumOperands(); i++) {
      if (candidate->mayStore() || candidate->isBranch())
        break;

      MachineOperand &MO = candidate->getOperand(i);
      if (!(MO.isReg() && Register::isVirtualRegister(MO.getReg())))
        continue;

      LLVM_DEBUG(dbgs() << "Enqueue register"; MO.dump(); dbgs() << "\n";);
      RegQueue.push(TypedVReg(MO.getReg()));
    }

    // Here we walk the root candidates. We start from the 0th operand because
    // the root is normally a store to a vreg.
    for (unsigned i = 0; i < candidate->getNumOperands(); i++) {

      if (!candidate->mayStore() && !candidate->isBranch())
        break;

      MachineOperand &MO = candidate->getOperand(i);

      // TODO: Do we want to only add vregs here?
      if (!MO.isReg() && !MO.isFI())
        continue;

      LLVM_DEBUG(dbgs() << "Enqueue Reg/FI"; MO.dump(); dbgs() << "\n";);

      RegQueue.push(MO.isReg() ? TypedVReg(MO.getReg())
                               : TypedVReg(RSE_FrameIndex));
    }

    doCandidateWalk(VRegs, RegQueue, VisitedMIs, MBB);
  }

  // If we have populated no vregs to rename then bail.
  // The rest of this function does the vreg remaping.
  if (VRegs.size() == 0)
    return Changed;

  auto VRegRenameMap = getVRegRenameMap(VRegs, renamedInOtherBB, MRI, NVC);
  Changed |= doVRegRenaming(renamedInOtherBB, VRegRenameMap, MRI);
  return Changed;
}
} // anonymous namespace

void NamedVRegCursor::skipVRegs() {
  unsigned VRegGapIndex = 1;
  if (!virtualVRegNumber) {
    VRegGapIndex = 0;
    virtualVRegNumber = MRI.createIncompleteVirtualRegister();
  }
  const unsigned VR_GAP = (++VRegGapIndex * SkipGapSize);

  unsigned I = virtualVRegNumber;
  const unsigned E = (((I + VR_GAP) / VR_GAP) + 1) * VR_GAP;

  virtualVRegNumber = E;
}

unsigned NamedVRegCursor::createVirtualRegister(unsigned VReg) {
  if (!virtualVRegNumber)
    skipVRegs();
  std::string S;
  raw_string_ostream OS(S);
  OS << "namedVReg" << (virtualVRegNumber & ~0x80000000);
  OS.flush();
  virtualVRegNumber++;
  if (auto RC = MRI.getRegClassOrNull(VReg))
    return MRI.createVirtualRegister(RC, OS.str());
  return MRI.createGenericVirtualRegister(MRI.getType(VReg), OS.str());
}

bool NamedVRegCursor::renameVRegs(MachineBasicBlock *MBB) {
  return ::renameVRegs(MBB, RenamedInOtherBB, *this);
}
