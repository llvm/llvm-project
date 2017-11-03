//===-------------- MIRCanonicalizer.cpp - MIR Canonicalizer --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The purpose of this pass is to employ a canonical code transformation so
// that code compiled with slightly different IR passes can be diffed more
// effectively than otherwise. This is done by renaming vregs in a given
// LiveRange in a canonical way. This pass also does a pseudo-scheduling to
// move defs closer to their use inorder to reduce diffs caused by slightly
// different schedules.
//
// Basic Usage:
//
// llc -o - -run-pass mir-canonicalizer example.mir
//
// Reorders instructions canonically.
// Renames virtual register operands canonically.
// Strips certain MIR artifacts (optionally).
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"

#include <queue>

using namespace llvm;

namespace llvm {
extern char &MIRCanonicalizerID;
} // namespace llvm

#define DEBUG_TYPE "mir-canonicalizer"

static cl::opt<unsigned>
CanonicalizeFunctionNumber("canon-nth-function", cl::Hidden, cl::init(~0u),
                           cl::value_desc("N"),
                           cl::desc("Function number to canonicalize."));

static cl::opt<unsigned>
CanonicalizeBasicBlockNumber("canon-nth-basicblock", cl::Hidden, cl::init(~0u),
                             cl::value_desc("N"),
                             cl::desc("BasicBlock number to canonicalize."));

namespace {

class MIRCanonicalizer : public MachineFunctionPass {
public:
  static char ID;
  MIRCanonicalizer() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "Rename register operands in a canonical ordering.";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end anonymous namespace

enum VRType { RSE_Reg = 0, RSE_FrameIndex, RSE_NewCandidate };
class TypedVReg {
  VRType type;
  unsigned reg;

public:
  TypedVReg(unsigned reg) : type(RSE_Reg), reg(reg) {}
  TypedVReg(VRType type) : type(type), reg(~0U) {
    assert(type != RSE_Reg && "Expected a non-register type.");
  }

  bool isReg()        const { return type == RSE_Reg;          }
  bool isFrameIndex() const { return type == RSE_FrameIndex;   }
  bool isCandidate()  const { return type == RSE_NewCandidate; }

  VRType getType() const { return type; }
  unsigned getReg() const {
    assert(this->isReg() && "Expected a virtual or physical register.");
    return reg;
  }
};

char MIRCanonicalizer::ID;

char &llvm::MIRCanonicalizerID = MIRCanonicalizer::ID;

INITIALIZE_PASS_BEGIN(MIRCanonicalizer, "mir-canonicalizer",
                      "Rename Register Operands Canonically", false, false);

INITIALIZE_PASS_END(MIRCanonicalizer, "mir-canonicalizer",
                    "Rename Register Operands Canonically", false, false);

static std::vector<MachineBasicBlock *> GetRPOList(MachineFunction &MF) {
  ReversePostOrderTraversal<MachineBasicBlock *> RPOT(&*MF.begin());
  std::vector<MachineBasicBlock *> RPOList;
  for (auto MBB : RPOT) {
    RPOList.push_back(MBB);
  }

  return RPOList;
}

// Set a dummy vreg. We use this vregs register class to generate throw-away
// vregs that are used to skip vreg numbers so that vreg numbers line up.
static unsigned GetDummyVReg(const MachineFunction &MF) {
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      for (auto &MO : MI.operands()) {
        if (!MO.isReg() || !TargetRegisterInfo::isVirtualRegister(MO.getReg()))
          continue;
        return MO.getReg();
      }
    }
  }

  return ~0U;
}

static bool rescheduleCanonically(MachineBasicBlock *MBB) {

  bool Changed = false;

  // Calculates the distance of MI from the begining of its parent BB.
  auto getInstrIdx = [](const MachineInstr &MI) {
    unsigned i = 0;
    for (auto &CurMI : *MI.getParent()) {
      if (&CurMI == &MI)
        return i;
      i++;
    }
    return ~0U;
  };

  // Pre-Populate vector of instructions to reschedule so that we don't
  // clobber the iterator.
  std::vector<MachineInstr *> Instructions;
  for (auto &MI : *MBB) {
    Instructions.push_back(&MI);
  }

  for (auto *II : Instructions) {
    if (II->getNumOperands() == 0)
      continue;

    MachineOperand &MO = II->getOperand(0);
    if (!MO.isReg() || !TargetRegisterInfo::isVirtualRegister(MO.getReg()))
      continue;

    DEBUG(dbgs() << "Operand " << 0 << " of "; II->dump(); MO.dump(););

    MachineInstr *Def = II;
    unsigned Distance = ~0U;
    MachineInstr *UseToBringDefCloserTo = nullptr;
    MachineRegisterInfo *MRI = &MBB->getParent()->getRegInfo();
    for (auto &UO : MRI->use_nodbg_operands(MO.getReg())) {
      MachineInstr *UseInst = UO.getParent();

      const unsigned DefLoc = getInstrIdx(*Def);
      const unsigned UseLoc = getInstrIdx(*UseInst);
      const unsigned Delta = (UseLoc - DefLoc);

      if (UseInst->getParent() != Def->getParent())
        continue;
      if (DefLoc >= UseLoc)
        continue;

      if (Delta < Distance) {
        Distance = Delta;
        UseToBringDefCloserTo = UseInst;
      }
    }

    const auto BBE = MBB->instr_end();
    MachineBasicBlock::iterator DefI = BBE;
    MachineBasicBlock::iterator UseI = BBE;

    for (auto BBI = MBB->instr_begin(); BBI != BBE; ++BBI) {

      if (DefI != BBE && UseI != BBE)
        break;

      if ((&*BBI != Def) && (&*BBI != UseToBringDefCloserTo))
        continue;

      if (&*BBI == Def) {
        DefI = BBI;
        continue;
      }

      if (&*BBI == UseToBringDefCloserTo) {
        UseI = BBI;
        continue;
      }
    }

    if (DefI == BBE || UseI == BBE)
      continue;

    DEBUG({
      dbgs() << "Splicing ";
      DefI->dump();
      dbgs() << " right before: ";
      UseI->dump();
    });

    Changed = true;
    MBB->splice(UseI, MBB, DefI);
  }

  return Changed;
}

/// Here we find our candidates. What makes an interesting candidate?
/// An candidate for a canonicalization tree root is normally any kind of
/// instruction that causes side effects such as a store to memory or a copy to
/// a physical register or a return instruction. We use these as an expression
/// tree root that we walk inorder to build a canonical walk which should result
/// in canoncal vreg renaming.
static std::vector<MachineInstr *> populateCandidates(MachineBasicBlock *MBB) {
  std::vector<MachineInstr *> Candidates;
  MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();

  for (auto II = MBB->begin(), IE = MBB->end(); II != IE; ++II) {
    MachineInstr *MI = &*II;

    bool DoesMISideEffect = false;

    if (MI->getNumOperands() > 0 && MI->getOperand(0).isReg()) {
      const unsigned Dst = MI->getOperand(0).getReg();
      DoesMISideEffect |= !TargetRegisterInfo::isVirtualRegister(Dst);

      for (auto UI = MRI.use_begin(Dst); UI != MRI.use_end(); ++UI) {
        if (DoesMISideEffect) break;
        DoesMISideEffect |= (UI->getParent()->getParent() != MI->getParent());
      }
    }

    if (!MI->mayStore() && !MI->isBranch() && !DoesMISideEffect)
      continue;

    DEBUG(dbgs() << "Found Candidate:  "; MI->dump(););
    Candidates.push_back(MI);
  }

  return Candidates;
}

void doCandidateWalk(std::vector<TypedVReg> &VRegs,
                     std::queue <TypedVReg> &RegQueue,
                     std::vector<MachineInstr *> &VisitedMIs,
                     const MachineBasicBlock *MBB) {

  const MachineFunction &MF = *MBB->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  while (!RegQueue.empty()) {

    auto TReg = RegQueue.front();
    RegQueue.pop();

    if (TReg.isFrameIndex()) {
      DEBUG(dbgs() << "Popping frame index.\n";);
      VRegs.push_back(TypedVReg(RSE_FrameIndex));
      continue;
    }

    assert(TReg.isReg() && "Expected vreg or physreg.");
    unsigned Reg = TReg.getReg();

    if (TargetRegisterInfo::isVirtualRegister(Reg)) {
      DEBUG({
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
      DEBUG(dbgs() << "Popping physreg.\n";);
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

      DEBUG({
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
          DEBUG(dbgs() << "Pushing frame index.\n";);
          RegQueue.push(TypedVReg(RSE_FrameIndex));
        }

        if (!MO.isReg())
          continue;
        RegQueue.push(TypedVReg(MO.getReg()));
      }
    }
  }
}

// TODO: Work to remove this in the future. One day when we have named vregs
// we should be able to form the canonical name based on some characteristic
// we see in that point of the expression tree (like if we were to name based
// on some sort of value numbering scheme).
static void SkipVRegs(unsigned &VRegGapIndex, MachineRegisterInfo &MRI,
                      const TargetRegisterClass *RC) {
  const unsigned VR_GAP = (++VRegGapIndex * 1000);

  DEBUG({
    dbgs() << "Adjusting per-BB VR_GAP for BB" << VRegGapIndex << " to "
           << VR_GAP << "\n";
  });

  unsigned I = MRI.createVirtualRegister(RC);
  const unsigned E = (((I + VR_GAP) / VR_GAP) + 1) * VR_GAP;
  while (I != E) {
    I = MRI.createVirtualRegister(RC);
  }
}

static std::map<unsigned, unsigned>
GetVRegRenameMap(const std::vector<TypedVReg> &VRegs,
                 const std::vector<unsigned> &renamedInOtherBB,
                 MachineRegisterInfo &MRI,
                 const TargetRegisterClass *RC) {
  std::map<unsigned, unsigned> VRegRenameMap;
  unsigned LastRenameReg = MRI.createVirtualRegister(RC);
  bool FirstCandidate = true;

  for (auto &vreg : VRegs) {
    if (vreg.isFrameIndex()) {
      // We skip one vreg for any frame index because there is a good chance
      // (especially when comparing SelectionDAG to GlobalISel generated MIR)
      // that in the other file we are just getting an incoming vreg that comes
      // from a copy from a frame index. So it's safe to skip by one.
      LastRenameReg = MRI.createVirtualRegister(RC);
      DEBUG(dbgs() << "Skipping rename for FI " << LastRenameReg << "\n";);
      continue;
    } else if (vreg.isCandidate()) {

      // After the first candidate, for every subsequent candidate, we skip mod
      // 10 registers so that the candidates are more likely to start at the
      // same vreg number making it more likely that the canonical walk from the
      // candidate insruction. We don't need to skip from the first candidate of
      // the BasicBlock because we already skip ahead several vregs for each BB.
      while (LastRenameReg % 10) {
        if (!FirstCandidate) break;
        LastRenameReg = MRI.createVirtualRegister(RC);

        DEBUG({
          dbgs() << "Skipping rename for new candidate " << LastRenameReg
                 << "\n";
        });
      }
      FirstCandidate = false;
      continue;
    } else if (!TargetRegisterInfo::isVirtualRegister(vreg.getReg())) {
      LastRenameReg = MRI.createVirtualRegister(RC);
      DEBUG({
        dbgs() << "Skipping rename for Phys Reg " << LastRenameReg << "\n";
      });
      continue;
    }

    auto Reg = vreg.getReg();
    if (llvm::find(renamedInOtherBB, Reg) != renamedInOtherBB.end()) {
      DEBUG(dbgs() << "Vreg " << Reg << " already renamed in other BB.\n";);
      continue;
    }

    auto Rename = MRI.createVirtualRegister(MRI.getRegClass(Reg));
    LastRenameReg = Rename;

    if (VRegRenameMap.find(Reg) == VRegRenameMap.end()) {
      DEBUG(dbgs() << "Mapping vreg ";);
      if (MRI.reg_begin(Reg) != MRI.reg_end()) {
        DEBUG(auto foo = &*MRI.reg_begin(Reg); foo->dump(););
      } else {
        DEBUG(dbgs() << Reg;);
      }
      DEBUG(dbgs() << " to ";);
      if (MRI.reg_begin(Rename) != MRI.reg_end()) {
        DEBUG(auto foo = &*MRI.reg_begin(Rename); foo->dump(););
      } else {
        DEBUG(dbgs() << Rename;);
      }
      DEBUG(dbgs() << "\n";);

      VRegRenameMap.insert(std::pair<unsigned, unsigned>(Reg, Rename));
    }
  }

  return VRegRenameMap;
}

static bool doVRegRenaming(std::vector<unsigned> &RenamedInOtherBB,
                           const std::map<unsigned, unsigned> &VRegRenameMap,
                           MachineRegisterInfo &MRI) {
  bool Changed = false;
  for (auto I = VRegRenameMap.begin(), E = VRegRenameMap.end(); I != E; ++I) {

    auto VReg = I->first;
    auto Rename = I->second;

    RenamedInOtherBB.push_back(Rename);

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

static bool doDefKillClear(MachineBasicBlock *MBB) {
  bool Changed = false;

  for (auto &MI : *MBB) {
    for (auto &MO : MI.operands()) {
      if (!MO.isReg())
        continue;
      if (!MO.isDef() && MO.isKill()) {
        Changed = true;
        MO.setIsKill(false);
      }

      if (MO.isDef() && MO.isDead()) {
        Changed = true;
        MO.setIsDead(false);
      }
    }
  }

  return Changed;
}

static bool runOnBasicBlock(MachineBasicBlock *MBB,
                            std::vector<StringRef> &bbNames,
                            std::vector<unsigned> &renamedInOtherBB,
                            unsigned &basicBlockNum, unsigned &VRegGapIndex) {

  if (CanonicalizeBasicBlockNumber != ~0U) {
    if (CanonicalizeBasicBlockNumber != basicBlockNum++)
      return false;
    DEBUG(dbgs() << "\n Canonicalizing BasicBlock " << MBB->getName() << "\n";);
  }

  if (llvm::find(bbNames, MBB->getName()) != bbNames.end()) {
    DEBUG({
      dbgs() << "Found potentially duplicate BasicBlocks: " << MBB->getName()
             << "\n";
    });
    return false;
  }

  DEBUG({
    dbgs() << "\n\n  NEW BASIC BLOCK: " << MBB->getName() << "  \n\n";
    dbgs() << "\n\n================================================\n\n";
  });

  bool Changed = false;
  MachineFunction &MF = *MBB->getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  const unsigned DummyVReg = GetDummyVReg(MF);
  const TargetRegisterClass *DummyRC =
    (DummyVReg == ~0U) ? nullptr : MRI.getRegClass(DummyVReg);
  if (!DummyRC) return false;

  bbNames.push_back(MBB->getName());
  DEBUG(dbgs() << "\n\n NEW BASIC BLOCK: " << MBB->getName() << "\n\n";);

  DEBUG(dbgs() << "MBB Before Scheduling:\n"; MBB->dump(););
  Changed |= rescheduleCanonically(MBB);
  DEBUG(dbgs() << "MBB After Scheduling:\n"; MBB->dump(););

  std::vector<MachineInstr *> Candidates = populateCandidates(MBB);
  std::vector<MachineInstr *> VisitedMIs;
  std::copy(Candidates.begin(), Candidates.end(),
            std::back_inserter(VisitedMIs));

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
      if (!(MO.isReg() && TargetRegisterInfo::isVirtualRegister(MO.getReg())))
        continue;

      DEBUG(dbgs() << "Enqueue register"; MO.dump(); dbgs() << "\n";);
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

      DEBUG(dbgs() << "Enqueue Reg/FI"; MO.dump(); dbgs() << "\n";);

      RegQueue.push(MO.isReg() ? TypedVReg(MO.getReg()) :
                                  TypedVReg(RSE_FrameIndex));
    }

    doCandidateWalk(VRegs, RegQueue, VisitedMIs, MBB);
  }

  // If we have populated no vregs to rename then bail.
  // The rest of this function does the vreg remaping.
  if (VRegs.size() == 0)
    return Changed;

  // Skip some vregs, so we can recon where we'll land next.
  SkipVRegs(VRegGapIndex, MRI, DummyRC);

  auto VRegRenameMap = GetVRegRenameMap(VRegs, renamedInOtherBB, MRI, DummyRC);
  Changed |= doVRegRenaming(renamedInOtherBB, VRegRenameMap, MRI);
  Changed |= doDefKillClear(MBB);

  DEBUG(dbgs() << "Updated MachineBasicBlock:\n"; MBB->dump(); dbgs() << "\n";);
  DEBUG(dbgs() << "\n\n================================================\n\n");
  return Changed;
}

bool MIRCanonicalizer::runOnMachineFunction(MachineFunction &MF) {

  static unsigned functionNum = 0;
  if (CanonicalizeFunctionNumber != ~0U) {
    if (CanonicalizeFunctionNumber != functionNum++)
      return false;
    DEBUG(dbgs() << "\n Canonicalizing Function " << MF.getName() << "\n";);
  }

  // we need a valid vreg to create a vreg type for skipping all those
  // stray vreg numbers so reach alignment/canonical vreg values.
  std::vector<MachineBasicBlock*> RPOList = GetRPOList(MF);

  DEBUG(
    dbgs() << "\n\n  NEW MACHINE FUNCTION: " << MF.getName() << "  \n\n";
    dbgs() << "\n\n================================================\n\n";
    dbgs() << "Total Basic Blocks: " << RPOList.size() << "\n";
    for (auto MBB : RPOList) {
      dbgs() << MBB->getName() << "\n";
    }
    dbgs() << "\n\n================================================\n\n";
  );

  std::vector<StringRef> BBNames;
  std::vector<unsigned> RenamedInOtherBB;

  unsigned GapIdx = 0;
  unsigned BBNum = 0;

  bool Changed = false;

  for (auto MBB : RPOList)
    Changed |= runOnBasicBlock(MBB, BBNames, RenamedInOtherBB, BBNum, GapIdx);

  return Changed;
}

