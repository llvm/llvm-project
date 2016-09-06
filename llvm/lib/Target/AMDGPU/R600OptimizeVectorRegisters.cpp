//===--------------------- R600MergeVectorRegisters.cpp -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass merges inputs of swizzeable instructions into vector sharing
/// common data and/or have enough undef subreg using swizzle abilities.
///
/// For instance let's consider the following pseudo code :
/// vreg5<def> = REG_SEQ vreg1, sub0, vreg2, sub1, vreg3, sub2, undef, sub3
/// ...
/// vreg7<def> = REG_SEQ vreg1, sub0, vreg3, sub1, undef, sub2, vreg4, sub3
/// (swizzable Inst) vreg7, SwizzleMask : sub0, sub1, sub2, sub3
///
/// is turned into :
/// vreg5<def> = REG_SEQ vreg1, sub0, vreg2, sub1, vreg3, sub2, undef, sub3
/// ...
/// vreg7<def> = INSERT_SUBREG vreg4, sub3
/// (swizzable Inst) vreg7, SwizzleMask : sub0, sub2, sub1, sub3
///
/// This allow regalloc to reduce register pressure for vector registers and
/// to reduce MOV count.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "R600Defines.h"
#include "R600InstrInfo.h"
#include "llvm/CodeGen/DFAPacketizer.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "vec-merger"

namespace {

static bool
isImplicitlyDef(MachineRegisterInfo &MRI, unsigned Reg) {
  for (MachineRegisterInfo::def_instr_iterator It = MRI.def_instr_begin(Reg),
      E = MRI.def_instr_end(); It != E; ++It) {
    return (*It).isImplicitDef();
  }
  if (MRI.isReserved(Reg)) {
    return false;
  }
  llvm_unreachable("Reg without a def");
  return false;
}

class RegSeqInfo {
public:
  MachineInstr *Instr;
  DenseMap<unsigned, unsigned> RegToChan;
  std::vector<unsigned> UndefReg;
  RegSeqInfo(MachineRegisterInfo &MRI, MachineInstr *MI) : Instr(MI) {
    assert(MI->getOpcode() == AMDGPU::REG_SEQUENCE);
    for (unsigned i = 1, e = Instr->getNumOperands(); i < e; i+=2) {
      MachineOperand &MO = Instr->getOperand(i);
      unsigned Chan = Instr->getOperand(i + 1).getImm();
      if (isImplicitlyDef(MRI, MO.getReg()))
        UndefReg.push_back(Chan);
      else
        RegToChan[MO.getReg()] = Chan;
    }
  }
  RegSeqInfo() {}

  bool operator==(const RegSeqInfo &RSI) const {
    return RSI.Instr == Instr;
  }
};

class R600VectorRegMerger : public MachineFunctionPass {
private:
  MachineRegisterInfo *MRI;
  const R600InstrInfo *TII;
  bool canSwizzle(const MachineInstr &) const;
  bool areAllUsesSwizzeable(unsigned Reg) const;
  void SwizzleInput(MachineInstr &,
      const std::vector<std::pair<unsigned, unsigned> > &) const;
  bool tryMergeVector(const RegSeqInfo *, RegSeqInfo *,
      std::vector<std::pair<unsigned, unsigned> > &Remap) const;
  bool tryMergeUsingCommonSlot(RegSeqInfo &RSI, RegSeqInfo &CompatibleRSI,
      std::vector<std::pair<unsigned, unsigned> > &RemapChan);
  bool tryMergeUsingFreeSlot(RegSeqInfo &RSI, RegSeqInfo &CompatibleRSI,
      std::vector<std::pair<unsigned, unsigned> > &RemapChan);
  MachineInstr *RebuildVector(RegSeqInfo *MI,
      const RegSeqInfo *BaseVec,
      const std::vector<std::pair<unsigned, unsigned> > &RemapChan) const;
  void RemoveMI(MachineInstr *);
  void trackRSI(const RegSeqInfo &RSI);

  typedef DenseMap<unsigned, std::vector<MachineInstr *> > InstructionSetMap;
  DenseMap<MachineInstr *, RegSeqInfo> PreviousRegSeq;
  InstructionSetMap PreviousRegSeqByReg;
  InstructionSetMap PreviousRegSeqByUndefCount;
public:
  static char ID;
  R600VectorRegMerger(TargetMachine &tm) : MachineFunctionPass(ID),
  TII(nullptr) { }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
    AU.addRequired<MachineLoopInfo>();
    AU.addPreserved<MachineLoopInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  const char *getPassName() const override {
    return "R600 Vector Registers Merge Pass";
  }

  bool runOnMachineFunction(MachineFunction &Fn) override;
};

char R600VectorRegMerger::ID = 0;

bool R600VectorRegMerger::canSwizzle(const MachineInstr &MI)
    const {
  if (TII->get(MI.getOpcode()).TSFlags & R600_InstFlag::TEX_INST)
    return true;
  switch (MI.getOpcode()) {
  case AMDGPU::R600_ExportSwz:
  case AMDGPU::EG_ExportSwz:
    return true;
  default:
    return false;
  }
}

bool R600VectorRegMerger::tryMergeVector(const RegSeqInfo *Untouched,
    RegSeqInfo *ToMerge, std::vector< std::pair<unsigned, unsigned> > &Remap)
    const {
  unsigned CurrentUndexIdx = 0;
  for (DenseMap<unsigned, unsigned>::iterator It = ToMerge->RegToChan.begin(),
      E = ToMerge->RegToChan.end(); It != E; ++It) {
    DenseMap<unsigned, unsigned>::const_iterator PosInUntouched =
        Untouched->RegToChan.find((*It).first);
    if (PosInUntouched != Untouched->RegToChan.end()) {
      Remap.push_back(std::pair<unsigned, unsigned>
          ((*It).second, (*PosInUntouched).second));
      continue;
    }
    if (CurrentUndexIdx >= Untouched->UndefReg.size())
      return false;
    Remap.push_back(std::pair<unsigned, unsigned>
        ((*It).second, Untouched->UndefReg[CurrentUndexIdx++]));
  }

  return true;
}

static
unsigned getReassignedChan(
    const std::vector<std::pair<unsigned, unsigned> > &RemapChan,
    unsigned Chan) {
  for (unsigned j = 0, je = RemapChan.size(); j < je; j++) {
    if (RemapChan[j].first == Chan)
      return RemapChan[j].second;
  }
  llvm_unreachable("Chan wasn't reassigned");
}

MachineInstr *R600VectorRegMerger::RebuildVector(
    RegSeqInfo *RSI, const RegSeqInfo *BaseRSI,
    const std::vector<std::pair<unsigned, unsigned> > &RemapChan) const {
  unsigned Reg = RSI->Instr->getOperand(0).getReg();
  MachineBasicBlock::iterator Pos = RSI->Instr;
  MachineBasicBlock &MBB = *Pos->getParent();
  DebugLoc DL = Pos->getDebugLoc();

  unsigned SrcVec = BaseRSI->Instr->getOperand(0).getReg();
  DenseMap<unsigned, unsigned> UpdatedRegToChan = BaseRSI->RegToChan;
  std::vector<unsigned> UpdatedUndef = BaseRSI->UndefReg;
  for (DenseMap<unsigned, unsigned>::iterator It = RSI->RegToChan.begin(),
      E = RSI->RegToChan.end(); It != E; ++It) {
    unsigned DstReg = MRI->createVirtualRegister(&AMDGPU::R600_Reg128RegClass);
    unsigned SubReg = (*It).first;
    unsigned Swizzle = (*It).second;
    unsigned Chan = getReassignedChan(RemapChan, Swizzle);

    MachineInstr *Tmp = BuildMI(MBB, Pos, DL, TII->get(AMDGPU::INSERT_SUBREG),
        DstReg)
        .addReg(SrcVec)
        .addReg(SubReg)
        .addImm(Chan);
    UpdatedRegToChan[SubReg] = Chan;
    std::vector<unsigned>::iterator ChanPos = find(UpdatedUndef, Chan);
    if (ChanPos != UpdatedUndef.end())
      UpdatedUndef.erase(ChanPos);
    assert(!is_contained(UpdatedUndef, Chan) &&
           "UpdatedUndef shouldn't contain Chan more than once!");
    DEBUG(dbgs() << "    ->"; Tmp->dump(););
    (void)Tmp;
    SrcVec = DstReg;
  }
  MachineInstr *NewMI =
      BuildMI(MBB, Pos, DL, TII->get(AMDGPU::COPY), Reg).addReg(SrcVec);
  DEBUG(dbgs() << "    ->"; NewMI->dump(););

  DEBUG(dbgs() << "  Updating Swizzle:\n");
  for (MachineRegisterInfo::use_instr_iterator It = MRI->use_instr_begin(Reg),
      E = MRI->use_instr_end(); It != E; ++It) {
    DEBUG(dbgs() << "    ";(*It).dump(); dbgs() << "    ->");
    SwizzleInput(*It, RemapChan);
    DEBUG((*It).dump());
  }
  RSI->Instr->eraseFromParent();

  // Update RSI
  RSI->Instr = NewMI;
  RSI->RegToChan = UpdatedRegToChan;
  RSI->UndefReg = UpdatedUndef;

  return NewMI;
}

void R600VectorRegMerger::RemoveMI(MachineInstr *MI) {
  for (InstructionSetMap::iterator It = PreviousRegSeqByReg.begin(),
      E = PreviousRegSeqByReg.end(); It != E; ++It) {
    std::vector<MachineInstr *> &MIs = (*It).second;
    MIs.erase(find(MIs, MI), MIs.end());
  }
  for (InstructionSetMap::iterator It = PreviousRegSeqByUndefCount.begin(),
      E = PreviousRegSeqByUndefCount.end(); It != E; ++It) {
    std::vector<MachineInstr *> &MIs = (*It).second;
    MIs.erase(find(MIs, MI), MIs.end());
  }
}

void R600VectorRegMerger::SwizzleInput(MachineInstr &MI,
    const std::vector<std::pair<unsigned, unsigned> > &RemapChan) const {
  unsigned Offset;
  if (TII->get(MI.getOpcode()).TSFlags & R600_InstFlag::TEX_INST)
    Offset = 2;
  else
    Offset = 3;
  for (unsigned i = 0; i < 4; i++) {
    unsigned Swizzle = MI.getOperand(i + Offset).getImm() + 1;
    for (unsigned j = 0, e = RemapChan.size(); j < e; j++) {
      if (RemapChan[j].first == Swizzle) {
        MI.getOperand(i + Offset).setImm(RemapChan[j].second - 1);
        break;
      }
    }
  }
}

bool R600VectorRegMerger::areAllUsesSwizzeable(unsigned Reg) const {
  for (MachineRegisterInfo::use_instr_iterator It = MRI->use_instr_begin(Reg),
      E = MRI->use_instr_end(); It != E; ++It) {
    if (!canSwizzle(*It))
      return false;
  }
  return true;
}

bool R600VectorRegMerger::tryMergeUsingCommonSlot(RegSeqInfo &RSI,
    RegSeqInfo &CompatibleRSI,
    std::vector<std::pair<unsigned, unsigned> > &RemapChan) {
  for (MachineInstr::mop_iterator MOp = RSI.Instr->operands_begin(),
      MOE = RSI.Instr->operands_end(); MOp != MOE; ++MOp) {
    if (!MOp->isReg())
      continue;
    if (PreviousRegSeqByReg[MOp->getReg()].empty())
      continue;
    for (MachineInstr *MI : PreviousRegSeqByReg[MOp->getReg()]) {
      CompatibleRSI = PreviousRegSeq[MI];
      if (RSI == CompatibleRSI)
        continue;
      if (tryMergeVector(&CompatibleRSI, &RSI, RemapChan))
        return true;
    }
  }
  return false;
}

bool R600VectorRegMerger::tryMergeUsingFreeSlot(RegSeqInfo &RSI,
    RegSeqInfo &CompatibleRSI,
    std::vector<std::pair<unsigned, unsigned> > &RemapChan) {
  unsigned NeededUndefs = 4 - RSI.UndefReg.size();
  if (PreviousRegSeqByUndefCount[NeededUndefs].empty())
    return false;
  std::vector<MachineInstr *> &MIs =
      PreviousRegSeqByUndefCount[NeededUndefs];
  CompatibleRSI = PreviousRegSeq[MIs.back()];
  tryMergeVector(&CompatibleRSI, &RSI, RemapChan);
  return true;
}

void R600VectorRegMerger::trackRSI(const RegSeqInfo &RSI) {
  for (DenseMap<unsigned, unsigned>::const_iterator
  It = RSI.RegToChan.begin(), E = RSI.RegToChan.end(); It != E; ++It) {
    PreviousRegSeqByReg[(*It).first].push_back(RSI.Instr);
  }
  PreviousRegSeqByUndefCount[RSI.UndefReg.size()].push_back(RSI.Instr);
  PreviousRegSeq[RSI.Instr] = RSI;
}

bool R600VectorRegMerger::runOnMachineFunction(MachineFunction &Fn) {
  if (skipFunction(*Fn.getFunction()))
    return false;

  const R600Subtarget &ST = Fn.getSubtarget<R600Subtarget>();
  TII = ST.getInstrInfo();
  MRI = &Fn.getRegInfo();

  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB) {
    MachineBasicBlock *MB = &*MBB;
    PreviousRegSeq.clear();
    PreviousRegSeqByReg.clear();
    PreviousRegSeqByUndefCount.clear();

    for (MachineBasicBlock::iterator MII = MB->begin(), MIIE = MB->end();
         MII != MIIE; ++MII) {
      MachineInstr &MI = *MII;
      if (MI.getOpcode() != AMDGPU::REG_SEQUENCE) {
        if (TII->get(MI.getOpcode()).TSFlags & R600_InstFlag::TEX_INST) {
          unsigned Reg = MI.getOperand(1).getReg();
          for (MachineRegisterInfo::def_instr_iterator
               It = MRI->def_instr_begin(Reg), E = MRI->def_instr_end();
               It != E; ++It) {
            RemoveMI(&(*It));
          }
        }
        continue;
      }

      RegSeqInfo RSI(*MRI, &MI);

      // All uses of MI are swizzeable ?
      unsigned Reg = MI.getOperand(0).getReg();
      if (!areAllUsesSwizzeable(Reg))
        continue;

      DEBUG({
        dbgs() << "Trying to optimize ";
        MI.dump();
      });

      RegSeqInfo CandidateRSI;
      std::vector<std::pair<unsigned, unsigned> > RemapChan;
      DEBUG(dbgs() << "Using common slots...\n";);
      if (tryMergeUsingCommonSlot(RSI, CandidateRSI, RemapChan)) {
        // Remove CandidateRSI mapping
        RemoveMI(CandidateRSI.Instr);
        MII = RebuildVector(&RSI, &CandidateRSI, RemapChan);
        trackRSI(RSI);
        continue;
      }
      DEBUG(dbgs() << "Using free slots...\n";);
      RemapChan.clear();
      if (tryMergeUsingFreeSlot(RSI, CandidateRSI, RemapChan)) {
        RemoveMI(CandidateRSI.Instr);
        MII = RebuildVector(&RSI, &CandidateRSI, RemapChan);
        trackRSI(RSI);
        continue;
      }
      //Failed to merge
      trackRSI(RSI);
    }
  }
  return false;
}

}

llvm::FunctionPass *llvm::createR600VectorRegMerger(TargetMachine &tm) {
  return new R600VectorRegMerger(tm);
}
