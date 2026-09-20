//===- GCNCreateVOPD.cpp - Create VOPD Instructions ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Form VOPD instructions from adjacent VALU operations on wave32. The post-RA
/// scheduler puts likely component pairs next to each other. This pass checks
/// their final physical-register constraints and selects a non-overlapping set.
///
/// VOPD3 components cannot encode literal operands. When all non-inline
/// immediates in a pair have the same 32-bit value, the pass can materialize
/// that value in an SGPR which is free over the pair. The move and its register
/// stay pair-local, so fusion adds at most one move and does not extend
/// register pressure across pairs.
///
/// The pass considers every adjacent candidate. It first maximizes the number
/// of pairs, then minimizes scalar moves among equal-size matchings. The
/// earlier candidate wins an exact tie.
///
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "GCNVOPDUtils.h"
#include "SIInstrInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gcn-create-vopd"
STATISTIC(NumVOPDCreated, "Number of VOPD Insts Created.");
STATISTIC(NumLiteralsMaterialized,
          "Number of immediates moved into a scalar register to allow VOPD3 "
          "pairing.");
STATISTIC(NumCandidateEdgesWithoutFreeSGPR,
          "Number of VOPD3 candidate edges skipped because no scalar register "
          "was free for their immediate.");

using namespace llvm;

namespace {

struct VOPDCandidate {
  VOPDMatchInfo Match;
  /// The register for Match.LiteralFixups, or a null register if none is free.
  Register MaterializationReg;

  bool needsMaterialization() const { return !Match.LiteralFixups.empty(); }

  bool isFeasible() const {
    return !needsMaterialization() || MaterializationReg;
  }
};

} // namespace

/// Add everything the instructions in [\p Begin, \p RangeEnd] touch to
/// \p Live, which already holds what is live after \p RangeEnd.
static void addRangeUses(LiveRegUnits &Live, MachineBasicBlock::iterator Begin,
                         MachineInstr &RangeEnd) {
  MachineBasicBlock::iterator After =
      std::next(MachineBasicBlock::iterator(&RangeEnd));
  for (MachineInstr &MI : make_range(Begin, After)) {
    if (!MI.isDebugInstr())
      Live.accumulate(MI);
  }
}

/// Return a scalar register which every fixup can read and which no value in
/// \p Live occupies, or a null register. Low registers are preferred, because
/// those are most likely in use already, so the function's register count does
/// not grow.
static Register takeFreeSGPR(const GCNSubtarget &ST,
                             const MachineRegisterInfo &MRI,
                             const LiveRegUnits &Live,
                             ArrayRef<VOPDLiteralFixup> Fixups) {
  assert(!Fixups.empty());
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  for (MCPhysReg Reg : AMDGPU::SGPR_32RegClass) {
    // SGPR_32 also holds the halves of VCC. Writing those changes VCCZ,
    // which is not modelled by \p Live, so a free half is not safe to use.
    if (MRI.isReserved(Reg) || TRI->isSubRegisterEq(AMDGPU::VCC, Reg) ||
        !Live.available(Reg))
      continue;
    if (!all_of(Fixups, [Reg](const VOPDLiteralFixup &Fixup) {
          return Fixup.SlotRC->contains(Reg);
        }))
      continue;
    return Reg;
  }
  return Register();
}

namespace {

class GCNCreateVOPD {
public:
  const GCNSubtarget *ST = nullptr;

  void
  assignMaterializationRegisters(MachineBasicBlock &MBB,
                                 MutableArrayRef<VOPDCandidate> Candidates) {
    auto Candidate = Candidates.rbegin();
    auto SkipPlainCandidates = [&] {
      while (Candidate != Candidates.rend() &&
             !Candidate->needsMaterialization())
        ++Candidate;
    };
    SkipPlainCandidates();
    if (Candidate == Candidates.rend())
      return;

    const MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
    LiveRegUnits Walk(*ST->getRegisterInfo());
    Walk.addLiveOuts(MBB);

    // Before stepping over an instruction, Walk holds what is live immediately
    // after it. This answers every pair-local range in one backward walk.
    for (MachineInstr &MI : reverse(MBB)) {
      if (Candidate != Candidates.rend() &&
          Candidate->Match.InOrder[1] == &MI) {
        LiveRegUnits RangeLive = Walk;
        addRangeUses(RangeLive, Candidate->Match.InOrder[0]->getIterator(),
                     *Candidate->Match.InOrder[1]);
        Candidate->MaterializationReg =
            takeFreeSGPR(*ST, MRI, RangeLive, Candidate->Match.LiteralFixups);
        ++Candidate;
        SkipPlainCandidates();
      }
      if (!MI.isDebugInstr())
        Walk.stepBackward(MI);
    }
    assert(Candidate == Candidates.rend() &&
           "every candidate range must end in this block");
  }

  static SmallVector<VOPDCandidate *, 8>
  selectCandidates(MutableArrayRef<VOPDCandidate> Candidates) {
    struct Score {
      unsigned NumPairs = 0;
      unsigned NumMoves = 0;
    };

    const size_t NumCandidates = Candidates.size();
    SmallVector<Score, 8> Best(NumCandidates + 1);
    SmallBitVector Take(NumCandidates);
    auto NextNonOverlapping = [&](size_t I) {
      size_t Next = I + 1;
      if (Next != NumCandidates &&
          Candidates[Next].Match.InOrder[0] == Candidates[I].Match.InOrder[1])
        ++Next;
      return Next;
    };

    // Maximize the number of pairs, then minimize the moves they need. Taking
    // the current edge on an exact tie preserves the old left-to-right choice.
    for (size_t I = NumCandidates; I-- != 0;) {
      Best[I] = Best[I + 1];
      if (!Candidates[I].isFeasible()) {
        ++NumCandidateEdgesWithoutFreeSGPR;
        continue;
      }

      Score With = Best[NextNonOverlapping(I)];
      ++With.NumPairs;
      With.NumMoves += Candidates[I].needsMaterialization();
      if (With.NumPairs > Best[I].NumPairs ||
          (With.NumPairs == Best[I].NumPairs &&
           With.NumMoves <= Best[I].NumMoves)) {
        Best[I] = With;
        Take.set(I);
      }
    }

    SmallVector<VOPDCandidate *, 8> Selected;
    for (size_t I = 0; I != NumCandidates;) {
      if (!Take[I]) {
        ++I;
        continue;
      }
      Selected.push_back(&Candidates[I]);
      I = NextNonOverlapping(I);
    }
    return Selected;
  }

  void materializeLiteral(const SIInstrInfo &TII, VOPDCandidate &Candidate) {
    if (!Candidate.needsMaterialization())
      return;

    ArrayRef<VOPDLiteralFixup> Fixups = Candidate.Match.LiteralFixups;
    assert(Candidate.MaterializationReg);
    assert(all_of(Fixups,
                  [Imm = Fixups.front().Imm](const VOPDLiteralFixup &Fixup) {
                    return Fixup.Imm == Imm;
                  }));

    MachineInstr *InsertPt = Candidate.Match.InOrder[0];
    BuildMI(*InsertPt->getParent(), InsertPt, DebugLoc(),
            TII.get(AMDGPU::S_MOV_B32), Candidate.MaterializationReg)
        .addImm(Fixups.front().Imm);
    ++NumLiteralsMaterialized;

    for (const VOPDLiteralFixup &Fixup : Fixups) {
      MachineInstr *MI = Fixup.CompIdx == AMDGPU::VOPD::X
                             ? Candidate.Match.getMIX()
                             : Candidate.Match.getMIY();
      MI->getOperand(Fixup.OpIdx)
          .ChangeToRegister(Candidate.MaterializationReg, /*isDef=*/false);
    }
  }

  bool doReplace(const SIInstrInfo *SII, VOPDMatchInfo &Match) {
    MachineInstr *MIX = Match.getMIX();
    MachineInstr *MIY = Match.getMIY();
    unsigned Opc1 = MIX->getOpcode();
    unsigned Opc2 = MIY->getOpcode();
    unsigned EncodingFamily =
        AMDGPU::getVOPDEncodingFamily(SII->getSubtarget());
    int NewOpcode =
        AMDGPU::getVOPDFull(AMDGPU::getVOPDOpcode(Opc1, Match.IsVOPD3),
                            AMDGPU::getVOPDOpcode(Opc2, Match.IsVOPD3),
                            EncodingFamily, Match.IsVOPD3);
    assert(NewOpcode != -1 &&
           "Should have previously determined this as a possible VOPD\n");

    auto VOPDInst =
        BuildMI(*MIX->getParent(), MIX, MIX->getDebugLoc(), SII->get(NewOpcode))
            .setMIFlags(MIX->getFlags() | MIY->getFlags());

    namespace VOPD = AMDGPU::VOPD;
    MachineInstr *MI[] = {MIX, MIY};
    auto InstInfo = AMDGPU::getVOPDInstInfo(MIX->getDesc(), MIY->getDesc());

    for (auto CompIdx : VOPD::COMPONENTS) {
      auto MCOprIdx = InstInfo[CompIdx].getIndexOfDstInMCOperands();
      VOPDInst.add(MI[CompIdx]->getOperand(MCOprIdx));
    }

    const AMDGPU::OpName Mods[2][3] = {
        {AMDGPU::OpName::src0X_modifiers, AMDGPU::OpName::vsrc1X_modifiers,
         AMDGPU::OpName::vsrc2X_modifiers},
        {AMDGPU::OpName::src0Y_modifiers, AMDGPU::OpName::vsrc1Y_modifiers,
         AMDGPU::OpName::vsrc2Y_modifiers}};
    const AMDGPU::OpName SrcMods[3] = {AMDGPU::OpName::src0_modifiers,
                                       AMDGPU::OpName::src1_modifiers,
                                       AMDGPU::OpName::src2_modifiers};
    const unsigned VOPDOpc = VOPDInst->getOpcode();

    for (auto CompIdx : VOPD::COMPONENTS) {
      auto CompSrcOprNum = InstInfo[CompIdx].getCompSrcOperandsNum();
      bool IsVOP3 = SII->isVOP3(*MI[CompIdx]);
      for (unsigned CompSrcIdx = 0; CompSrcIdx < CompSrcOprNum; ++CompSrcIdx) {
        if (AMDGPU::hasNamedOperand(VOPDOpc, Mods[CompIdx][CompSrcIdx])) {
          const MachineOperand *Mod =
              SII->getNamedOperand(*MI[CompIdx], SrcMods[CompSrcIdx]);
          VOPDInst.addImm(Mod ? Mod->getImm() : 0);
        }
        auto MCOprIdx =
            InstInfo[CompIdx].getIndexOfSrcInMCOperands(CompSrcIdx, IsVOP3);
        VOPDInst.add(MI[CompIdx]->getOperand(MCOprIdx));
      }
      if (MI[CompIdx]->getOpcode() == AMDGPU::V_CNDMASK_B32_e32 &&
          Match.IsVOPD3)
        VOPDInst.addReg(AMDGPU::VCC_LO);
    }

    if (Match.IsVOPD3) {
      if (unsigned BitOp2 = AMDGPU::getBitOp2(Opc2))
        VOPDInst.addImm(BitOp2);
    }

    SII->fixImplicitOperands(*VOPDInst);
    for (auto CompIdx : VOPD::COMPONENTS)
      VOPDInst.copyImplicitOps(*MI[CompIdx]);

    LLVM_DEBUG(dbgs() << "VOPD Fused: " << *VOPDInst << " from\tX: " << *MIX
                      << "\tY: " << *MIY << "\n");

    for (auto CompIdx : VOPD::COMPONENTS)
      MI[CompIdx]->eraseFromParent();

    ++NumVOPDCreated;
    return true;
  }

  bool run(MachineFunction &MF) {
    ST = &MF.getSubtarget<GCNSubtarget>();
    if (!AMDGPU::hasVOPD(*ST) || !ST->isWave32())
      return false;
    LLVM_DEBUG(dbgs() << "CreateVOPD Pass:\n");

    const SIInstrInfo *SII = ST->getInstrInfo();
    bool Changed = false;

    for (MachineBasicBlock &MBB : MF) {
      SmallVector<VOPDCandidate, 8> Candidates;
      auto MII = MBB.begin(), E = MBB.end();
      while (MII != E) {
        MachineInstr *FirstMI = &*MII;
        MII = next_nodbg(MII, MBB.end());
        if (MII == MBB.end())
          break;
        if (FirstMI->isDebugInstr())
          continue;
        MachineInstr *SecondMI = &*MII;

        if (std::optional<VOPDMatchInfo> Match =
                tryMatchVOPDPair(*SII, *FirstMI, *SecondMI))
          Candidates.push_back({std::move(*Match), Register()});
      }

      assignMaterializationRegisters(MBB, Candidates);
      SmallVector<VOPDCandidate *, 8> Selected = selectCandidates(Candidates);
      for (VOPDCandidate *Candidate : Selected) {
        materializeLiteral(*SII, *Candidate);
        Changed |= doReplace(SII, Candidate->Match);
      }
    }

    return Changed;
  }
};

class GCNCreateVOPDLegacy : public MachineFunctionPass {
public:
  static char ID;
  GCNCreateVOPDLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "GCN Create VOPD Instructions";
  }

protected:
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;

    return GCNCreateVOPD().run(MF);
  }
};

} // namespace

PreservedAnalyses
llvm::GCNCreateVOPDPass::run(MachineFunction &MF,
                             MachineFunctionAnalysisManager &AM) {
  if (!GCNCreateVOPD().run(MF))
    return PreservedAnalyses::all();
  return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
}

char GCNCreateVOPDLegacy::ID = 0;

char &llvm::GCNCreateVOPDID = GCNCreateVOPDLegacy::ID;

INITIALIZE_PASS(GCNCreateVOPDLegacy, DEBUG_TYPE, "GCN Create VOPD Instructions",
                false, false)
