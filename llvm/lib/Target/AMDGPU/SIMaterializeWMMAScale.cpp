//===- SIMaterializeWMMAScale.cpp - Materialize WMMA scales to VGPRs ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Copy an SGPR WMMA scale into a low VGPR before the scheduler.
///
/// v_wmma_scale scale operands accept an SGPR or a VGPR. depctr_va_ssrc is one
/// deep, so each VALU instruction that reads an SGPR scale takes the only
/// scalar-source slot and a run of such WMMA ops serializes. One cross-bank
/// copy pays for that slot once; later WMMA ops then read a VGPR.
///
/// A WMMA occupies va_ssrc when either scale is an SGPR. That includes a
/// physical register and a subregister that cannot be copied. Copies are
/// chosen together: (S, T0) and (S, T1) stay on the SGPR, because copying
/// only S leaves both instructions on the counter and the copy adds one.
/// A set of copies is emitted when the instructions that then read no SGPR
/// scale outnumber the copies. A tie is kept when every copy sits outside a
/// loop that would otherwise repeat that many reads. A copy inside a loop
/// that modifies exec stays in that loop. The VGPR latches each lane as it
/// drops out, so a use after the loop keeps the SGPR and reads the
/// reconverged scalar. A subregister use is copied at that width, so two
/// lanes of one SGPR tuple stay independent.
//
//===----------------------------------------------------------------------===//

#include "SIMaterializeWMMAScale.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "si-materialize-wmma-scale"

STATISTIC(NumWMMAScalesMaterialized,
          "Number of SGPR WMMA scales materialized into VGPRs");

/// VALU SGPR-source issue slots. depctr_va_ssrc is a 1-bit counter.
static constexpr unsigned VaSsrcDepth = 1;

/// Exact search over copy sets. Larger components fall back to single-key
/// decisions, which never copy a scale that still leaves its reader on an SGPR.
static constexpr unsigned MaxExactCopySet = 16;

namespace {

using ScaleKey = std::pair<Register, unsigned>;

struct ScaleCandidate {
  SmallVector<unsigned, 2> KeyIdx;
  bool LoopHoist = false;
};

/// Pick a subset of \p CompKeys. An instruction is cleared only when every
/// key it needs is copied, so a shared scale is not copied while its partner
/// stays on an SGPR. \p Selected is indexed by the global key id.
void chooseCopies(ArrayRef<ScaleCandidate> Cands, ArrayRef<unsigned> CompKeys,
                  SmallVectorImpl<bool> &Selected) {
  const unsigned K = CompKeys.size();
  DenseMap<unsigned, unsigned> Local;
  for (auto [I, Key] : enumerate(CompKeys))
    Local[Key] = I;

  auto consider = [&](unsigned Cost, unsigned Benefit, bool AllHoist,
                      function_ref<void()> Accept) {
    if (Benefit == 0 || Cost == 0)
      return;
    const bool Pays = Benefit > Cost * VaSsrcDepth ||
                      (Benefit == Cost * VaSsrcDepth && AllHoist);
    if (Pays)
      Accept();
  };

  if (K > MaxExactCopySet) {
    for (unsigned L = 0; L < K; ++L) {
      unsigned Benefit = 0;
      bool AllHoist = true;
      for (const ScaleCandidate &C : Cands) {
        if (C.KeyIdx.size() != 1 || Local[C.KeyIdx[0]] != L)
          continue;
        ++Benefit;
        AllHoist &= C.LoopHoist;
      }
      consider(1, Benefit, AllHoist, [&] { Selected[CompKeys[L]] = true; });
    }
    return;
  }

  int BestScore = 0;
  bool BestHoist = false;
  unsigned BestCost = 0;
  uint32_t BestMask = 0;
  for (uint32_t Mask = 1; Mask < (1u << K); ++Mask) {
    const unsigned Cost = llvm::popcount(Mask);
    unsigned Benefit = 0;
    bool AllHoist = true;
    for (const ScaleCandidate &C : Cands) {
      bool Covered = true;
      for (unsigned Key : C.KeyIdx)
        Covered &= (Mask & (1u << Local[Key])) != 0;
      if (!Covered)
        continue;
      ++Benefit;
      AllHoist &= C.LoopHoist;
    }
    if (Benefit == 0)
      continue;
    const int Score =
        static_cast<int>(Benefit) - static_cast<int>(Cost * VaSsrcDepth);
    const bool Hoist = Score == 0 && AllHoist;
    if (Score < 0 || (Score == 0 && !Hoist))
      continue;
    const bool Better = Score != BestScore   ? Score > BestScore
                        : Hoist != BestHoist ? Hoist
                                             : Cost < BestCost;
    if (!Better)
      continue;
    BestScore = Score;
    BestHoist = Hoist;
    BestCost = Cost;
    BestMask = Mask;
  }
  for (unsigned L = 0; L < K; ++L)
    if (BestMask & (1u << L))
      Selected[CompKeys[L]] = true;
}

class SIMaterializeWMMAScale {
  struct ScaleUse {
    MachineInstr *MI;
    unsigned OpNo;
  };

  MachineRegisterInfo *MRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  const MachineLoopInfo *MLI = nullptr;
  DenseMap<const MachineLoop *, bool> ExecLoopCache;

  bool loopModifiesExec(const MachineLoop &L);
  bool crossesUnsafeDivergentLoop(const MachineBasicBlock *DefBB,
                                  const MachineBasicBlock *UseBB);
  bool materialize(Register Reg, unsigned SubReg, ArrayRef<ScaleUse> Uses);

public:
  SIMaterializeWMMAScale() = default;
  bool run(MachineFunction &MF, const MachineLoopInfo &Loops);
};

class SIMaterializeWMMAScaleLegacy : public MachineFunctionPass {
public:
  static char ID;

  SIMaterializeWMMAScaleLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "SI Materialize WMMA Scale"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }
};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(SIMaterializeWMMAScaleLegacy, DEBUG_TYPE,
                      "SI Materialize WMMA Scale", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(SIMaterializeWMMAScaleLegacy, DEBUG_TYPE,
                    "SI Materialize WMMA Scale", false, false)

char SIMaterializeWMMAScaleLegacy::ID = 0;

char &llvm::SIMaterializeWMMAScaleLegacyID = SIMaterializeWMMAScaleLegacy::ID;

FunctionPass *llvm::createSIMaterializeWMMAScaleLegacyPass() {
  return new SIMaterializeWMMAScaleLegacy();
}

bool SIMaterializeWMMAScale::loopModifiesExec(const MachineLoop &L) {
  auto [It, Inserted] = ExecLoopCache.try_emplace(&L, false);
  if (!Inserted)
    return It->second;

  for (const MachineBasicBlock *MBB : L.getBlocks()) {
    for (const MachineInstr &MI : *MBB) {
      if (MI.modifiesRegister(TRI->getExec(), TRI)) {
        It->second = true;
        return true;
      }
    }
  }
  return false;
}

bool SIMaterializeWMMAScale::crossesUnsafeDivergentLoop(
    const MachineBasicBlock *DefBB, const MachineBasicBlock *UseBB) {
  // A loop that contains the copy and not the use, and that modifies exec,
  // reconverges to one SGPR value. The VGPR would still hold each lane's
  // latched value.
  for (const MachineLoop *L = MLI->getLoopFor(DefBB); L && !L->contains(UseBB);
       L = L->getParentLoop()) {
    if (loopModifiesExec(*L))
      return true;
  }
  return false;
}

bool SIMaterializeWMMAScale::materialize(Register Reg, unsigned SubReg,
                                         ArrayRef<ScaleUse> Uses) {
  MachineInstr *Def = MRI->getVRegDef(Reg);
  if (!Def || Def->isTerminator())
    return false;

  // Size the VGPR from the subregister, not the surrounding tuple. Copying
  // the whole SGPR would move lanes this use does not read.
  unsigned BitWidth =
      SubReg ? TRI->getSubRegIdxSize(SubReg)
             : TRI->getRegSizeInBits(*MRI->getRegClass(Reg)).getFixedValue();
  const TargetRegisterClass *VRC =
      TRI->getAlignedLo256VGPRClassForBitWidth(BitWidth);
  if (!VRC || Uses.empty())
    return false;

  Register VReg = MRI->createVirtualRegister(VRC);
  const MachineBasicBlock *DefBB = Def->getParent();
  MachineBasicBlock &MBB = *Def->getParent();
  MachineBasicBlock::iterator InsertPt =
      Def->isPHI() ? MBB.getFirstNonPHI() : std::next(Def->getIterator());
  BuildMI(MBB, InsertPt, Def->getDebugLoc(), TII->get(TargetOpcode::COPY), VReg)
      .addUse(Reg, {}, SubReg);

  for (const ScaleUse &Use : Uses) {
    if (crossesUnsafeDivergentLoop(DefBB, Use.MI->getParent()))
      continue;
    MachineOperand &MO = Use.MI->getOperand(Use.OpNo);
    MO.setReg(VReg);
    MO.setSubReg(0);
    MO.setIsKill(false);
  }

  LLVM_DEBUG(dbgs() << "Materialized WMMA scale " << printReg(Reg, TRI, SubReg)
                    << " -> " << printReg(VReg, TRI) << '\n');
  ++NumWMMAScalesMaterialized;
  return true;
}

bool SIMaterializeWMMAScale::run(MachineFunction &MF,
                                 const MachineLoopInfo &Loops) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasWMMAN16Insts() && !ST.hasWMMAF4Insts())
    return false;

  MRI = &MF.getRegInfo();
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  MLI = &Loops;
  ExecLoopCache.clear();

  // Insertion order keeps copy placement deterministic. Each lane is its own
  // scale. A physical SGPR, or a subregister with no legal VGPR copy, still
  // occupies va_ssrc and is not a key.
  MapVector<ScaleKey, SmallVector<ScaleUse, 4>> ScaleUses;
  DenseMap<ScaleKey, unsigned> KeyIndex;
  SmallVector<const MachineBasicBlock *, 8> KeyDefBB;
  SmallVector<ScaleCandidate, 8> Candidates;

  auto canCopy = [&](const MachineOperand &MO, const MachineBasicBlock *UseBB,
                     const MachineBasicBlock *&DefBB) {
    if (!MO.isReg() || !MO.getReg().isVirtual())
      return false;
    MachineInstr *Def = MRI->getVRegDef(MO.getReg());
    if (!Def || Def->isTerminator())
      return false;
    unsigned SubReg = MO.getSubReg();
    unsigned BitWidth =
        SubReg ? TRI->getSubRegIdxSize(SubReg)
               : TRI->getRegSizeInBits(*MRI->getRegClass(MO.getReg()))
                     .getFixedValue();
    if (!TRI->getAlignedLo256VGPRClassForBitWidth(BitWidth))
      return false;
    DefBB = Def->getParent();
    return !crossesUnsafeDivergentLoop(DefBB, UseBB);
  };

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!SIInstrInfo::isWMMA(MI) ||
          !AMDGPU::hasNamedOperand(MI.getOpcode(), AMDGPU::OpName::scale_src0))
        continue;

      bool FixedSGPR = false;
      SmallVector<unsigned, 2> Keys;
      for (AMDGPU::OpName Name :
           {AMDGPU::OpName::scale_src0, AMDGPU::OpName::scale_src1}) {
        int Idx = AMDGPU::getNamedOperandIdx(MI.getOpcode(), Name);
        if (Idx < 0)
          continue;
        MachineOperand &MO = MI.getOperand(Idx);
        if (!MO.isReg() || !MO.getReg() || !TRI->isSGPRReg(*MRI, MO.getReg()))
          continue;
        const MachineBasicBlock *DefBB = nullptr;
        if (!canCopy(MO, MI.getParent(), DefBB)) {
          FixedSGPR = true;
          continue;
        }
        ScaleKey Key{MO.getReg(), MO.getSubReg()};
        auto [It, Inserted] = KeyIndex.try_emplace(Key, KeyIndex.size());
        if (Inserted) {
          ScaleUses[Key] = {};
          KeyDefBB.push_back(DefBB);
        }
        ScaleUses[Key].push_back({&MI, static_cast<unsigned>(Idx)});
        if (!is_contained(Keys, It->second))
          Keys.push_back(It->second);
      }
      if (FixedSGPR || Keys.empty())
        continue;

      const MachineLoop *UseLoop = MLI->getLoopFor(MI.getParent());
      bool Hoist = UseLoop != nullptr;
      if (Hoist) {
        for (unsigned Key : Keys)
          Hoist &= !UseLoop->contains(KeyDefBB[Key]);
      }
      Candidates.push_back({std::move(Keys), Hoist});
    }
  }

  SmallVector<bool, 8> Selected(KeyIndex.size(), false);
  if (!Candidates.empty()) {
    SmallVector<unsigned, 8> Parent(KeyIndex.size());
    for (unsigned I = 0, E = Parent.size(); I != E; ++I)
      Parent[I] = I;
    auto findRoot = [&](unsigned X) {
      while (Parent[X] != X) {
        Parent[X] = Parent[Parent[X]];
        X = Parent[X];
      }
      return X;
    };
    for (const ScaleCandidate &C : Candidates) {
      unsigned Root = findRoot(C.KeyIdx.front());
      for (unsigned Key : C.KeyIdx)
        Parent[findRoot(Key)] = Root;
    }

    MapVector<unsigned, SmallVector<unsigned, 8>> ByRoot;
    for (auto [I, C] : enumerate(Candidates))
      ByRoot[findRoot(C.KeyIdx.front())].push_back(I);

    for (const auto &[Root, CandIdxs] : ByRoot) {
      (void)Root;
      SmallVector<unsigned, 8> CompKeys;
      SmallVector<ScaleCandidate, 8> CompCands;
      for (unsigned CI : CandIdxs) {
        CompCands.push_back(Candidates[CI]);
        for (unsigned Key : Candidates[CI].KeyIdx)
          if (!is_contained(CompKeys, Key))
            CompKeys.push_back(Key);
      }
      chooseCopies(CompCands, CompKeys, Selected);
    }
  }

  bool Changed = false;
  unsigned Idx = 0;
  for (const auto &[Key, Uses] : ScaleUses) {
    if (Selected[Idx])
      Changed |= materialize(Key.first, Key.second, Uses);
    ++Idx;
  }
  return Changed;
}

bool SIMaterializeWMMAScaleLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  auto &Loops = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  return SIMaterializeWMMAScale().run(MF, Loops);
}

PreservedAnalyses
SIMaterializeWMMAScalePass::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  MFPropsModifier _(*this, MF);
  if (MF.getFunction().hasOptNone())
    return PreservedAnalyses::all();

  auto &Loops = MFAM.getResult<MachineLoopAnalysis>(MF);
  if (!SIMaterializeWMMAScale().run(MF, Loops))
    return PreservedAnalyses::all();

  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserve<MachineLoopAnalysis>();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
