//===-- AMDGPUGlobalISelDivergenceLowering.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// GlobalISel pass that selects divergent i1 phis as lane mask phis.
/// Lane mask merging uses same algorithm as SDAG in SILowerI1Copies.
/// Handles all cases of temporal divergence.
/// For divergent non-phi i1 and uniform i1 uses outside of the cycle this pass
/// currently depends on LCSSA to insert phis with one incoming.
/// Also removes the readfirstlane AMDGPUInlineAsmLowering inserts for every
/// inline asm operand with an SGPR constraint once uniformity proves the
/// operand uniform, and the dword split that readfirstlane needed.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUGlobalISelUtils.h"
#include "SILowerI1Copies.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/MachineUniformityAnalysis.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "amdgpu-global-isel-divergence-lowering"

using namespace llvm;

namespace {

class AMDGPUGlobalISelDivergenceLoweringLegacy : public MachineFunctionPass {
public:
  static char ID;

public:
  AMDGPUGlobalISelDivergenceLoweringLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU GlobalISel divergence lowering";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachinePostDominatorTreeWrapperPass>();
    AU.addRequired<MachineUniformityAnalysisPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

class DivergenceLoweringHelper : public AMDGPU::PhiLoweringHelper {
public:
  DivergenceLoweringHelper(MachineFunction &MF, MachineDominatorTree &DT,
                           MachinePostDominatorTree &PDT,
                           MachineUniformityInfo *MUI);

private:
  MachineUniformityInfo *MUI = nullptr;
  MachineIRBuilder B;
  Register buildRegCopyToLaneMask(Register Reg);

public:
  void markAsLaneMask(Register DstReg) const override;
  void getCandidatesForLowering(
      SmallVectorImpl<MachineInstr *> &Vreg1Phis) const override;
  void collectIncomingValuesFromPhi(
      const MachineInstr *MI,
      SmallVectorImpl<AMDGPU::Incoming> &Incomings) const override;
  void replaceDstReg(Register NewReg, Register OldReg,
                     MachineBasicBlock *MBB) override;
  void buildMergeLaneMasks(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I, const DebugLoc &DL,
                           Register DstReg, Register PrevReg,
                           Register CurReg) override;
  void constrainAsLaneMask(AMDGPU::Incoming &In) override;

  bool lowerTemporalDivergence();
  bool lowerTemporalDivergenceI1();
  bool isOnlyUsedByInlineAsm(Register Reg) const;
  bool foldUniformReadFirstLane();
  bool foldSplitAsmOperand();
};

DivergenceLoweringHelper::DivergenceLoweringHelper(
    MachineFunction &MF, MachineDominatorTree &DT,
    MachinePostDominatorTree &PDT, MachineUniformityInfo *MUI)
    : PhiLoweringHelper(MF, DT, PDT), MUI(MUI), B(MF) {}

// _(s1) -> SReg_32/64(s1)
void DivergenceLoweringHelper::markAsLaneMask(Register DstReg) const {
  assert(MRI->getType(DstReg) == LLT::scalar(1));

  if (MRI->getRegClassOrNull(DstReg)) {
    if (MRI->constrainRegClass(DstReg, ST->getBoolRC()))
      return;
    llvm_unreachable("Failed to constrain register class");
  }

  MRI->setRegClass(DstReg, ST->getBoolRC());
}

void DivergenceLoweringHelper::getCandidatesForLowering(
    SmallVectorImpl<MachineInstr *> &Vreg1Phis) const {
  LLT S1 = LLT::scalar(1);

  // Add divergent i1 G_PHIs to the list. Only consider G_PHI instructions,
  // not PHI instructions that may have been created by earlier lowering stages
  // (e.g., lowerTemporalDivergenceI1).
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB.phis()) {
      if (MI.getOpcode() != TargetOpcode::G_PHI)
        continue;
      Register Dst = MI.getOperand(0).getReg();
      if (MRI->getType(Dst) == S1 && MUI->isDivergentAtDef(Dst))
        Vreg1Phis.push_back(&MI);
    }
  }
}

void DivergenceLoweringHelper::collectIncomingValuesFromPhi(
    const MachineInstr *MI,
    SmallVectorImpl<AMDGPU::Incoming> &Incomings) const {
  for (unsigned i = 1; i < MI->getNumOperands(); i += 2) {
    Incomings.emplace_back(MI->getOperand(i).getReg(),
                           MI->getOperand(i + 1).getMBB(), Register());
  }
}

void DivergenceLoweringHelper::replaceDstReg(Register NewReg, Register OldReg,
                                             MachineBasicBlock *MBB) {
  BuildMI(*MBB, MBB->getFirstNonPHI(), {}, TII->get(AMDGPU::COPY), OldReg)
      .addReg(NewReg);
}

// Copy Reg to new lane mask register, insert a copy after instruction that
// defines Reg while skipping phis if needed.
Register DivergenceLoweringHelper::buildRegCopyToLaneMask(Register Reg) {
  Register LaneMask = AMDGPU::createLaneMaskReg(MRI, LaneMaskRegAttrs);
  MachineInstr *Instr = MRI->getVRegDef(Reg);
  MachineBasicBlock *MBB = Instr->getParent();
  B.setInsertPt(*MBB, MBB->SkipPHIsAndLabels(std::next(Instr->getIterator())));
  B.buildCopy(LaneMask, Reg);
  return LaneMask;
}

// bb.previous
//   %PrevReg = ...
//
// bb.current
//   %CurReg = ...
//
//   %DstReg - not defined
//
// -> (wave32 example, new registers have sreg_32 reg class and S1 LLT)
//
// bb.previous
//   %PrevReg = ...
//   %PrevRegCopy:sreg_32(s1) = COPY %PrevReg
//
// bb.current
//   %CurReg = ...
//   %CurRegCopy:sreg_32(s1) = COPY %CurReg
//   ...
//   %PrevMaskedReg:sreg_32(s1) = ANDN2 %PrevRegCopy, ExecReg - active lanes 0
//   %CurMaskedReg:sreg_32(s1)  = AND %ExecReg, CurRegCopy - inactive lanes to 0
//   %DstReg:sreg_32(s1)        = OR %PrevMaskedReg, CurMaskedReg
//
// DstReg = for active lanes rewrite bit in PrevReg with bit from CurReg
void DivergenceLoweringHelper::buildMergeLaneMasks(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator I, const DebugLoc &DL,
    Register DstReg, Register PrevReg, Register CurReg) {
  // DstReg = (PrevReg & !EXEC) | (CurReg & EXEC)
  // TODO: check if inputs are constants or results of a compare.

  Register PrevRegCopy = buildRegCopyToLaneMask(PrevReg);
  Register CurRegCopy = buildRegCopyToLaneMask(CurReg);
  Register PrevMaskedReg = AMDGPU::createLaneMaskReg(MRI, LaneMaskRegAttrs);
  Register CurMaskedReg = AMDGPU::createLaneMaskReg(MRI, LaneMaskRegAttrs);

  B.setInsertPt(MBB, I);
  B.buildInstr(LMC->AndN2Opc, {PrevMaskedReg}, {PrevRegCopy, LMC->ExecReg})
      .setOperandDead(3);
  B.buildInstr(LMC->AndOpc, {CurMaskedReg}, {LMC->ExecReg, CurRegCopy})
      .setOperandDead(3);
  B.buildInstr(LMC->OrOpc, {DstReg}, {PrevMaskedReg, CurMaskedReg})
      .setOperandDead(3);
}

// GlobalISel has to constrain S1 incoming taken as-is with lane mask register
// class. Insert a copy of Incoming.Reg to new lane mask inside Incoming.Block,
// Incoming.Reg becomes that new lane mask.
void DivergenceLoweringHelper::constrainAsLaneMask(AMDGPU::Incoming &In) {
  B.setInsertPt(*In.Block, In.Block->getFirstTerminator());

  auto Copy = B.buildCopy(LLT::scalar(1), In.Reg);
  MRI->setRegClass(Copy.getReg(0), ST->getBoolRC());
  In.Reg = Copy.getReg(0);
}

void replaceUsesOfRegInInstWith(Register Reg, MachineInstr *Inst,
                                Register NewReg) {
  for (MachineOperand &Op : Inst->operands()) {
    if (Op.isReg() && Op.getReg() == Reg)
      Op.setReg(NewReg);
  }
}

bool DivergenceLoweringHelper::lowerTemporalDivergence() {
  AMDGPU::IntrinsicLaneMaskAnalyzer ILMA(MF);
  DenseMap<Register, Register> TDCache;

  for (auto [Reg, UseInst, _] : MUI->getTemporalDivergenceList()) {
    if (MRI->getType(Reg) == LLT::scalar(1) || MUI->isDivergentAtDef(Reg) ||
        ILMA.isS32S64LaneMask(Reg))
      continue;

    Register CachedTDCopy = TDCache.lookup(Reg);
    if (CachedTDCopy) {
      replaceUsesOfRegInInstWith(Reg, UseInst, CachedTDCopy);
      continue;
    }

    MachineInstr *Inst = MRI->getVRegDef(Reg);
    MachineBasicBlock *MBB = Inst->getParent();
    B.setInsertPt(*MBB, MBB->SkipPHIsAndLabels(std::next(Inst->getIterator())));

    Register VgprReg = MRI->createGenericVirtualRegister(MRI->getType(Reg));
    B.buildInstr(AMDGPU::COPY, {VgprReg}, {Reg})
        .addUse(LMC->ExecReg, RegState::Implicit);

    replaceUsesOfRegInInstWith(Reg, UseInst, VgprReg);
    TDCache[Reg] = VgprReg;
  }
  return false;
}

bool DivergenceLoweringHelper::lowerTemporalDivergenceI1() {
  MachineRegisterInfo::VRegAttrs BoolS1 = {ST->getBoolRC(), LLT::scalar(1)};
  initializeLaneMaskRegisterAttributes(BoolS1);
  MachineSSAUpdater SSAUpdater(MF);

  const auto &CInfo = MUI->getCycleInfo();

  // In case of use outside muliple nested cycles or muliple uses we only need
  // to merge lane mask across largest relevant cycle.
  SmallDenseMap<Register, std::pair<CycleRef, Register>> LRCCache;
  for (auto [Reg, UseInst, LRC] : MUI->getTemporalDivergenceList()) {
    if (MRI->getType(Reg) != LLT::scalar(1))
      continue;

    auto [LRCCacheIter, RegNotCached] = LRCCache.try_emplace(Reg);
    auto &CycleMergedMask = LRCCacheIter->getSecond();
    CycleRef &CachedLRC = CycleMergedMask.first;
    if (RegNotCached || CInfo.contains(LRC, CachedLRC)) {
      CachedLRC = LRC;
    }
  }

  for (auto &LRCCacheEntry : LRCCache) {
    Register Reg = LRCCacheEntry.first;
    auto &CycleMergedMask = LRCCacheEntry.getSecond();
    CycleRef Cycle = CycleMergedMask.first;

    Register MergedMask = MRI->createVirtualRegister(BoolS1);
    SSAUpdater.Initialize(MergedMask);

    MachineBasicBlock *MBB = MRI->getDefBlock(Reg);
    SSAUpdater.AddAvailableValue(MBB, MergedMask);

    for (auto Entry : CInfo.getEntries(Cycle)) {
      for (MachineBasicBlock *Pred : Entry->predecessors()) {
        if (!CInfo.contains(Cycle, Pred)) {
          B.setInsertPt(*Pred, Pred->getFirstTerminator());
          auto ImplDef = B.buildInstr(AMDGPU::IMPLICIT_DEF, {BoolS1}, {});
          SSAUpdater.AddAvailableValue(Pred, ImplDef.getReg(0));
        }
      }
    }

    buildMergeLaneMasks(*MBB, MBB->getFirstTerminator(), {}, MergedMask,
                        SSAUpdater.GetValueInMiddleOfBlock(MBB), Reg);

    CycleMergedMask.second = MergedMask;
  }

  for (auto [Reg, UseInst, Cycle] : MUI->getTemporalDivergenceList()) {
    if (MRI->getType(Reg) != LLT::scalar(1))
      continue;

    replaceUsesOfRegInInstWith(Reg, UseInst, LRCCache.lookup(Reg).second);
  }

  return false;
}

/// Returns true if \p Reg is only used by inline asm operands, possibly through
/// the merge the lowering uses to reassemble a value it had to split into
/// dwords. This matches a use pattern, it is not a record of which instruction
/// inserted \p Reg; a transformation that adds an instruction between it and
/// the asm operand makes this return false, which only costs the fold.
bool DivergenceLoweringHelper::isOnlyUsedByInlineAsm(Register Reg) const {
  SmallVector<Register, 8> Worklist = {Reg};
  SmallSet<Register, 8> Visited;
  while (!Worklist.empty()) {
    Register Current = Worklist.pop_back_val();
    if (!Visited.insert(Current).second)
      continue;
    if (MRI->use_nodbg_empty(Current))
      return false;

    for (const MachineOperand &Use : MRI->use_nodbg_operands(Current)) {
      const MachineInstr *UseMI = Use.getParent();
      if (UseMI->isInlineAsm())
        continue;
      if (UseMI->isCopy()) {
        // The copy target is the register the constraint selected.
        Worklist.push_back(UseMI->getOperand(0).getReg());
        continue;
      }
      switch (UseMI->getOpcode()) {
      case TargetOpcode::G_BUILD_VECTOR:
      case TargetOpcode::G_BUILD_VECTOR_TRUNC:
      case TargetOpcode::G_MERGE_VALUES:
      case TargetOpcode::G_CONCAT_VECTORS:
      case TargetOpcode::G_UNMERGE_VALUES:
      case TargetOpcode::G_EXTRACT:
      case TargetOpcode::G_BITCAST:
        for (const MachineOperand &Def : UseMI->defs())
          Worklist.push_back(Def.getReg());
        continue;
      default:
        return false;
      }
    }
  }
  return true;
}

/// Inline asm lowering has no divergence information and inserts readfirstlane
/// for every operand with an SGPR constraint, see AMDGPUInlineAsmLowering. The
/// readfirstlane of a value the uniformity analysis proves uniform is a no-op,
/// so replace it with the copy into the operand register. A readfirstlane left
/// behind is not incorrect, it just costs a round trip through a VGPR: the
/// regbank legalizer rules for amdgcn_readfirstlane expect a divergent source
/// and force a uniform one into a VGPR first.
bool DivergenceLoweringHelper::foldUniformReadFirstLane() {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      auto *GMI = dyn_cast<GIntrinsic>(&MI);
      if (!GMI || !GMI->is(Intrinsic::amdgcn_readfirstlane))
        continue;

      Register Dst = GMI->getReg(0);
      // The intrinsic ID comes after the defs, the argument follows it.
      Register Src = MI.getOperand(GMI->getNumExplicitDefs() + 1).getReg();
      // Folding a lane mask readfirstlane would bypass the lane mask lowering.
      if (MRI->getType(Dst) == LLT::scalar(1) || MUI->isDivergentAtDef(Src) ||
          !isOnlyUsedByInlineAsm(Dst))
        continue;

      B.setInstrAndDebugLoc(MI);
      B.buildCopy(Dst, Src);
      MI.eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

/// The inline asm lowering splits an operand wider than a dword to insert one
/// readfirstlane per dword. Once the pieces are known to be uniform the split
/// is pointless, so reassemble the original value again:
///
/// %bc = G_BITCAST %src
/// %lo, %hi = G_UNMERGE_VALUES %bc
/// %m = G_BUILD_VECTOR %lo, %hi
/// ->
/// %m replaced with %bc
bool DivergenceLoweringHelper::foldSplitAsmOperand() {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      if (MI.getOpcode() != TargetOpcode::G_BUILD_VECTOR &&
          MI.getOpcode() != TargetOpcode::G_MERGE_VALUES)
        continue;

      Register Dst = MI.getOperand(0).getReg();
      if (!isOnlyUsedByInlineAsm(Dst))
        continue;

      // All pieces have to come from the same unmerge, in order. The copies in
      // between only carry the readfirstlane result, remember them so the
      // whole chain can be deleted again.
      SmallVector<Register, 8> Chain;
      GUnmerge *Unmerge = nullptr;
      bool IsSplit = true;
      for (unsigned I = 1; I != MI.getNumOperands() && IsSplit; ++I) {
        Register Piece = MI.getOperand(I).getReg();
        while (MachineInstr *Def = MRI->getVRegDef(Piece)) {
          if (!Def->isCopy() || !MRI->hasOneUse(Piece))
            break;
          Register CopySrc = Def->getOperand(1).getReg();
          if (!CopySrc.isVirtual())
            break;
          Chain.push_back(Piece);
          Piece = CopySrc;
        }

        auto *Def = dyn_cast_or_null<GUnmerge>(MRI->getVRegDef(Piece));
        if (!Def || (Unmerge && Def != Unmerge) ||
            Def->findRegisterDefOperandIdx(Piece, nullptr) != int(I) - 1) {
          IsSplit = false;
          break;
        }
        Unmerge = Def;
      }

      if (!IsSplit || !Unmerge)
        continue;
      Register SplitSrc = Unmerge->getSourceReg();
      if (MRI->getType(SplitSrc) != MRI->getType(Dst))
        continue;

      for (const MachineOperand &Def : Unmerge->defs())
        Chain.push_back(Def.getReg());

      MRI->replaceRegWith(Dst, SplitSrc);
      MI.eraseFromParent();
      Changed = true;

      // Drop what the reassembly made dead. The pieces only become dead after
      // the copies that carried the readfirstlane result are gone, so iterate
      // over the chain until nothing else dies. Everything here was created by
      // the inline asm lowering for this operand, it is not a general DCE.
      bool Erased = true;
      while (Erased) {
        Erased = false;
        for (Register Reg : Chain) {
          MachineInstr *Def = MRI->getVRegDef(Reg);
          if (!Def || !Def->isDead(*MRI))
            continue;
          Def->eraseFromParent();
          Erased = true;
        }
      }
    }
  }

  return Changed;
}

static bool runDivergenceLowering(MachineFunction &MF, MachineDominatorTree &DT,
                                  MachinePostDominatorTree &PDT,
                                  MachineUniformityInfo &MUI) {
  DivergenceLoweringHelper Helper(MF, DT, PDT, &MUI);

  bool Changed = false;
  // Temporal divergence lowering needs to inspect list of instructions used
  // outside cycle with divergent exit provided by uniformity analysis. Uniform
  // instructions from the list require lowering, no instruction is deleted.
  // Thus it needs to be run before lowerPhis that deletes phis that require
  // lowering and replaces them with new instructions.

  // Non-i1 temporal divergence lowering.
  Changed |= Helper.lowerTemporalDivergence();
  // This covers both uniform and divergent i1s. Lane masks are in sgpr and need
  // to be updated in each iteration.
  Changed |= Helper.lowerTemporalDivergenceI1();
  // Temporal divergence lowering of divergent i1 phi used outside of the cycle
  // could also be handled by lowerPhis but we do it in lowerTempDivergenceI1
  // since in some case lowerPhis does unnecessary lane mask merging.
  Changed |= Helper.lowerPhis();
  Changed |= Helper.foldUniformReadFirstLane();
  Changed |= Helper.foldSplitAsmOperand();
  return Changed;
}

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPUGlobalISelDivergenceLoweringLegacy, DEBUG_TYPE,
                      "AMDGPU GlobalISel divergence lowering", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineUniformityAnalysisPass)
INITIALIZE_PASS_END(AMDGPUGlobalISelDivergenceLoweringLegacy, DEBUG_TYPE,
                    "AMDGPU GlobalISel divergence lowering", false, false)

char AMDGPUGlobalISelDivergenceLoweringLegacy::ID = 0;

char &llvm::AMDGPUGlobalISelDivergenceLoweringLegacyID =
    AMDGPUGlobalISelDivergenceLoweringLegacy::ID;

FunctionPass *llvm::createAMDGPUGlobalISelDivergenceLoweringPass() {
  return new AMDGPUGlobalISelDivergenceLoweringLegacy();
}

bool AMDGPUGlobalISelDivergenceLoweringLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  MachineDominatorTree &DT =
      getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MachinePostDominatorTree &PDT =
      getAnalysis<MachinePostDominatorTreeWrapperPass>().getPostDomTree();
  MachineUniformityInfo &MUI =
      getAnalysis<MachineUniformityAnalysisPass>().getUniformityInfo();

  return runDivergenceLowering(MF, DT, PDT, MUI);
}

PreservedAnalyses AMDGPUGlobalISelDivergenceLoweringPass::run(
    MachineFunction &MF, MachineFunctionAnalysisManager &MFAM) {
  MachineDominatorTree &DT = MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  MachinePostDominatorTree &PDT =
      MFAM.getResult<MachinePostDominatorTreeAnalysis>(MF);
  MachineUniformityInfo &MUI = MFAM.getResult<MachineUniformityAnalysis>(MF);

  if (!runDivergenceLowering(MF, DT, PDT, MUI))
    return PreservedAnalyses::all();

  return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
}
