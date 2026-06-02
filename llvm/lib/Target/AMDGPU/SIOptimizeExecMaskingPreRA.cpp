//===-- SIOptimizeExecMaskingPreRA.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass performs exec mask handling peephole optimizations which needs
/// to be done before register allocation to reduce register pressure.
///
//===----------------------------------------------------------------------===//

#include "SIOptimizeExecMaskingPreRA.h"
#include "AMDGPU.h"
#include "AMDGPULaneMaskUtils.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-optimize-exec-masking-pre-ra"

namespace {

class SIOptimizeExecMaskingPreRA {
private:
  const GCNSubtarget &ST;
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;
  const AMDGPU::LaneMaskConstants &LMC;

  MCRegister CondReg;
  MCRegister ExecReg;

  bool matchAndWithExec(MachineInstr &And, MachineOperand *&AndCC,
                        MachineOperand *&AndExec) const;
  SlotIndex replaceAndWithAndN2(MachineBasicBlock &MBB, MachineInstr &And,
                                const MachineOperand &Exec,
                                const MachineOperand &CC, MachineInstr *&Andn2,
                                bool IsDstDead, bool IsSCCDead);

  bool optimizeVcndVcmpPair(MachineBasicBlock &MBB);
  bool optimizeSccSelectBranch(MachineBasicBlock &MBB);
  bool optimizeElseBranch(MachineBasicBlock &MBB);

public:
  SIOptimizeExecMaskingPreRA(MachineFunction &MF, LiveIntervals *LIS)
      : ST(MF.getSubtarget<GCNSubtarget>()), TRI(ST.getRegisterInfo()),
        TII(ST.getInstrInfo()), MRI(&MF.getRegInfo()), LIS(LIS),
        LMC(AMDGPU::LaneMaskConstants::get(ST)) {}
  bool run(MachineFunction &MF);
};

class SIOptimizeExecMaskingPreRALegacy : public MachineFunctionPass {
public:
  static char ID;

  SIOptimizeExecMaskingPreRALegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI optimize exec mask operations pre-RA";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SIOptimizeExecMaskingPreRALegacy, DEBUG_TYPE,
                      "SI optimize exec mask operations pre-RA", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(SIOptimizeExecMaskingPreRALegacy, DEBUG_TYPE,
                    "SI optimize exec mask operations pre-RA", false, false)

char SIOptimizeExecMaskingPreRALegacy::ID = 0;

char &llvm::SIOptimizeExecMaskingPreRAID = SIOptimizeExecMaskingPreRALegacy::ID;

FunctionPass *llvm::createSIOptimizeExecMaskingPreRAPass() {
  return new SIOptimizeExecMaskingPreRALegacy();
}

// See if there is a def between \p AndIdx and \p SelIdx that needs to live
// beyond \p AndIdx.
static bool isDefBetween(const LiveRange &LR, SlotIndex AndIdx,
                         SlotIndex SelIdx) {
  LiveQueryResult AndLRQ = LR.Query(AndIdx);
  return (!AndLRQ.isKill() && AndLRQ.valueIn() != LR.Query(SelIdx).valueOut());
}

// FIXME: Why do we bother trying to handle physical registers here?
static bool isDefBetween(const SIRegisterInfo &TRI,
                         LiveIntervals *LIS, Register Reg,
                         const MachineInstr &Sel, const MachineInstr &And) {
  SlotIndex AndIdx = LIS->getInstructionIndex(And).getRegSlot();
  SlotIndex SelIdx = LIS->getInstructionIndex(Sel).getRegSlot();

  if (Reg.isVirtual())
    return isDefBetween(LIS->getInterval(Reg), AndIdx, SelIdx);

  for (MCRegUnit Unit : TRI.regunits(Reg.asMCReg())) {
    if (isDefBetween(LIS->getRegUnit(Unit), AndIdx, SelIdx))
      return true;
  }

  return false;
}

static bool isVccBranch(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  return Opc == AMDGPU::S_CBRANCH_VCCZ || Opc == AMDGPU::S_CBRANCH_VCCNZ;
}

static bool isSccBranch(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  return Opc == AMDGPU::S_CBRANCH_SCC0 || Opc == AMDGPU::S_CBRANCH_SCC1;
}

static bool matchRegImmOperands(MachineOperand *&RegOp, MachineOperand *&ImmOp,
                                int64_t Imm) {
  if (RegOp->isImm() && ImmOp->isReg())
    std::swap(RegOp, ImmOp);

  return RegOp->isReg() && ImmOp->isImm() && ImmOp->getImm() == Imm;
}

static bool matchImmOperands(const MachineOperand &Op0,
                             const MachineOperand &Op1, int64_t Imm0,
                             int64_t Imm1) {
  return Op0.isImm() && Op1.isImm() && Op0.getImm() == Imm0 &&
         Op1.getImm() == Imm1;
}

static bool isKillFromLiveRange(const SIRegisterInfo &TRI, LiveIntervals &LIS,
                                Register Reg, const MachineInstr &MI) {
  SlotIndex Idx = LIS.getInstructionIndex(MI).getRegSlot();

  if (Reg.isVirtual())
    return LIS.getInterval(Reg).Query(Idx).isKill();

  return llvm::all_of(TRI.regunits(Reg.asMCReg()), [&](MCRegUnit Unit) {
    return LIS.getRegUnit(Unit).Query(Idx).isKill();
  });
}

static void updateKillFlagFromLiveRange(const SIRegisterInfo &TRI,
                                        LiveIntervals &LIS, Register Reg,
                                        MachineInstr &MI) {
  int OpIdx = MI.findRegisterUseOperandIdx(Reg, &TRI);
  if (OpIdx == -1)
    return;

  MI.getOperand(OpIdx).setIsKill(isKillFromLiveRange(TRI, LIS, Reg, MI));
}

bool SIOptimizeExecMaskingPreRA::matchAndWithExec(
    MachineInstr &And, MachineOperand *&AndCC, MachineOperand *&AndExec) const {
  if (And.getOpcode() != LMC.AndOpc || !And.getOperand(1).isReg() ||
      !And.getOperand(2).isReg())
    return false;

  MachineOperand *Op0 = &And.getOperand(1);
  MachineOperand *Op1 = &And.getOperand(2);
  if (Op0->getReg() == Register(ExecReg)) {
    AndExec = Op0;
    AndCC = Op1;
  } else if (Op1->getReg() == Register(ExecReg)) {
    AndExec = Op1;
    AndCC = Op0;
  } else
    return false;

  return true;
}

SlotIndex SIOptimizeExecMaskingPreRA::replaceAndWithAndN2(
    MachineBasicBlock &MBB, MachineInstr &And, const MachineOperand &Exec,
    const MachineOperand &CC, MachineInstr *&Andn2, bool IsDstDead,
    bool IsSCCDead) {
  assert(And.getOperand(3).getReg() == AMDGPU::SCC);
  assert(Exec.getReg() == Register(ExecReg));

  MachineOperand Dst = And.getOperand(0);
  MachineOperand ExecOp = Exec;
  MachineOperand CCOp = CC;

  Dst.setIsDead(IsDstDead);

  Andn2 = BuildMI(MBB, And, And.getDebugLoc(), TII->get(LMC.AndN2Opc))
              .add(Dst)
              .add(ExecOp)
              .add(CCOp);

  MachineOperand &Andn2SCC = Andn2->getOperand(3);
  assert(Andn2SCC.getReg() == AMDGPU::SCC);
  Andn2SCC.setIsDead(IsSCCDead);

  SlotIndex AndIdx = LIS->ReplaceMachineInstrInMaps(And, *Andn2);
  And.eraseFromParent();
  return AndIdx;
}

// Optimize sequence
//    %sel = V_CNDMASK_B32_e64 0, 1, %cc
//    %cmp = V_CMP_NE_U32 1, %sel
//    $vcc = S_AND_B64 $exec, %cmp
//    S_CBRANCH_VCC[N]Z
// =>
//    $vcc = S_ANDN2_B64 $exec, %cc
//    S_CBRANCH_VCC[N]Z
//
// It is the negation pattern inserted by DAGCombiner::visitBRCOND() in the
// rebuildSetCC(). We start with S_CBRANCH to avoid exhaustive search, but
// only 3 first instructions are really needed. S_AND_B64 with exec is a
// required part of the pattern since V_CNDMASK_B32 writes zeroes for inactive
// lanes.
//
// Returns true on success.
bool SIOptimizeExecMaskingPreRA::optimizeVcndVcmpPair(MachineBasicBlock &MBB) {
  MachineBasicBlock::iterator I = llvm::find_if(MBB.terminators(), isVccBranch);
  if (I == MBB.terminators().end())
    return false;

  MachineInstr *And =
      TRI->findReachingDef(CondReg, AMDGPU::NoSubRegister, *I, *MRI, LIS);
  MachineOperand *AndCmp = nullptr;
  MachineOperand *AndExec = nullptr;
  if (!And || !matchAndWithExec(*And, AndCmp, AndExec))
    return false;

  Register CmpReg = AndCmp->getReg();
  unsigned CmpSubReg = AndCmp->getSubReg();

  MachineInstr *Cmp = TRI->findReachingDef(CmpReg, CmpSubReg, *And, *MRI, LIS);
  if (!Cmp || !(Cmp->getOpcode() == AMDGPU::V_CMP_NE_U32_e32 ||
                Cmp->getOpcode() == AMDGPU::V_CMP_NE_U32_e64) ||
      Cmp->getParent() != And->getParent())
    return false;

  MachineOperand *Op1 = TII->getNamedOperand(*Cmp, AMDGPU::OpName::src0);
  MachineOperand *Op2 = TII->getNamedOperand(*Cmp, AMDGPU::OpName::src1);
  if (!matchRegImmOperands(Op1, Op2, 1))
    return false;

  Register SelReg = Op1->getReg();
  if (SelReg.isPhysical())
    return false;

  auto *Sel = TRI->findReachingDef(SelReg, Op1->getSubReg(), *Cmp, *MRI, LIS);
  if (!Sel || Sel->getOpcode() != AMDGPU::V_CNDMASK_B32_e64)
    return false;

  if (TII->hasModifiersSet(*Sel, AMDGPU::OpName::src0_modifiers) ||
      TII->hasModifiersSet(*Sel, AMDGPU::OpName::src1_modifiers))
    return false;

  Op1 = TII->getNamedOperand(*Sel, AMDGPU::OpName::src0);
  Op2 = TII->getNamedOperand(*Sel, AMDGPU::OpName::src1);
  MachineOperand *CC = TII->getNamedOperand(*Sel, AMDGPU::OpName::src2);
  if (!matchImmOperands(*Op1, *Op2, 0, 1) || !CC->isReg())
    return false;

  Register CCReg = CC->getReg();

  // If there was a def between the select and the and, we would need to move it
  // to fold this.
  if (isDefBetween(*TRI, LIS, CCReg, *Sel, *And))
    return false;

  // Cannot safely mirror live intervals with PHI nodes, so check for these
  // before optimization.
  SlotIndex SelIdx = LIS->getInstructionIndex(*Sel);
  LiveInterval *SelLI = &LIS->getInterval(SelReg);
  if (llvm::any_of(SelLI->vnis(),
                    [](const VNInfo *VNI) {
                      return VNI->isPHIDef();
                    }))
    return false;

  // TODO: Guard against implicit def operands?
  LLVM_DEBUG(dbgs() << "Folding sequence:\n\t" << *Sel << '\t' << *Cmp << '\t'
                    << *And);

  MachineOperand CCOp = *CC;
  if (CC->isKill())
    CC->setIsKill(false);

  MachineInstr *Andn2 = nullptr;
  SlotIndex AndIdx =
      replaceAndWithAndN2(MBB, *And, *AndExec, CCOp, Andn2, /*IsDstDead=*/false,
                          /*IsSCCDead=*/And->getOperand(3).isDead());

  LLVM_DEBUG(dbgs() << "=>\n\t" << *Andn2 << '\n');

  // Update live intervals for CCReg before potentially removing CmpReg/SelReg,
  // and their associated liveness information.
  SlotIndex CmpIdx = LIS->getInstructionIndex(*Cmp);
  if (CCReg.isVirtual()) {
    LiveInterval &CCLI = LIS->getInterval(CCReg);
    auto CCQ = CCLI.Query(SelIdx.getRegSlot());
    if (CCQ.valueIn()) {
      LIS->removeInterval(CCReg);
      LIS->createAndComputeVirtRegInterval(CCReg);
    }
  } else
    LIS->removeAllRegUnitsForPhysReg(CCReg);

  // Try to remove compare. Cmp value should not used in between of cmp
  // and s_and_b64 if VCC or just unused if any other register.
  LiveInterval *CmpLI = CmpReg.isVirtual() ? &LIS->getInterval(CmpReg) : nullptr;
  if ((CmpLI && CmpLI->Query(AndIdx.getRegSlot()).isKill()) ||
      (CmpReg == Register(CondReg) &&
       std::none_of(std::next(Cmp->getIterator()), Andn2->getIterator(),
                    [&](const MachineInstr &MI) {
                      return MI.readsRegister(CondReg, TRI);
                    }))) {
    LLVM_DEBUG(dbgs() << "Erasing: " << *Cmp << '\n');
    if (CmpLI)
      LIS->removeVRegDefAt(*CmpLI, CmpIdx.getRegSlot());
    LIS->RemoveMachineInstrFromMaps(*Cmp);
    Cmp->eraseFromParent();

    // Try to remove v_cndmask_b32.
    // Kill status must be checked before shrinking the live range.
    bool IsKill = SelLI->Query(CmpIdx.getRegSlot()).isKill();
    LIS->shrinkToUses(SelLI);
    bool IsDead = SelLI->Query(SelIdx.getRegSlot()).isDeadDef();
    if (MRI->use_nodbg_empty(SelReg) && (IsKill || IsDead)) {
      LLVM_DEBUG(dbgs() << "Erasing: " << *Sel << '\n');

      LIS->removeVRegDefAt(*SelLI, SelIdx.getRegSlot());
      LIS->RemoveMachineInstrFromMaps(*Sel);
      bool ShrinkSel = Sel->getOperand(0).readsReg();
      Sel->eraseFromParent();
      if (ShrinkSel) {
        // The result of the V_CNDMASK was a subreg def which counted as a read
        // from the other parts of the reg. Shrink their live ranges.
        LIS->shrinkToUses(SelLI);
      }
    }
  }

  return true;
}

// Optimize sequence
//    %cc = S_CSELECT -1, 0, %uniformcc
//    dead %and = S_AND %cc, $exec
//    %bool = S_CSELECT_B32 1, 0
//    S_CMP_LG_U32 %bool, 1
//    S_CBRANCH_SCC1
// =>
//    dead %and = S_ANDN2 $exec, %cc
//    S_CBRANCH_SCC1
//
// This is the scalar form of the negated uniform branch pattern handled by
// optimizeVcndVcmpPair().
bool SIOptimizeExecMaskingPreRA::optimizeSccSelectBranch(
    MachineBasicBlock &MBB) {
  MachineBasicBlock::iterator I = llvm::find_if(MBB.terminators(), isSccBranch);
  if (I == MBB.terminators().end())
    return false;

  MachineInstr *Cmp =
      TRI->findReachingDef(AMDGPU::SCC, AMDGPU::NoSubRegister, *I, *MRI, LIS);
  if (!Cmp || Cmp->getOpcode() != AMDGPU::S_CMP_LG_U32 ||
      Cmp->getParent() != I->getParent())
    return false;

  MachineOperand *CmpBool = &Cmp->getOperand(0);
  MachineOperand *CmpOne = &Cmp->getOperand(1);
  if (!matchRegImmOperands(CmpBool, CmpOne, 1))
    return false;

  Register BoolReg = CmpBool->getReg();
  if (BoolReg.isPhysical() || CmpBool->getSubReg() != AMDGPU::NoSubRegister ||
      !MRI->hasOneNonDBGUse(BoolReg) ||
      &*MRI->use_instr_nodbg_begin(BoolReg) != Cmp)
    return false;

  MachineInstr *BoolSel =
      TRI->findReachingDef(BoolReg, AMDGPU::NoSubRegister, *Cmp, *MRI, LIS);
  if (!BoolSel || BoolSel->getOpcode() != AMDGPU::S_CSELECT_B32 ||
      BoolSel->getParent() != Cmp->getParent() ||
      !matchImmOperands(BoolSel->getOperand(1), BoolSel->getOperand(2), 1, 0))
    return false;

  MachineInstr *And = TRI->findReachingDef(AMDGPU::SCC, AMDGPU::NoSubRegister,
                                           *BoolSel, *MRI, LIS);
  MachineOperand *AndCC = nullptr;
  MachineOperand *AndExec = nullptr;
  if (!And || And->getParent() != BoolSel->getParent() ||
      !matchAndWithExec(*And, AndCC, AndExec))
    return false;

  Register AndDst = And->getOperand(0).getReg();
  if (AndDst.isPhysical() || !MRI->use_nodbg_empty(AndDst))
    return false;

  Register CCReg = AndCC->getReg();
  if (CCReg.isPhysical())
    return false;

  MachineInstr *CCSel =
      TRI->findReachingDef(CCReg, AndCC->getSubReg(), *And, *MRI, LIS);
  if (!CCSel || CCSel->getOpcode() != LMC.CSelectOpc ||
      CCSel->getParent() != And->getParent() ||
      !matchImmOperands(CCSel->getOperand(1), CCSel->getOperand(2), -1, 0))
    return false;

  LLVM_DEBUG(dbgs() << "Folding scalar SCC branch sequence:\n\t" << *And << '\t'
                    << *BoolSel << '\t' << *Cmp);

  MachineInstr *Andn2 = nullptr;
  replaceAndWithAndN2(MBB, *And, *AndExec, *AndCC, Andn2,
                      /*IsDstDead=*/true, /*IsSCCDead=*/false);

  SlotIndex BoolSelIdx = LIS->getInstructionIndex(*BoolSel);
  LiveInterval &BoolLI = LIS->getInterval(BoolReg);
  LIS->RemoveMachineInstrFromMaps(*Cmp);
  Cmp->eraseFromParent();

  LIS->removeVRegDefAt(BoolLI, BoolSelIdx.getRegSlot());
  LIS->RemoveMachineInstrFromMaps(*BoolSel);
  BoolSel->eraseFromParent();
  if (MRI->reg_nodbg_empty(BoolReg))
    LIS->removeInterval(BoolReg);

  LIS->removeAllRegUnitsForPhysReg(AMDGPU::SCC);
  updateKillFlagFromLiveRange(*TRI, *LIS, AMDGPU::SCC, *I);

  LLVM_DEBUG(dbgs() << "=>\n\t" << *Andn2 << '\n');
  return true;
}

// Optimize sequence
//    %dst = S_OR_SAVEEXEC %src
//    ... instructions not modifying exec ...
//    %tmp = S_AND $exec, %dst
//    $exec = S_XOR_term $exec, %tmp
// =>
//    %dst = S_OR_SAVEEXEC %src
//    ... instructions not modifying exec ...
//    $exec = S_XOR_term $exec, %dst
//
// Clean up potentially unnecessary code added for safety during
// control flow lowering.
//
// Return whether any changes were made to MBB.
bool SIOptimizeExecMaskingPreRA::optimizeElseBranch(MachineBasicBlock &MBB) {
  if (MBB.empty())
    return false;

  // Check this is an else block.
  auto First = MBB.begin();
  MachineInstr &SaveExecMI = *First;
  if (SaveExecMI.getOpcode() != LMC.OrSaveExecOpc)
    return false;

  auto I = llvm::find_if(MBB.terminators(), [this](const MachineInstr &MI) {
    return MI.getOpcode() == LMC.XorTermOpc;
  });
  if (I == MBB.terminators().end())
    return false;

  MachineInstr &XorTermMI = *I;
  if (XorTermMI.getOperand(1).getReg() != Register(ExecReg))
    return false;

  Register SavedExecReg = SaveExecMI.getOperand(0).getReg();
  Register DstReg = XorTermMI.getOperand(2).getReg();

  // Find potentially unnecessary S_AND
  MachineInstr *AndExecMI = nullptr;
  I--;
  while (I != First && !AndExecMI) {
    if (I->getOpcode() == LMC.AndOpc && I->getOperand(0).getReg() == DstReg &&
        I->getOperand(1).getReg() == Register(ExecReg))
      AndExecMI = &*I;
    I--;
  }
  if (!AndExecMI)
    return false;

  // Check for exec modifying instructions.
  // Note: exec defs do not create live ranges beyond the
  // instruction so isDefBetween cannot be used.
  // Instead just check that the def segments are adjacent.
  SlotIndex StartIdx = LIS->getInstructionIndex(SaveExecMI);
  SlotIndex EndIdx = LIS->getInstructionIndex(*AndExecMI);
  for (MCRegUnit Unit : TRI->regunits(ExecReg)) {
    LiveRange &RegUnit = LIS->getRegUnit(Unit);
    if (RegUnit.find(StartIdx) != std::prev(RegUnit.find(EndIdx)))
      return false;
  }

  // Remove unnecessary S_AND
  LIS->removeInterval(SavedExecReg);
  LIS->removeInterval(DstReg);

  SaveExecMI.getOperand(0).setReg(DstReg);

  LIS->RemoveMachineInstrFromMaps(*AndExecMI);
  AndExecMI->eraseFromParent();

  LIS->createAndComputeVirtRegInterval(DstReg);

  return true;
}

PreservedAnalyses
SIOptimizeExecMaskingPreRAPass::run(MachineFunction &MF,
                                    MachineFunctionAnalysisManager &MFAM) {
  auto &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  SIOptimizeExecMaskingPreRA(MF, &LIS).run(MF);
  return PreservedAnalyses::all();
}

bool SIOptimizeExecMaskingPreRALegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  auto *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  return SIOptimizeExecMaskingPreRA(MF, LIS).run(MF);
}

bool SIOptimizeExecMaskingPreRA::run(MachineFunction &MF) {
  CondReg = MCRegister::from(LMC.VccReg);
  ExecReg = MCRegister::from(LMC.ExecReg);

  DenseSet<Register> RecalcRegs({AMDGPU::EXEC_LO, AMDGPU::EXEC_HI});
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {

    if (optimizeElseBranch(MBB)) {
      RecalcRegs.insert(AMDGPU::SCC);
      Changed = true;
    }

    if (optimizeVcndVcmpPair(MBB)) {
      RecalcRegs.insert(AMDGPU::VCC_LO);
      RecalcRegs.insert(AMDGPU::VCC_HI);
      RecalcRegs.insert(AMDGPU::SCC);
      Changed = true;
    }

    if (optimizeSccSelectBranch(MBB)) {
      RecalcRegs.insert(AMDGPU::SCC);
      Changed = true;
    }

    // Try to remove unneeded instructions before s_endpgm.
    if (MBB.succ_empty()) {
      if (MBB.empty())
        continue;

      // Skip this if the endpgm has any implicit uses, otherwise we would need
      // to be careful to update / remove them.
      // S_ENDPGM always has a single imm operand that is not used other than to
      // end up in the encoding
      MachineInstr &Term = MBB.back();
      if (Term.getOpcode() != AMDGPU::S_ENDPGM || Term.getNumOperands() != 1)
        continue;

      SmallVector<MachineBasicBlock*, 4> Blocks({&MBB});

      while (!Blocks.empty()) {
        auto *CurBB = Blocks.pop_back_val();
        auto I = CurBB->rbegin(), E = CurBB->rend();
        if (I != E) {
          if (I->isUnconditionalBranch() || I->getOpcode() == AMDGPU::S_ENDPGM)
            ++I;
          else if (I->isBranch())
            continue;
        }

        while (I != E) {
          if (I->isDebugInstr()) {
            I = std::next(I);
            continue;
          }

          if (I->mayStore() || I->isBarrier() || I->isCall() ||
              I->hasUnmodeledSideEffects() || I->hasOrderedMemoryRef())
            break;

          LLVM_DEBUG(dbgs()
                     << "Removing no effect instruction: " << *I << '\n');

          for (auto &Op : I->operands()) {
            if (Op.isReg())
              RecalcRegs.insert(Op.getReg());
          }

          auto Next = std::next(I);
          LIS->RemoveMachineInstrFromMaps(*I);
          I->eraseFromParent();
          I = Next;

          Changed = true;
        }

        if (I != E)
          continue;

        // Try to ascend predecessors.
        for (auto *Pred : CurBB->predecessors()) {
          if (Pred->succ_size() == 1)
            Blocks.push_back(Pred);
        }
      }
      continue;
    }

    // If the only user of a logical operation is move to exec, fold it now
    // to prevent forming of saveexec. I.e.:
    //
    //    %0:sreg_64 = COPY $exec
    //    %1:sreg_64 = S_AND_B64 %0:sreg_64, %2:sreg_64
    // =>
    //    %1 = S_AND_B64 $exec, %2:sreg_64
    unsigned ScanThreshold = 10;
    for (auto I = MBB.rbegin(), E = MBB.rend(); I != E
         && ScanThreshold--; ++I) {
      // Continue scanning if this is not a full exec copy
      if (!(I->isFullCopy() && I->getOperand(1).getReg() == Register(ExecReg)))
        continue;

      Register SavedExec = I->getOperand(0).getReg();
      if (SavedExec.isVirtual() && MRI->hasOneNonDBGUse(SavedExec)) {
        MachineInstr *SingleExecUser = &*MRI->use_instr_nodbg_begin(SavedExec);
        int Idx = SingleExecUser->findRegisterUseOperandIdx(SavedExec,
                                                            /*TRI=*/nullptr);
        assert(Idx != -1);
        if (SingleExecUser->getParent() == I->getParent() &&
            !SingleExecUser->getOperand(Idx).isImplicit() &&
            static_cast<unsigned>(Idx) <
                SingleExecUser->getDesc().getNumOperands() &&
            TII->isOperandLegal(*SingleExecUser, Idx, &I->getOperand(1))) {
          LLVM_DEBUG(dbgs() << "Redundant EXEC COPY: " << *I << '\n');
          LIS->RemoveMachineInstrFromMaps(*I);
          I->eraseFromParent();
          MRI->replaceRegWith(SavedExec, ExecReg);
          LIS->removeInterval(SavedExec);
          Changed = true;
        }
      }
      break;
    }
  }

  if (Changed) {
    for (auto Reg : RecalcRegs) {
      if (Reg.isVirtual()) {
        LIS->removeInterval(Reg);
        if (!MRI->reg_empty(Reg))
          LIS->createAndComputeVirtRegInterval(Reg);
      } else {
        LIS->removeAllRegUnitsForPhysReg(Reg);
      }
    }
  }

  return Changed;
}
