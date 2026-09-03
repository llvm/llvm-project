//===- RISCVPostRAV0Rewrite.cpp - Rewrite post-RA V0 copies --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// After register allocation, remove two narrowly proven classes of copies to
// V0.  The first retargets a safe adjacent mask producer directly to V0:
//
//   D = safe mask-producing instruction
//   V0 = COPY killed D
//
// The second rewrites either form of an adjacent masked-compare chain:
//
//   D = COPY V0                 or   V0 = COPY D
//   D = masked RVV compare D(tied), ..., V0(mask)
//   V0 = COPY D
//
// Every erased COPY may carry only shared, implicit, side-effect-free VL or
// VTYPE uses.  LivePhysRegs proves D and all its aliases dead after the final
// copy, and a whole-function debug scan prevents stale physical-register debug
// operands.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-post-ra-v0-rewrite"
#define RISCV_POST_RA_V0_REWRITE_NAME "RISC-V Post-RA V0 Rewrite"

STATISTIC(NumV0ProducersRetargeted,
          "Number of post-RA mask producers retargeted to V0");
STATISTIC(NumV0CompareChainsRewritten,
          "Number of post-RA masked-compare V0 chains rewritten");

namespace {

class RISCVPostRAV0RewriteImpl {
  const TargetInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const MachineRegisterInfo *MRI = nullptr;
  BitVector DebugReferencedRegs;

  bool tryRetargetProducer(MachineBasicBlock::instr_iterator Producer,
                           MachineBasicBlock::instr_iterator Copy,
                           const LivePhysRegs &LiveAfterCopy) const;
  bool tryRewriteCompareChain(MachineBasicBlock::instr_iterator Save,
                              MachineBasicBlock::instr_iterator Body,
                              MachineBasicBlock::instr_iterator Restore,
                              const LivePhysRegs &LiveAfterRestore) const;
  bool rewriteBlock(MachineBasicBlock &MBB) const;

public:
  bool run(MachineFunction &MF);
};

class RISCVPostRAV0RewriteLegacy : public MachineFunctionPass {
public:
  static char ID;

  RISCVPostRAV0RewriteLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;
    return RISCVPostRAV0RewriteImpl().run(MF);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs().setTracksLiveness();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return RISCV_POST_RA_V0_REWRITE_NAME;
  }
};

} // end anonymous namespace

char RISCVPostRAV0RewriteLegacy::ID = 0;

INITIALIZE_PASS(RISCVPostRAV0RewriteLegacy, DEBUG_TYPE,
                RISCV_POST_RA_V0_REWRITE_NAME, false, false)

static bool isStructurallyPlainCopy(const MachineInstr &MI) {
  if (!MI.isFullCopy() || MI.getFlag(MachineInstr::LRSplit) || MI.isBundled() ||
      MI.peekDebugInstrNum() || MI.getFlags() != MachineInstr::NoFlags ||
      MI.getAsmPrinterFlags() || MI.getNumExplicitOperands() != 2 ||
      !MI.memoperands_empty() || MI.getPreInstrSymbol() ||
      MI.getPostInstrSymbol() || MI.getHeapAllocMarker() ||
      MI.getPCSections() || MI.getMMRAMetadata() || MI.getCFIType() ||
      MI.getDeactivationSymbol() || MI.isCall() || MI.isInlineAsm() ||
      MI.isTerminator() || MI.mayLoadOrStore() || MI.hasUnmodeledSideEffects())
    return false;

  const MachineOperand &Dst = MI.getOperand(0);
  const MachineOperand &Src = MI.getOperand(1);
  return Dst.isReg() && Dst.isDef() && !Dst.readsReg() &&
         !Dst.isInternalRead() && !Dst.isImplicit() && Dst.getReg() &&
         !Dst.getSubReg() && !Dst.isEarlyClobber() && !Dst.getTargetFlags() &&
         Src.isReg() && Src.readsReg() && !Src.isUndef() && !Src.isImplicit() &&
         !Src.isInternalRead() && Src.getReg() && !Src.getSubReg() &&
         !Src.getTargetFlags();
}

static bool isPlainFullCopy(const MachineInstr &MI, Register Dst, Register Src,
                            bool RequireKilledSrc = false) {
  return isStructurallyPlainCopy(MI) && MI.getOperand(0).getReg() == Dst &&
         MI.getOperand(1).getReg() == Src &&
         (!RequireKilledSrc || MI.getOperand(1).isKill());
}

static bool hasImplicitUse(const MachineInstr &MI, Register Reg) {
  for (const MachineOperand &MO : MI.operands())
    if (MO.isReg() && MO.isImplicit() && MO.isUse() && MO.readsReg() &&
        !MO.isUndef() && MO.getReg() == Reg && !MO.getSubReg())
      return true;
  return false;
}

static bool copyHasOnlySharedImplicitExtras(const MachineInstr &Copy,
                                            const MachineInstr &Producer) {
  bool SeenVL = false;
  bool SeenVType = false;
  for (unsigned I = 2, E = Copy.getNumOperands(); I != E; ++I) {
    const MachineOperand &MO = Copy.getOperand(I);
    if (!MO.isReg() || !MO.isImplicit() || !MO.isUse() || !MO.readsReg() ||
        MO.isUndef() || MO.isKill() || MO.isInternalRead() ||
        MO.isEarlyClobber() || MO.getSubReg() || MO.getTargetFlags())
      return false;

    if (MO.getReg() == RISCV::VL) {
      if (SeenVL || !hasImplicitUse(Producer, RISCV::VL))
        return false;
      SeenVL = true;
      continue;
    }
    if (MO.getReg() == RISCV::VTYPE) {
      if (SeenVType || !hasImplicitUse(Producer, RISCV::VTYPE))
        return false;
      SeenVType = true;
      continue;
    }
    return false;
  }
  return true;
}

static bool isSafeProducer(const MachineInstr &MI, unsigned &TiedPassthruIdx) {
  TiedPassthruIdx = ~0U;
  unsigned MCOpcode = RISCV::getRVVMCOpcode(MI.getOpcode());
  if (MCOpcode == RISCV::VMAND_MM)
    return true;

  // Comparisons have an untied mask destination.  Exclude masked variants:
  // they explicitly read V0 and may impose no-V0 destination constraints.
  if (RISCVInstrInfo::isRVVCompare(MI) &&
      !RISCV::getMaskedPseudoInfo(MI.getOpcode()))
    return true;

  bool IsUndefPassthruProducer =
      MCOpcode == RISCV::VMV_V_I || MCOpcode == RISCV::VMV_S_X ||
      MCOpcode == RISCV::VLM_V || MCOpcode == RISCV::VSLIDEDOWN_VI ||
      MCOpcode == RISCV::VSLIDEDOWN_VX || MCOpcode == RISCV::VLE64_V;
  if (!IsUndefPassthruProducer ||
      !MI.isRegTiedToUseOperand(/*DefOpIdx=*/0, &TiedPassthruIdx))
    return false;

  const MachineOperand &Passthru = MI.getOperand(TiedPassthruIdx);
  return Passthru.isReg() && Passthru.isUse() && Passthru.isUndef() &&
         !Passthru.isImplicit() && !Passthru.getSubReg();
}

bool RISCVPostRAV0RewriteImpl::tryRetargetProducer(
    MachineBasicBlock::instr_iterator Producer,
    MachineBasicBlock::instr_iterator Copy,
    const LivePhysRegs &LiveAfterCopy) const {
  MachineInstr &ProducerMI = *Producer;
  MachineInstr &CopyMI = *Copy;
  if (ProducerMI.isBundled() || ProducerMI.peekDebugInstrNum() ||
      ProducerMI.getFlag(MachineInstr::LRSplit) ||
      ProducerMI.getNumExplicitDefs() != 1 || !isStructurallyPlainCopy(CopyMI))
    return false;

  MachineOperand &Def = ProducerMI.getOperand(0);
  MachineOperand &CopySrc = CopyMI.getOperand(1);
  bool IsUnmaskedCompare = RISCVInstrInfo::isRVVCompare(ProducerMI) &&
                           !RISCV::getMaskedPseudoInfo(ProducerMI.getOpcode());
  if (!Def.isReg() || !Def.isDef() || Def.isImplicit() || Def.readsReg() ||
      Def.isInternalRead() || (Def.isEarlyClobber() && !IsUnmaskedCompare) ||
      Def.getSubReg() || Def.getTargetFlags() || !Def.getReg().isPhysical() ||
      Def.getReg() == RISCV::V0 || Def.getReg() != CopySrc.getReg() ||
      CopyMI.getOperand(0).getReg() != RISCV::V0 ||
      !RISCV::VRRegClass.contains(Def.getReg()))
    return false;
  Register Dst = Def.getReg();
  if (DebugReferencedRegs.test(Dst.id()) ||
      DebugReferencedRegs.test(RISCV::V0) ||
      !copyHasOnlySharedImplicitExtras(CopyMI, ProducerMI))
    return false;

  unsigned TiedPassthruIdx;
  if (!isSafeProducer(ProducerMI, TiedPassthruIdx))
    return false;
  if (TiedPassthruIdx != ~0U &&
      ProducerMI.getOperand(TiedPassthruIdx).getReg() != Dst)
    return false;

  if (ProducerMI.isMetaInstruction() || ProducerMI.isPosition() ||
      ProducerMI.isBarrier() || ProducerMI.mayStore())
    return false;

  // The destination constraint must admit the physical mask register.
  const TargetRegisterClass *DefRC =
      ProducerMI.getRegClassConstraint(0, TII, TRI);
  if (!DefRC || !DefRC->contains(RISCV::V0))
    return false;
  if (TiedPassthruIdx != ~0U) {
    const TargetRegisterClass *PassthruRC =
        ProducerMI.getRegClassConstraint(TiedPassthruIdx, TII, TRI);
    if (!PassthruRC || !PassthruRC->contains(RISCV::V0))
      return false;
  }

  if (!LiveAfterCopy.available(*MRI, Dst))
    return false;

  MachineOperand *TiedPassthru = TiedPassthruIdx == ~0U
                                     ? nullptr
                                     : &ProducerMI.getOperand(TiedPassthruIdx);
  bool IsVMAND =
      RISCV::getRVVMCOpcode(ProducerMI.getOpcode()) == RISCV::VMAND_MM;
  unsigned NumExplicitVMANDSources = 0;
  for (MachineOperand &MO : ProducerMI.operands()) {
    if (MO.isRegMask())
      return false;
    if (!MO.isReg() || !MO.getReg())
      continue;
    if (&MO == &Def || &MO == TiedPassthru)
      continue;
    if (IsVMAND && !MO.isImplicit() && MO.isUse())
      ++NumExplicitVMANDSources;
    if (TRI->regsOverlap(MO.getReg(), RISCV::V0))
      return false;
    if (!TRI->regsOverlap(MO.getReg(), Dst))
      continue;

    const TargetRegisterClass *SrcRC =
        ProducerMI.getRegClassConstraint(MO.getOperandNo(), TII, TRI);
    if (!IsVMAND || (MO.getOperandNo() != 1 && MO.getOperandNo() != 2) ||
        !MO.isUse() || !MO.readsReg() || MO.isUndef() || !MO.isKill() ||
        MO.isImplicit() || MO.isInternalRead() || MO.isEarlyClobber() ||
        MO.getSubReg() || MO.getTargetFlags() || MO.getReg() != Dst || !SrcRC ||
        !SrcRC->contains(Dst))
      return false;
  }
  if (IsVMAND && NumExplicitVMANDSources != 2)
    return false;

  LLVM_DEBUG(dbgs() << "Retargeting post-RA producer to V0:\n  " << ProducerMI
                    << "  " << CopyMI);

  Def.setReg(RISCV::V0);
  Def.setIsRenamable(false);
  if (TiedPassthru) {
    TiedPassthru->setReg(RISCV::V0);
    TiedPassthru->setIsRenamable(false);
  }
  CopyMI.eraseFromParent();
  ++NumV0ProducersRetargeted;
  return true;
}

bool RISCVPostRAV0RewriteImpl::tryRewriteCompareChain(
    MachineBasicBlock::instr_iterator Save,
    MachineBasicBlock::instr_iterator Body,
    MachineBasicBlock::instr_iterator Restore,
    const LivePhysRegs &LiveAfterRestore) const {
  MachineInstr &SaveMI = *Save;
  MachineInstr &BodyMI = *Body;
  MachineInstr &RestoreMI = *Restore;
  if (!isStructurallyPlainCopy(SaveMI) || !isStructurallyPlainCopy(RestoreMI))
    return false;

  MachineOperand &SaveDst = SaveMI.getOperand(0);
  MachineOperand &SaveSrc = SaveMI.getOperand(1);
  bool SaveFromV0 = SaveSrc.getReg() == RISCV::V0;
  bool SaveToV0 = SaveDst.getReg() == RISCV::V0;
  if (SaveFromV0 == SaveToV0)
    return false;
  Register Dst = SaveFromV0 ? SaveDst.getReg() : SaveSrc.getReg();
  if (!Dst.isPhysical() || Dst == RISCV::V0 ||
      !RISCV::VRRegClass.contains(Dst) ||
      !isPlainFullCopy(RestoreMI, RISCV::V0, Dst))
    return false;

  if (DebugReferencedRegs.test(Dst.id()) || DebugReferencedRegs.test(RISCV::V0))
    return false;

  const RISCV::RISCVMaskedPseudoInfo *MaskedInfo =
      RISCV::getMaskedPseudoInfo(BodyMI.getOpcode());
  if (!MaskedInfo || !RISCVInstrInfo::isRVVCompare(BodyMI) ||
      BodyMI.isBundled() || BodyMI.peekDebugInstrNum() ||
      BodyMI.getFlag(MachineInstr::LRSplit) || BodyMI.getNumExplicitDefs() != 1)
    return false;

  unsigned PassthruIdx;
  if (!BodyMI.isRegTiedToUseOperand(/*DefOpIdx=*/0, &PassthruIdx))
    return false;
  unsigned MaskIdx = MaskedInfo->MaskOpIdx + BodyMI.getNumExplicitDefs();
  if (PassthruIdx >= BodyMI.getNumExplicitOperands() ||
      MaskIdx >= BodyMI.getNumExplicitOperands() || PassthruIdx == MaskIdx)
    return false;

  MachineOperand &Def = BodyMI.getOperand(0);
  MachineOperand &Passthru = BodyMI.getOperand(PassthruIdx);
  MachineOperand &Mask = BodyMI.getOperand(MaskIdx);
  if (!Def.isReg() || !Def.isDef() || Def.isImplicit() ||
      Def.isEarlyClobber() || Def.getReg() != Dst || Def.getSubReg() ||
      !Passthru.isReg() || !Passthru.readsReg() || Passthru.isUndef() ||
      Passthru.isImplicit() || Passthru.getReg() != Dst ||
      Passthru.getSubReg() || !Mask.isReg() || !Mask.readsReg() ||
      Mask.isUndef() || Mask.isImplicit() || Mask.getReg() != RISCV::V0 ||
      Mask.getSubReg())
    return false;

  auto [LMul, Fractional] =
      RISCVVType::decodeVLMUL(RISCVII::getLMul(BodyMI.getDesc().TSFlags));
  if (!Fractional && LMul != 1)
    return false;

  const TargetRegisterClass *DefRC = BodyMI.getRegClassConstraint(0, TII, TRI);
  const TargetRegisterClass *PassthruRC =
      BodyMI.getRegClassConstraint(PassthruIdx, TII, TRI);
  if (!DefRC || !PassthruRC || !DefRC->contains(RISCV::V0) ||
      !PassthruRC->contains(RISCV::V0))
    return false;

  if (!copyHasOnlySharedImplicitExtras(SaveMI, BodyMI) ||
      !copyHasOnlySharedImplicitExtras(RestoreMI, BodyMI) ||
      !LiveAfterRestore.available(*MRI, Dst))
    return false;

  for (const MachineOperand &MO : BodyMI.operands()) {
    if (MO.isRegMask())
      return false;
    if (!MO.isReg() || !MO.getReg())
      continue;
    if (!TRI->regsOverlap(MO.getReg(), Dst) &&
        !TRI->regsOverlap(MO.getReg(), RISCV::V0))
      continue;
    if (&MO != &Def && &MO != &Passthru && &MO != &Mask)
      return false;
  }

  LLVM_DEBUG(dbgs() << "Rewriting post-RA V0 compare chain:\n  " << SaveMI
                    << "  " << BodyMI << "  " << RestoreMI);

  Def.setReg(RISCV::V0);
  Def.setIsRenamable(false);
  Passthru.setReg(RISCV::V0);
  Passthru.setIsRenamable(false);
  Mask.setIsRenamable(false);

  if (SaveFromV0)
    SaveMI.eraseFromParent();
  RestoreMI.eraseFromParent();
  ++NumV0CompareChainsRewritten;
  return true;
}

bool RISCVPostRAV0RewriteImpl::rewriteBlock(MachineBasicBlock &MBB) const {
  bool Changed = false;
  bool LocalChange;
  do {
    LocalChange = false;
    LivePhysRegs Live(*TRI);
    Live.addLiveOuts(MBB);

    // Restart after every rewrite so erased copies cannot invalidate the
    // reverse scan and newly adjacent candidates are reconsidered.
    for (auto I = MBB.instr_rbegin(), E = MBB.instr_rend(); I != E; ++I) {
      MachineInstr &MI = *I;
      if (MI.isDebugInstr())
        continue;
      if (MI.isBundledWithPred())
        continue;

      if (isStructurallyPlainCopy(MI) &&
          MI.getOperand(0).getReg() == RISCV::V0) {
        MachineBasicBlock::instr_iterator Restore(MI);
        if (Restore != MBB.instr_begin()) {
          auto Previous = prev_nodbg(Restore, MBB.instr_begin(),
                                     /*SkipPseudoOp=*/false);
          if (tryRetargetProducer(Previous, Restore, Live)) {
            LocalChange = true;
            Changed = true;
            break;
          }

          if (Previous != MBB.instr_begin()) {
            auto Save = prev_nodbg(Previous, MBB.instr_begin(),
                                   /*SkipPseudoOp=*/false);
            if (tryRewriteCompareChain(Save, Previous, Restore, Live)) {
              LocalChange = true;
              Changed = true;
              break;
            }
          }
        }
      }

      Live.stepBackward(MI);
    }
  } while (LocalChange);

  return Changed;
}

bool RISCVPostRAV0RewriteImpl::run(MachineFunction &MF) {
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasVInstructions())
    return false;

  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  MRI = &MF.getRegInfo();
  if (!MRI->tracksLiveness() || !MF.getProperties().hasNoVRegs())
    return false;

  DebugReferencedRegs.resize(TRI->getNumRegs());
  for (const MachineBasicBlock &MBB : MF)
    for (const MachineInstr &MI : MBB) {
      if (!MI.isDebugInstr())
        continue;
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg().isPhysical())
          continue;
        for (MCRegAliasIterator AI(MO.getReg(), TRI, /*IncludeSelf=*/true);
             AI.isValid(); ++AI)
          DebugReferencedRegs.set(*AI);
      }
    }

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    if (!rewriteBlock(MBB))
      continue;
    recomputeLivenessFlags(MBB);
    Changed = true;
  }
  return Changed;
}

PreservedAnalyses
RISCVPostRAV0RewritePass::run(MachineFunction &MF,
                              MachineFunctionAnalysisManager &MFAM) {
  MFPropsModifier _(*this, MF);
  if (!RISCVPostRAV0RewriteImpl().run(MF))
    return PreservedAnalyses::all();

  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

FunctionPass *llvm::createRISCVPostRAV0RewritePass() {
  return new RISCVPostRAV0RewriteLegacy();
}
