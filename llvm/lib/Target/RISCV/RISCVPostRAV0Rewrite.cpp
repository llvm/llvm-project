//===- RISCVPostRAV0Rewrite.cpp - Rewrite post-RA V0 copies --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// After register allocation, remove two narrowly proven classes of copies to
// V0.  The first retargets a safe same-block mask producer directly to V0:
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
STATISTIC(NumVMANDSingleWebsRewritten,
          "Number of post-RA VMAND single webs rewritten");
STATISTIC(NumVMANDFanoutWebsRewritten,
          "Number of post-RA VMAND fanout webs rewritten");

// Bound each backwards producer search so the pass remains O(N * K) per MBB.
// The frozen corpus that motivated this transform needed at most five
// intervening instructions.
static constexpr unsigned MaxProducerSearchInstructions = 5;

namespace {

class RISCVPostRAV0RewriteImpl {
  const TargetInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const MachineRegisterInfo *MRI = nullptr;
  BitVector DebugReferencedRegs;

  bool tryRetargetProducer(MachineBasicBlock::instr_iterator Producer,
                           MachineBasicBlock::instr_iterator Copy,
                           const LivePhysRegs &LiveAfterCopy) const;
  MachineBasicBlock::instr_iterator
  findProducer(MachineBasicBlock &MBB, MachineBasicBlock::instr_iterator Copy,
               Register Dst) const;
  bool tryRewriteCompareChain(MachineBasicBlock::instr_iterator Save,
                              MachineBasicBlock::instr_iterator Body,
                              MachineBasicBlock::instr_iterator Restore,
                              const LivePhysRegs &LiveAfterRestore) const;
  bool tryRewriteVMANDWeb(MachineBasicBlock &MBB,
                          MachineBasicBlock::instr_iterator Consumer,
                          const LivePhysRegs &LiveAfterConsumer) const;
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

static bool hasRegMask(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands())
    if (MO.isRegMask())
      return true;
  return false;
}

static bool isStructurallyTransparent(const MachineInstr &MI) {
  bool IsSafeLoad = MI.mayLoad() && !MI.mayStore() && !MI.hasOrderedMemoryRef();
  return !MI.isDebugInstr() && !MI.isMetaInstruction() && !MI.isPosition() &&
         !MI.isPseudoProbe() && !MI.isBundled() && !MI.peekDebugInstrNum() &&
         !MI.getFlag(MachineInstr::LRSplit) &&
         MI.getFlags() == MachineInstr::NoFlags && !MI.getAsmPrinterFlags() &&
         (MI.memoperands_empty() || IsSafeLoad) && !MI.getPreInstrSymbol() &&
         !MI.getPostInstrSymbol() && !MI.getHeapAllocMarker() &&
         !MI.getPCSections() && !MI.getMMRAMetadata() && !MI.getCFIType() &&
         !MI.getDeactivationSymbol() && !MI.isCall() && !MI.isInlineAsm() &&
         !MI.isTerminator() && !MI.isBarrier() && !MI.isConvergent() &&
         (!MI.mayLoadOrStore() || IsSafeLoad) && !MI.mayRaiseFPException() &&
         !MI.hasUnmodeledSideEffects() && !hasRegMask(MI);
}

MachineBasicBlock::instr_iterator
RISCVPostRAV0RewriteImpl::findProducer(MachineBasicBlock &MBB,
                                       MachineBasicBlock::instr_iterator Copy,
                                       Register Dst) const {
  bool CopyUsesVL = hasImplicitUse(*Copy, RISCV::VL);
  bool CopyUsesVType = hasImplicitUse(*Copy, RISCV::VTYPE);
  auto I = Copy;
  unsigned NumIntervening = 0;

  while (I != MBB.instr_begin()) {
    --I;
    MachineInstr &MI = *I;
    if (MI.isDebugInstr())
      continue;

    // This is the closest D definition. Let tryRetargetProducer validate the
    // producer itself; all other instructions in the bounded window must be
    // structurally plain and transparent to D and V0.
    if (MI.modifiesRegister(Dst, TRI))
      return I;
    if (NumIntervening == MaxProducerSearchInstructions ||
        !isStructurallyTransparent(MI) || MI.readsRegister(Dst, TRI) ||
        MI.readsRegister(RISCV::V0, TRI) ||
        MI.modifiesRegister(RISCV::V0, TRI) ||
        (CopyUsesVL && MI.modifiesRegister(RISCV::VL, TRI)) ||
        (CopyUsesVType && MI.modifiesRegister(RISCV::VTYPE, TRI)))
      return MBB.instr_end();
    ++NumIntervening;
  }
  return MBB.instr_end();
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

namespace {

struct MaskLogicMatch {
  MachineOperand *Def, *Src0, *Src1, *AVL;
  unsigned MaskRatio;
};

struct MaskCompareMatch {
  MachineOperand *Def, *Passthru, *Mask, *AVL, *Policy;
  unsigned MaskRatio;
};

static bool previousNonDebug(MachineBasicBlock &MBB,
                             MachineBasicBlock::instr_iterator &I) {
  while (I != MBB.instr_begin()) {
    --I;
    if (!I->isDebugInstr())
      return true;
  }
  return false;
}

static bool isPlainWebInstruction(const MachineInstr &MI) {
  if (MI.isDebugInstr() || MI.isMetaInstruction() || MI.isPosition() ||
      MI.isPseudoProbe() || MI.isBundled() || MI.peekDebugInstrNum() ||
      MI.getFlag(MachineInstr::LRSplit) ||
      MI.getFlags() != MachineInstr::NoFlags || MI.getAsmPrinterFlags() ||
      !MI.memoperands_empty() || MI.getPreInstrSymbol() ||
      MI.getPostInstrSymbol() || MI.getHeapAllocMarker() ||
      MI.getPCSections() || MI.getMMRAMetadata() || MI.getCFIType() ||
      MI.getDeactivationSymbol() || MI.isCall() || MI.isInlineAsm() ||
      MI.isTerminator() || MI.isBarrier() || MI.isConvergent() ||
      MI.mayLoadOrStore() || MI.hasUnmodeledSideEffects())
    return false;
  return llvm::none_of(MI.explicit_operands(), [](const MachineOperand &MO) {
    return MO.getTargetFlags();
  });
}

static bool isPlainRegOperand(const MachineOperand &MO, Register Reg,
                              bool IsDef) {
  return MO.isReg() && MO.isDef() == IsDef && MO.readsReg() != IsDef &&
         !MO.isUndef() && !MO.isImplicit() && !MO.isInternalRead() &&
         !MO.isEarlyClobber() && !MO.getSubReg() && !MO.getTargetFlags() &&
         MO.getReg() == Reg;
}

static bool hasExactVectorStateUses(const MachineInstr &MI) {
  bool SeenVL = false, SeenVType = false;
  for (unsigned I = MI.getNumExplicitOperands(), E = MI.getNumOperands();
       I != E; ++I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (!MO.isReg() || !MO.isImplicit() || !MO.isUse() || !MO.readsReg() ||
        MO.isUndef() || MO.isKill() || MO.isInternalRead() ||
        MO.isEarlyClobber() || MO.getSubReg() || MO.getTargetFlags())
      return false;
    if (MO.getReg() == RISCV::VL) {
      if (SeenVL)
        return false;
      SeenVL = true;
    } else if (MO.getReg() == RISCV::VTYPE) {
      if (SeenVType)
        return false;
      SeenVType = true;
    } else
      return false;
  }
  return SeenVL && SeenVType;
}

static bool operandClassContains(const MachineInstr &MI, unsigned OpIdx,
                                 Register Reg, const TargetInstrInfo *TII,
                                 const TargetRegisterInfo *TRI) {
  const TargetRegisterClass *RC = MI.getRegClassConstraint(OpIdx, TII, TRI);
  return RC && RC->contains(Reg);
}

static bool matchMaskLogic(MachineInstr &MI, Register DefReg,
                           const TargetInstrInfo *TII,
                           const TargetRegisterInfo *TRI,
                           MaskLogicMatch &Match) {
  if (!isPlainWebInstruction(MI) || MI.getNumExplicitDefs() != 1 ||
      MI.getNumExplicitOperands() != 5 || !hasExactVectorStateUses(MI) ||
      !RISCVII::hasVLOp(MI.getDesc().TSFlags) ||
      !RISCVII::hasSEWOp(MI.getDesc().TSFlags))
    return false;
  MachineOperand &Def = MI.getOperand(0), &Src0 = MI.getOperand(1),
                 &Src1 = MI.getOperand(2);
  unsigned VLIdx = RISCVII::getVLOpNum(MI.getDesc());
  unsigned SEWIdx = RISCVII::getSEWOpNum(MI.getDesc());
  if (VLIdx >= 5 || SEWIdx >= 5 || !isPlainRegOperand(Def, DefReg, true) ||
      !Src0.isReg() || !Src0.getReg().isPhysical() || !Src1.isReg() ||
      !Src1.getReg().isPhysical() ||
      !isPlainRegOperand(Src0, Src0.getReg(), false) ||
      !isPlainRegOperand(Src1, Src1.getReg(), false) ||
      !operandClassContains(MI, 0, DefReg, TII, TRI) ||
      !operandClassContains(MI, 1, Src0.getReg(), TII, TRI) ||
      !operandClassContains(MI, 2, Src1.getReg(), TII, TRI) ||
      !MI.getOperand(SEWIdx).isImm() || MI.getOperand(SEWIdx).getImm() != 0)
    return false;
  Match = {
      &Def, &Src0, &Src1, &MI.getOperand(VLIdx),
      RISCVVType::getSEWLMULRatio(8, RISCVII::getLMul(MI.getDesc().TSFlags))};
  return true;
}

static bool matchMaskedCompare(MachineInstr &MI, Register Reg,
                               const TargetInstrInfo *TII,
                               const TargetRegisterInfo *TRI,
                               MaskCompareMatch &Match) {
  const RISCV::RISCVMaskedPseudoInfo *Info =
      RISCV::getMaskedPseudoInfo(MI.getOpcode());
  if (!Info || !RISCVInstrInfo::isRVVCompare(MI) || !MI.mayRaiseFPException() ||
      !isPlainWebInstruction(MI) || MI.getNumExplicitDefs() != 1 ||
      !hasExactVectorStateUses(MI) || !RISCVII::hasVLOp(MI.getDesc().TSFlags) ||
      !RISCVII::hasSEWOp(MI.getDesc().TSFlags) ||
      !RISCVII::hasVecPolicyOp(MI.getDesc().TSFlags))
    return false;
  unsigned PassthruIdx;
  if (!MI.isRegTiedToUseOperand(0, &PassthruIdx))
    return false;
  unsigned MaskIdx = Info->MaskOpIdx + 1;
  unsigned VLIdx = RISCVII::getVLOpNum(MI.getDesc());
  unsigned SEWIdx = RISCVII::getSEWOpNum(MI.getDesc());
  unsigned PolicyIdx = RISCVII::getVecPolicyOpNum(MI.getDesc());
  if (PassthruIdx >= MI.getNumExplicitOperands() ||
      MaskIdx >= MI.getNumExplicitOperands() ||
      VLIdx >= MI.getNumExplicitOperands() ||
      SEWIdx >= MI.getNumExplicitOperands() ||
      PolicyIdx >= MI.getNumExplicitOperands() || PassthruIdx == MaskIdx)
    return false;
  MachineOperand &Def = MI.getOperand(0),
                 &Passthru = MI.getOperand(PassthruIdx),
                 &Mask = MI.getOperand(MaskIdx), &SEW = MI.getOperand(SEWIdx),
                 &Policy = MI.getOperand(PolicyIdx);
  if (!isPlainRegOperand(Def, Reg, true) ||
      !isPlainRegOperand(Passthru, Reg, false) ||
      !isPlainRegOperand(Mask, RISCV::V0, false) || !SEW.isImm() ||
      SEW.getImm() < 3 || SEW.getImm() > 6 || !Policy.isImm() ||
      (Policy.getImm() & RISCVVType::MASK_AGNOSTIC) ||
      !operandClassContains(MI, 0, Reg, TII, TRI) ||
      !operandClassContains(MI, 0, RISCV::V0, TII, TRI) ||
      !operandClassContains(MI, PassthruIdx, Reg, TII, TRI) ||
      !operandClassContains(MI, PassthruIdx, RISCV::V0, TII, TRI))
    return false;
  const TargetRegisterClass *MaskRC =
      MI.getRegClassConstraint(MaskIdx, TII, TRI);
  auto [LMul, Fractional] =
      RISCVVType::decodeVLMUL(RISCVII::getLMul(MI.getDesc().TSFlags));
  if (!MaskRC || !RISCVRegisterInfo::isV0OnlyRegClass(MaskRC, 0, *TRI) ||
      (!Fractional && LMul != 1))
    return false;
  Match = {&Def,
           &Passthru,
           &Mask,
           &MI.getOperand(VLIdx),
           &Policy,
           RISCVVType::getSEWLMULRatio(1U << SEW.getImm(),
                                       RISCVII::getLMul(MI.getDesc().TSFlags))};
  return true;
}

static bool matchWebCopy(MachineInstr &MI, Register Dst, Register Src,
                         const MachineInstr &Producer) {
  return isPlainFullCopy(MI, Dst, Src) &&
         isPlainRegOperand(MI.getOperand(0), Dst, true) &&
         isPlainRegOperand(MI.getOperand(1), Src, false) &&
         !MI.getOperand(1).isKill() &&
         copyHasOnlySharedImplicitExtras(MI, Producer);
}

static bool hasSameImmediateAVL(ArrayRef<MachineOperand *> AVLs) {
  if (!AVLs.front()->isImm())
    return false;
  for (MachineOperand *AVL : AVLs.drop_front())
    if (!AVL->isImm() || !AVL->isIdenticalTo(*AVLs.front()))
      return false;
  return true;
}

static bool validateClosedWeb(ArrayRef<MachineInstr *> Roles,
                              ArrayRef<Register> Colors,
                              ArrayRef<const MachineOperand *> Allowed,
                              ArrayRef<unsigned> Expected,
                              const TargetRegisterInfo *TRI,
                              const BitVector &DebugReferencedRegs) {
  SmallVector<unsigned, 3> Actual(Colors.size());
  for (unsigned I = 0; I != Colors.size(); ++I) {
    Register Reg = Colors[I];
    if (!Reg.isPhysical() || !RISCV::VRRegClass.contains(Reg) ||
        DebugReferencedRegs.test(Reg.id()))
      return false;
    for (Register Other : Colors.take_front(I))
      if (TRI->regsOverlap(Reg, Other))
        return false;
  }
  for (const MachineInstr *MI : Roles)
    for (const MachineOperand &MO : MI->operands()) {
      if (MO.isRegMask()) {
        for (Register Reg : Colors)
          if (MO.clobbersPhysReg(Reg))
            return false;
        continue;
      }
      if (!MO.isReg() || !MO.getReg())
        continue;
      for (unsigned I = 0; I != Colors.size(); ++I) {
        if (!TRI->regsOverlap(MO.getReg(), Colors[I]))
          continue;
        if (MO.getReg() != Colors[I] || !llvm::is_contained(Allowed, &MO))
          return false;
        ++Actual[I];
      }
    }
  return Actual == Expected;
}

static void makeNonRenamable(ArrayRef<MachineOperand *> Operands) {
  for (MachineOperand *MO : Operands)
    MO->setIsRenamable(false);
}

} // end anonymous namespace

bool RISCVPostRAV0RewriteImpl::tryRewriteVMANDWeb(
    MachineBasicBlock &MBB, MachineBasicBlock::instr_iterator Consumer,
    const LivePhysRegs &LiveAfterConsumer) const {
  unsigned ConsumerOpcode = RISCV::getRVVMCOpcode(Consumer->getOpcode());
  bool IsSingle = ConsumerOpcode == RISCV::VMNAND_MM;
  bool IsFanout =
      ConsumerOpcode == RISCV::VMOR_MM || ConsumerOpcode == RISCV::VMNOR_MM;
  if (!IsSingle && !IsFanout)
    return false;

  auto CompareD = Consumer;
  if (!previousNonDebug(MBB, CompareD))
    return false;
  auto CopyV0 = CompareD;
  MachineBasicBlock::instr_iterator CopyE = MBB.instr_end();
  MachineBasicBlock::instr_iterator CompareE = MBB.instr_end();
  if (IsFanout) {
    if (!previousNonDebug(MBB, CopyV0))
      return false;
    CompareE = CopyV0;
    CopyE = CompareE;
    if (!previousNonDebug(MBB, CopyE))
      return false;
    CopyV0 = CopyE;
  }
  if (!previousNonDebug(MBB, CopyV0))
    return false;
  auto Producer = CopyV0;
  if (!previousNonDebug(MBB, Producer) || !isStructurallyPlainCopy(*CopyV0) ||
      (IsFanout && !isStructurallyPlainCopy(*CopyE)))
    return false;

  Register D = CopyV0->getOperand(1).getReg();
  Register E = IsFanout ? CopyE->getOperand(0).getReg() : Register();
  if (!D.isPhysical() || D == RISCV::V0 ||
      (IsFanout && (!E.isPhysical() || E == RISCV::V0)))
    return false;

  MaskLogicMatch ProducerMatch, ConsumerMatch;
  MaskCompareMatch CompareDMatch, CompareEMatch;
  if (RISCV::getRVVMCOpcode(Producer->getOpcode()) != RISCV::VMAND_MM ||
      !matchMaskLogic(*Producer, D, TII, TRI, ProducerMatch) ||
      !matchWebCopy(*CopyV0, RISCV::V0, D, *Producer) ||
      !matchMaskedCompare(*CompareD, D, TII, TRI, CompareDMatch) ||
      !matchMaskLogic(*Consumer, RISCV::V0, TII, TRI, ConsumerMatch))
    return false;

  MachineOperand *ProducerD = nullptr, *ProducerOther = nullptr;
  for (MachineOperand *MO : {ProducerMatch.Src0, ProducerMatch.Src1}) {
    if (!MO->isKill())
      return false;
    if (MO->getReg() == D)
      ProducerD = ProducerD ? nullptr : MO;
    else
      ProducerOther = ProducerOther ? nullptr : MO;
  }
  if (!ProducerD || !ProducerOther ||
      TRI->regsOverlap(ProducerOther->getReg(), D) ||
      TRI->regsOverlap(ProducerOther->getReg(), RISCV::V0) ||
      !CompareDMatch.Passthru->isKill() || !CompareDMatch.Mask->isKill() ||
      ProducerMatch.MaskRatio != CompareDMatch.MaskRatio ||
      ProducerMatch.MaskRatio != ConsumerMatch.MaskRatio ||
      !LiveAfterConsumer.available(*MRI, D))
    return false;

  if (IsSingle) {
    if (ConsumerMatch.Src0->getReg() != D ||
        ConsumerMatch.Src1->getReg() != D ||
        ConsumerMatch.Src0->isKill() == ConsumerMatch.Src1->isKill() ||
        !operandClassContains(*Producer, 0, RISCV::V0, TII, TRI) ||
        !operandClassContains(*Consumer, 1, RISCV::V0, TII, TRI) ||
        !operandClassContains(*Consumer, 2, RISCV::V0, TII, TRI))
      return false;
    MachineInstr *Roles[] = {&*Producer, &*CopyV0, &*CompareD, &*Consumer};
    MachineOperand *AVLs[] = {ProducerMatch.AVL, CompareDMatch.AVL,
                              ConsumerMatch.AVL};
    const MachineOperand *Allowed[] = {
        ProducerMatch.Def,      ProducerD,         &CopyV0->getOperand(0),
        &CopyV0->getOperand(1), CompareDMatch.Def, CompareDMatch.Passthru,
        CompareDMatch.Mask,     ConsumerMatch.Def, ConsumerMatch.Src0,
        ConsumerMatch.Src1};
    const Register Colors[] = {D, RISCV::V0};
    const unsigned Counts[] = {7, 3};
    if (!hasSameImmediateAVL(AVLs) ||
        !validateClosedWeb(Roles, Colors, Allowed, Counts, TRI,
                           DebugReferencedRegs))
      return false;

    ProducerMatch.Def->setReg(RISCV::V0);
    CompareDMatch.Def->setReg(RISCV::V0);
    CompareDMatch.Passthru->setReg(RISCV::V0);
    ConsumerMatch.Src0->setReg(RISCV::V0);
    ConsumerMatch.Src1->setReg(RISCV::V0);
    makeNonRenamable({ProducerMatch.Def, CompareDMatch.Def,
                      CompareDMatch.Passthru, CompareDMatch.Mask,
                      ConsumerMatch.Def, ConsumerMatch.Src0,
                      ConsumerMatch.Src1});
    CopyV0->eraseFromParent();
    ++NumVMANDSingleWebsRewritten;
    return true;
  }

  if (ProducerOther->getReg() != E || !matchWebCopy(*CopyE, E, D, *Producer) ||
      !matchMaskedCompare(*CompareE, E, TII, TRI, CompareEMatch) ||
      !CompareEMatch.Passthru->isKill() || CompareEMatch.Mask->isKill() ||
      !LiveAfterConsumer.available(*MRI, E) ||
      ProducerMatch.MaskRatio != CompareEMatch.MaskRatio ||
      !CompareDMatch.Policy->isIdenticalTo(*CompareEMatch.Policy) ||
      !operandClassContains(*Producer, 0, E, TII, TRI))
    return false;
  MachineOperand *ConsumerD = nullptr, *ConsumerE = nullptr;
  for (MachineOperand *MO : {ConsumerMatch.Src0, ConsumerMatch.Src1}) {
    if (!MO->isKill())
      return false;
    if (MO->getReg() == D)
      ConsumerD = ConsumerD ? nullptr : MO;
    else if (MO->getReg() == E)
      ConsumerE = ConsumerE ? nullptr : MO;
  }
  if (!ConsumerD || !ConsumerE ||
      !operandClassContains(*Consumer, ConsumerD->getOperandNo(), RISCV::V0,
                            TII, TRI))
    return false;

  MachineInstr *Roles[] = {&*Producer, &*CopyV0,   &*CopyE,
                           &*CompareE, &*CompareD, &*Consumer};
  MachineOperand *AVLs[] = {ProducerMatch.AVL, CompareEMatch.AVL,
                            CompareDMatch.AVL, ConsumerMatch.AVL};
  const MachineOperand *Allowed[] = {ProducerMatch.Def,
                                     ProducerD,
                                     ProducerOther,
                                     &CopyV0->getOperand(0),
                                     &CopyV0->getOperand(1),
                                     &CopyE->getOperand(0),
                                     &CopyE->getOperand(1),
                                     CompareEMatch.Def,
                                     CompareEMatch.Passthru,
                                     CompareEMatch.Mask,
                                     CompareDMatch.Def,
                                     CompareDMatch.Passthru,
                                     CompareDMatch.Mask,
                                     ConsumerMatch.Def,
                                     ConsumerD,
                                     ConsumerE};
  const Register Colors[] = {D, E, RISCV::V0};
  const unsigned Counts[] = {7, 5, 4};
  if (!hasSameImmediateAVL(AVLs) ||
      !validateClosedWeb(Roles, Colors, Allowed, Counts, TRI,
                         DebugReferencedRegs))
    return false;

  ProducerMatch.Def->setReg(E);
  CopyV0->getOperand(1).setReg(E);
  CompareDMatch.Def->setReg(RISCV::V0);
  CompareDMatch.Passthru->setReg(RISCV::V0);
  ConsumerD->setReg(RISCV::V0);
  makeNonRenamable(
      {ProducerMatch.Def, &CopyV0->getOperand(0), &CopyV0->getOperand(1),
       CompareEMatch.Def, CompareEMatch.Passthru, CompareEMatch.Mask,
       CompareDMatch.Def, CompareDMatch.Passthru, CompareDMatch.Mask,
       ConsumerMatch.Def, ConsumerD, ConsumerE});
  CopyE->eraseFromParent();
  ++NumVMANDFanoutWebsRewritten;
  return true;
}

bool RISCVPostRAV0RewriteImpl::rewriteBlock(MachineBasicBlock &MBB) const {
  bool Changed = false;
  bool RestartAfterWeb;
  do {
    RestartAfterWeb = false;
    LivePhysRegs Live(*TRI);
    Live.addLiveOuts(MBB);

    // Local COPY rewrites leave this reverse cursor valid. A web recolors
    // several earlier instructions, so recompute flags and restart.
    for (auto I = MBB.instr_end(); I != MBB.instr_begin();) {
      auto Current = std::prev(I);
      MachineInstr &MI = *Current;
      if (MI.isDebugInstr() || MI.isBundledWithPred()) {
        I = Current;
        continue;
      }

      unsigned MCOpcode = RISCV::getRVVMCOpcode(MI.getOpcode());
      if ((MCOpcode == RISCV::VMNAND_MM || MCOpcode == RISCV::VMOR_MM ||
           MCOpcode == RISCV::VMNOR_MM) &&
          tryRewriteVMANDWeb(MBB, Current, Live)) {
        Changed = RestartAfterWeb = true;
        recomputeLivenessFlags(MBB);
        break;
      }

      if (isStructurallyPlainCopy(MI) &&
          MI.getOperand(0).getReg() == RISCV::V0) {
        MachineBasicBlock::instr_iterator Restore(MI);
        if (Restore != MBB.instr_begin()) {
          auto Previous = prev_nodbg(Restore, MBB.instr_begin(),
                                     /*SkipPseudoOp=*/false);
          Register Dst = Restore->getOperand(1).getReg();
          auto Producer = Dst.isPhysical() && Dst != RISCV::V0 &&
                                  RISCV::VRRegClass.contains(Dst)
                              ? findProducer(MBB, Restore, Dst)
                              : MBB.instr_end();
          if (Producer != MBB.instr_end() &&
              tryRetargetProducer(Producer, Restore, Live)) {
            Changed = true;
            continue;
          }

          if (Previous != MBB.instr_begin()) {
            auto Save = prev_nodbg(Previous, MBB.instr_begin(),
                                   /*SkipPseudoOp=*/false);
            if (tryRewriteCompareChain(Save, Previous, Restore, Live)) {
              Changed = true;
              continue;
            }
          }
        }
      }

      Live.stepBackward(MI);
      I = Current;
    }
  } while (RestartAfterWeb);

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
