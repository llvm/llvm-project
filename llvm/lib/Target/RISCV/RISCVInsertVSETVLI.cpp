//===- RISCVInsertVSETVLI.cpp - Insert VSETVLI instructions ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function pass that inserts VSETVLI instructions where
// needed.
//
// This pass consists of 3 phases:
//
// Phase 1 collects how each basic block affects VL/VTYPE.
//
// Phase 2 uses the information from phase 1 to do a data flow analysis to
// propagate the VL/VTYPE changes through the function. This gives us the
// VL/VTYPE at the start of each basic block.
//
// Phase 3 inserts VSETVLI instructions in each basic block. Information from
// phase 2 is used to prevent inserting a VSETVLI before the first vector
// instruction in the block if possible.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include <queue>
using namespace llvm;

#define DEBUG_TYPE "riscv-insert-vsetvli"
#define RISCV_INSERT_VSETVLI_NAME "RISCV Insert VSETVLI pass"

static cl::opt<bool> DisableInsertVSETVLPHIOpt(
    "riscv-disable-insert-vsetvl-phi-opt", cl::init(false), cl::Hidden,
    cl::desc("Disable looking through phis when inserting vsetvlis."));

namespace {

class VSETVLIInfo {
  union {
    Register AVLReg;
    unsigned AVLImm;
  };

  enum : uint8_t {
    Uninitialized,
    AVLIsReg,
    AVLIsImm,
    Unknown,
  } State = Uninitialized;

  // Fields from VTYPE.
  RISCVII::VLMUL VLMul = RISCVII::LMUL_1;
  uint8_t SEW = 0;
  uint8_t TailAgnostic : 1;
  uint8_t MaskAgnostic : 1;
  uint8_t MaskRegOp : 1;
  uint8_t StoreOp : 1;
  uint8_t SEWLMULRatioOnly : 1;

public:
  VSETVLIInfo()
      : AVLImm(0), TailAgnostic(false), MaskAgnostic(false), MaskRegOp(false),
        StoreOp(false), SEWLMULRatioOnly(false) {}

  static VSETVLIInfo getUnknown() {
    VSETVLIInfo Info;
    Info.setUnknown();
    return Info;
  }

  bool isValid() const { return State != Uninitialized; }
  void setUnknown() { State = Unknown; }
  bool isUnknown() const { return State == Unknown; }

  void setAVLReg(Register Reg) {
    AVLReg = Reg;
    State = AVLIsReg;
  }

  void setAVLImm(unsigned Imm) {
    AVLImm = Imm;
    State = AVLIsImm;
  }

  bool hasAVLImm() const { return State == AVLIsImm; }
  bool hasAVLReg() const { return State == AVLIsReg; }
  Register getAVLReg() const {
    assert(hasAVLReg());
    return AVLReg;
  }
  unsigned getAVLImm() const {
    assert(hasAVLImm());
    return AVLImm;
  }

  bool hasSameAVL(const VSETVLIInfo &Other) const {
    assert(isValid() && Other.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!isUnknown() && !Other.isUnknown() &&
           "Can't compare AVL in unknown state");
    if (hasAVLReg() && Other.hasAVLReg())
      return getAVLReg() == Other.getAVLReg();

    if (hasAVLImm() && Other.hasAVLImm())
      return getAVLImm() == Other.getAVLImm();

    return false;
  }

  void setVTYPE(unsigned VType) {
    assert(isValid() && !isUnknown() &&
           "Can't set VTYPE for uninitialized or unknown");
    VLMul = RISCVVType::getVLMUL(VType);
    SEW = RISCVVType::getSEW(VType);
    TailAgnostic = RISCVVType::isTailAgnostic(VType);
    MaskAgnostic = RISCVVType::isMaskAgnostic(VType);
  }
  void setVTYPE(RISCVII::VLMUL L, unsigned S, bool TA, bool MA, bool MRO,
                bool IsStore) {
    assert(isValid() && !isUnknown() &&
           "Can't set VTYPE for uninitialized or unknown");
    VLMul = L;
    SEW = S;
    TailAgnostic = TA;
    MaskAgnostic = MA;
    MaskRegOp = MRO;
    StoreOp = IsStore;
  }

  unsigned encodeVTYPE() const {
    assert(isValid() && !isUnknown() && !SEWLMULRatioOnly &&
           "Can't encode VTYPE for uninitialized or unknown");
    return RISCVVType::encodeVTYPE(VLMul, SEW, TailAgnostic, MaskAgnostic);
  }

  bool hasSEWLMULRatioOnly() const { return SEWLMULRatioOnly; }

  bool hasSameVTYPE(const VSETVLIInfo &Other) const {
    assert(isValid() && Other.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!isUnknown() && !Other.isUnknown() &&
           "Can't compare VTYPE in unknown state");
    assert(!SEWLMULRatioOnly && !Other.SEWLMULRatioOnly &&
           "Can't compare when only LMUL/SEW ratio is valid.");
    return std::tie(VLMul, SEW, TailAgnostic, MaskAgnostic) ==
           std::tie(Other.VLMul, Other.SEW, Other.TailAgnostic,
                    Other.MaskAgnostic);
  }

  static unsigned getSEWLMULRatio(unsigned SEW, RISCVII::VLMUL VLMul) {
    unsigned LMul;
    bool Fractional;
    std::tie(LMul, Fractional) = RISCVVType::decodeVLMUL(VLMul);

    // Convert LMul to a fixed point value with 3 fractional bits.
    LMul = Fractional ? (8 / LMul) : (LMul * 8);

    assert(SEW >= 8 && "Unexpected SEW value");
    return (SEW * 8) / LMul;
  }

  unsigned getSEWLMULRatio() const {
    assert(isValid() && !isUnknown() &&
           "Can't use VTYPE for uninitialized or unknown");
    return getSEWLMULRatio(SEW, VLMul);
  }

  // Check if the VTYPE for these two VSETVLIInfos produce the same VLMAX.
  bool hasSameVLMAX(const VSETVLIInfo &Other) const {
    assert(isValid() && Other.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!isUnknown() && !Other.isUnknown() &&
           "Can't compare VTYPE in unknown state");
    return getSEWLMULRatio() == Other.getSEWLMULRatio();
  }

  // Determine whether the vector instructions requirements represented by
  // InstrInfo are compatible with the previous vsetvli instruction represented
  // by this.
  bool isCompatible(const VSETVLIInfo &InstrInfo, bool Strict) const {
    assert(isValid() && InstrInfo.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!InstrInfo.SEWLMULRatioOnly &&
           "Expected a valid VTYPE for instruction!");
    // Nothing is compatible with Unknown.
    if (isUnknown() || InstrInfo.isUnknown())
      return false;

    // If only our VLMAX ratio is valid, then this isn't compatible.
    if (SEWLMULRatioOnly)
      return false;

    // If the instruction doesn't need an AVLReg and the SEW matches, consider
    // it compatible.
    if (!Strict && InstrInfo.hasAVLReg() &&
        InstrInfo.AVLReg == RISCV::NoRegister) {
      if (SEW == InstrInfo.SEW)
        return true;
    }

    // The AVL must match.
    if (!hasSameAVL(InstrInfo))
      return false;

    // Simple case, see if full VTYPE matches.
    if (hasSameVTYPE(InstrInfo))
      return true;

    // Strict matches must ensure a full VTYPE match.
    if (Strict)
      return false;

    // If this is a mask reg operation, it only cares about VLMAX.
    // FIXME: Mask reg operations are probably ok if "this" VLMAX is larger
    // than "InstrInfo".
    if (InstrInfo.MaskRegOp && hasSameVLMAX(InstrInfo) &&
        TailAgnostic == InstrInfo.TailAgnostic &&
        MaskAgnostic == InstrInfo.MaskAgnostic)
      return true;

    // Store instructions don't use the policy fields.
    if (InstrInfo.StoreOp && VLMul == InstrInfo.VLMul && SEW == InstrInfo.SEW)
      return true;

    // Anything else is not compatible.
    return false;
  }

  bool isCompatibleWithLoadStoreEEW(unsigned EEW,
                                    const VSETVLIInfo &InstrInfo) const {
    assert(isValid() && InstrInfo.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!InstrInfo.SEWLMULRatioOnly &&
           "Expected a valid VTYPE for instruction!");
    assert(EEW == InstrInfo.SEW && "Mismatched EEW/SEW for store");

    if (isUnknown() || hasSEWLMULRatioOnly())
      return false;

    if (!hasSameAVL(InstrInfo))
      return false;

    // Stores can ignore the tail and mask policies.
    if (!InstrInfo.StoreOp && (TailAgnostic != InstrInfo.TailAgnostic ||
                               MaskAgnostic != InstrInfo.MaskAgnostic))
      return false;

    return getSEWLMULRatio() == getSEWLMULRatio(EEW, InstrInfo.VLMul);
  }

  bool operator==(const VSETVLIInfo &Other) const {
    // Uninitialized is only equal to another Uninitialized.
    if (!isValid())
      return !Other.isValid();
    if (!Other.isValid())
      return !isValid();

    // Unknown is only equal to another Unknown.
    if (isUnknown())
      return Other.isUnknown();
    if (Other.isUnknown())
      return isUnknown();

    if (!hasSameAVL(Other))
      return false;

    // If only the VLMAX is valid, check that it is the same.
    if (SEWLMULRatioOnly && Other.SEWLMULRatioOnly)
      return hasSameVLMAX(Other);

    // If the full VTYPE is valid, check that it is the same.
    if (!SEWLMULRatioOnly && !Other.SEWLMULRatioOnly)
      return hasSameVTYPE(Other);

    // If the SEWLMULRatioOnly bits are different, then they aren't equal.
    return false;
  }

  // Calculate the VSETVLIInfo visible to a block assuming this and Other are
  // both predecessors.
  VSETVLIInfo intersect(const VSETVLIInfo &Other) const {
    // If the new value isn't valid, ignore it.
    if (!Other.isValid())
      return *this;

    // If this value isn't valid, this must be the first predecessor, use it.
    if (!isValid())
      return Other;

    // If either is unknown, the result is unknown.
    if (isUnknown() || Other.isUnknown())
      return VSETVLIInfo::getUnknown();

    // If we have an exact, match return this.
    if (*this == Other)
      return *this;

    // Not an exact match, but maybe the AVL and VLMAX are the same. If so,
    // return an SEW/LMUL ratio only value.
    if (hasSameAVL(Other) && hasSameVLMAX(Other)) {
      VSETVLIInfo MergeInfo = *this;
      MergeInfo.SEWLMULRatioOnly = true;
      return MergeInfo;
    }

    // Otherwise the result is unknown.
    return VSETVLIInfo::getUnknown();
  }

  // Calculate the VSETVLIInfo visible at the end of the block assuming this
  // is the predecessor value, and Other is change for this block.
  VSETVLIInfo merge(const VSETVLIInfo &Other) const {
    assert(isValid() && "Can only merge with a valid VSETVLInfo");

    // Nothing changed from the predecessor, keep it.
    if (!Other.isValid())
      return *this;

    // If the change is compatible with the input, we won't create a VSETVLI
    // and should keep the predecessor.
    if (isCompatible(Other, /*Strict*/ true))
      return *this;

    // Otherwise just use whatever is in this block.
    return Other;
  }
};

struct BlockData {
  // The VSETVLIInfo that represents the net changes to the VL/VTYPE registers
  // made by this block. Calculated in Phase 1.
  VSETVLIInfo Change;

  // The VSETVLIInfo that represents the VL/VTYPE settings on exit from this
  // block. Calculated in Phase 2.
  VSETVLIInfo Exit;

  // The VSETVLIInfo that represents the VL/VTYPE settings from all predecessor
  // blocks. Calculated in Phase 2, and used by Phase 3.
  VSETVLIInfo Pred;

  // Keeps track of whether the block is already in the queue.
  bool InQueue = false;

  BlockData() {}
};

class RISCVInsertVSETVLI : public MachineFunctionPass {
  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;

  std::vector<BlockData> BlockInfo;
  std::queue<const MachineBasicBlock *> WorkList;

public:
  static char ID;

  RISCVInsertVSETVLI() : MachineFunctionPass(ID) {
    initializeRISCVInsertVSETVLIPass(*PassRegistry::getPassRegistry());
  }
  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_INSERT_VSETVLI_NAME; }

private:
  bool needVSETVLI(const VSETVLIInfo &Require, const VSETVLIInfo &CurInfo);
  bool needVSETVLIPHI(const VSETVLIInfo &Require, const MachineBasicBlock &MBB);
  void insertVSETVLI(MachineBasicBlock &MBB, MachineInstr &MI,
                     const VSETVLIInfo &Info, const VSETVLIInfo &PrevInfo);

  bool computeVLVTYPEChanges(const MachineBasicBlock &MBB);
  void computeIncomingVLVTYPE(const MachineBasicBlock &MBB);
  void emitVSETVLIs(MachineBasicBlock &MBB);
};

} // end anonymous namespace

char RISCVInsertVSETVLI::ID = 0;

INITIALIZE_PASS(RISCVInsertVSETVLI, DEBUG_TYPE, RISCV_INSERT_VSETVLI_NAME,
                false, false)

static MachineInstr *elideCopies(MachineInstr *MI,
                                 const MachineRegisterInfo *MRI) {
  while (true) {
    if (!MI->isFullCopy())
      return MI;
    if (!Register::isVirtualRegister(MI->getOperand(1).getReg()))
      return nullptr;
    MI = MRI->getVRegDef(MI->getOperand(1).getReg());
    if (!MI)
      return nullptr;
  }
}

static VSETVLIInfo computeInfoForInstr(const MachineInstr &MI, uint64_t TSFlags,
                                       const MachineRegisterInfo *MRI) {
  VSETVLIInfo InstrInfo;
  unsigned NumOperands = MI.getNumExplicitOperands();
  bool HasPolicy = RISCVII::hasVecPolicyOp(TSFlags);

  // Default to tail agnostic unless the destination is tied to a source.
  // Unless the source is undef. In that case the user would have some control
  // over the tail values. Some pseudo instructions force a tail agnostic policy
  // despite having a tied def.
  bool ForceTailAgnostic = RISCVII::doesForceTailAgnostic(TSFlags);
  bool TailAgnostic = true;
  // If the instruction has policy argument, use the argument.
  if (HasPolicy) {
    const MachineOperand &Op = MI.getOperand(MI.getNumExplicitOperands() - 1);
    TailAgnostic = Op.getImm() & 0x1;
  }

  unsigned UseOpIdx;
  if (!(ForceTailAgnostic || (HasPolicy && TailAgnostic)) &&
      MI.isRegTiedToUseOperand(0, &UseOpIdx)) {
    TailAgnostic = false;
    // If the tied operand is an IMPLICIT_DEF we can keep TailAgnostic.
    const MachineOperand &UseMO = MI.getOperand(UseOpIdx);
    MachineInstr *UseMI = MRI->getVRegDef(UseMO.getReg());
    if (UseMI) {
      UseMI = elideCopies(UseMI, MRI);
      if (UseMI && UseMI->isImplicitDef())
        TailAgnostic = true;
    }
  }

  // Remove the tail policy so we can find the SEW and VL.
  if (HasPolicy)
    --NumOperands;

  RISCVII::VLMUL VLMul = RISCVII::getLMul(TSFlags);

  unsigned Log2SEW = MI.getOperand(NumOperands - 1).getImm();
  // A Log2SEW of 0 is an operation on mask registers only.
  bool MaskRegOp = Log2SEW == 0;
  unsigned SEW = Log2SEW ? 1 << Log2SEW : 8;
  assert(RISCVVType::isValidSEW(SEW) && "Unexpected SEW");

  // If there are no explicit defs, this is a store instruction which can
  // ignore the tail and mask policies.
  bool StoreOp = MI.getNumExplicitDefs() == 0;

  if (RISCVII::hasVLOp(TSFlags)) {
    const MachineOperand &VLOp = MI.getOperand(NumOperands - 2);
    if (VLOp.isImm()) {
      int64_t Imm = VLOp.getImm();
      // Conver the VLMax sentintel to X0 register.
      if (Imm == RISCV::VLMaxSentinel)
        InstrInfo.setAVLReg(RISCV::X0);
      else
        InstrInfo.setAVLImm(Imm);
    } else {
      InstrInfo.setAVLReg(VLOp.getReg());
    }
  } else
    InstrInfo.setAVLReg(RISCV::NoRegister);
  InstrInfo.setVTYPE(VLMul, SEW, /*TailAgnostic*/ TailAgnostic,
                     /*MaskAgnostic*/ false, MaskRegOp, StoreOp);

  return InstrInfo;
}

void RISCVInsertVSETVLI::insertVSETVLI(MachineBasicBlock &MBB, MachineInstr &MI,
                                       const VSETVLIInfo &Info,
                                       const VSETVLIInfo &PrevInfo) {
  DebugLoc DL = MI.getDebugLoc();

  // Use X0, X0 form if the AVL is the same and the SEW+LMUL gives the same
  // VLMAX.
  if (PrevInfo.isValid() && !PrevInfo.isUnknown() &&
      Info.hasSameAVL(PrevInfo) && Info.hasSameVLMAX(PrevInfo)) {
    BuildMI(MBB, MI, DL, TII->get(RISCV::PseudoVSETVLIX0))
        .addReg(RISCV::X0, RegState::Define | RegState::Dead)
        .addReg(RISCV::X0, RegState::Kill)
        .addImm(Info.encodeVTYPE())
        .addReg(RISCV::VL, RegState::Implicit);
    return;
  }

  if (Info.hasAVLImm()) {
    BuildMI(MBB, MI, DL, TII->get(RISCV::PseudoVSETIVLI))
        .addReg(RISCV::X0, RegState::Define | RegState::Dead)
        .addImm(Info.getAVLImm())
        .addImm(Info.encodeVTYPE());
    return;
  }

  Register AVLReg = Info.getAVLReg();
  if (AVLReg == RISCV::NoRegister) {
    // We can only use x0, x0 if there's no chance of the vtype change causing
    // the previous vl to become invalid.
    if (PrevInfo.isValid() && !PrevInfo.isUnknown() &&
        Info.hasSameVLMAX(PrevInfo)) {
      BuildMI(MBB, MI, DL, TII->get(RISCV::PseudoVSETVLIX0))
          .addReg(RISCV::X0, RegState::Define | RegState::Dead)
          .addReg(RISCV::X0, RegState::Kill)
          .addImm(Info.encodeVTYPE())
          .addReg(RISCV::VL, RegState::Implicit);
      return;
    }
    // Otherwise use an AVL of 0 to avoid depending on previous vl.
    BuildMI(MBB, MI, DL, TII->get(RISCV::PseudoVSETIVLI))
        .addReg(RISCV::X0, RegState::Define | RegState::Dead)
        .addImm(0)
        .addImm(Info.encodeVTYPE());
    return;
  }

  if (AVLReg.isVirtual())
    MRI->constrainRegClass(AVLReg, &RISCV::GPRNoX0RegClass);

  // Use X0 as the DestReg unless AVLReg is X0. We also need to change the
  // opcode if the AVLReg is X0 as they have different register classes for
  // the AVL operand.
  Register DestReg = RISCV::X0;
  unsigned Opcode = RISCV::PseudoVSETVLI;
  if (AVLReg == RISCV::X0) {
    DestReg = MRI->createVirtualRegister(&RISCV::GPRRegClass);
    Opcode = RISCV::PseudoVSETVLIX0;
  }
  BuildMI(MBB, MI, DL, TII->get(Opcode))
      .addReg(DestReg, RegState::Define | RegState::Dead)
      .addReg(AVLReg)
      .addImm(Info.encodeVTYPE());
}

// Return a VSETVLIInfo representing the changes made by this VSETVLI or
// VSETIVLI instruction.
static VSETVLIInfo getInfoForVSETVLI(const MachineInstr &MI) {
  VSETVLIInfo NewInfo;
  if (MI.getOpcode() == RISCV::PseudoVSETIVLI) {
    NewInfo.setAVLImm(MI.getOperand(1).getImm());
  } else {
    assert(MI.getOpcode() == RISCV::PseudoVSETVLI ||
           MI.getOpcode() == RISCV::PseudoVSETVLIX0);
    Register AVLReg = MI.getOperand(1).getReg();
    assert((AVLReg != RISCV::X0 || MI.getOperand(0).getReg() != RISCV::X0) &&
           "Can't handle X0, X0 vsetvli yet");
    NewInfo.setAVLReg(AVLReg);
  }
  NewInfo.setVTYPE(MI.getOperand(2).getImm());

  return NewInfo;
}

bool RISCVInsertVSETVLI::needVSETVLI(const VSETVLIInfo &Require,
                                     const VSETVLIInfo &CurInfo) {
  if (CurInfo.isCompatible(Require, /*Strict*/ false))
    return false;

  // We didn't find a compatible value. If our AVL is a virtual register,
  // it might be defined by a VSET(I)VLI. If it has the same VTYPE we need
  // and the last VL/VTYPE we observed is the same, we don't need a
  // VSETVLI here.
  if (!CurInfo.isUnknown() && Require.hasAVLReg() &&
      Require.getAVLReg().isVirtual() && !CurInfo.hasSEWLMULRatioOnly() &&
      Require.hasSameVTYPE(CurInfo)) {
    if (MachineInstr *DefMI = MRI->getVRegDef(Require.getAVLReg())) {
      if (DefMI->getOpcode() == RISCV::PseudoVSETVLI ||
          DefMI->getOpcode() == RISCV::PseudoVSETVLIX0 ||
          DefMI->getOpcode() == RISCV::PseudoVSETIVLI) {
        VSETVLIInfo DefInfo = getInfoForVSETVLI(*DefMI);
        if (DefInfo.hasSameAVL(CurInfo) && DefInfo.hasSameVTYPE(CurInfo))
          return false;
      }
    }
  }

  return true;
}

bool canSkipVSETVLIForLoadStore(const MachineInstr &MI,
                                const VSETVLIInfo &Require,
                                const VSETVLIInfo &CurInfo) {
  unsigned EEW;
  switch (MI.getOpcode()) {
  default:
    return false;
  case RISCV::PseudoVLE8_V_M1:
  case RISCV::PseudoVLE8_V_M1_MASK:
  case RISCV::PseudoVLE8_V_M2:
  case RISCV::PseudoVLE8_V_M2_MASK:
  case RISCV::PseudoVLE8_V_M4:
  case RISCV::PseudoVLE8_V_M4_MASK:
  case RISCV::PseudoVLE8_V_M8:
  case RISCV::PseudoVLE8_V_M8_MASK:
  case RISCV::PseudoVLE8_V_MF2:
  case RISCV::PseudoVLE8_V_MF2_MASK:
  case RISCV::PseudoVLE8_V_MF4:
  case RISCV::PseudoVLE8_V_MF4_MASK:
  case RISCV::PseudoVLE8_V_MF8:
  case RISCV::PseudoVLE8_V_MF8_MASK:
  case RISCV::PseudoVLSE8_V_M1:
  case RISCV::PseudoVLSE8_V_M1_MASK:
  case RISCV::PseudoVLSE8_V_M2:
  case RISCV::PseudoVLSE8_V_M2_MASK:
  case RISCV::PseudoVLSE8_V_M4:
  case RISCV::PseudoVLSE8_V_M4_MASK:
  case RISCV::PseudoVLSE8_V_M8:
  case RISCV::PseudoVLSE8_V_M8_MASK:
  case RISCV::PseudoVLSE8_V_MF2:
  case RISCV::PseudoVLSE8_V_MF2_MASK:
  case RISCV::PseudoVLSE8_V_MF4:
  case RISCV::PseudoVLSE8_V_MF4_MASK:
  case RISCV::PseudoVLSE8_V_MF8:
  case RISCV::PseudoVLSE8_V_MF8_MASK:
  case RISCV::PseudoVSE8_V_M1:
  case RISCV::PseudoVSE8_V_M1_MASK:
  case RISCV::PseudoVSE8_V_M2:
  case RISCV::PseudoVSE8_V_M2_MASK:
  case RISCV::PseudoVSE8_V_M4:
  case RISCV::PseudoVSE8_V_M4_MASK:
  case RISCV::PseudoVSE8_V_M8:
  case RISCV::PseudoVSE8_V_M8_MASK:
  case RISCV::PseudoVSE8_V_MF2:
  case RISCV::PseudoVSE8_V_MF2_MASK:
  case RISCV::PseudoVSE8_V_MF4:
  case RISCV::PseudoVSE8_V_MF4_MASK:
  case RISCV::PseudoVSE8_V_MF8:
  case RISCV::PseudoVSE8_V_MF8_MASK:
  case RISCV::PseudoVSSE8_V_M1:
  case RISCV::PseudoVSSE8_V_M1_MASK:
  case RISCV::PseudoVSSE8_V_M2:
  case RISCV::PseudoVSSE8_V_M2_MASK:
  case RISCV::PseudoVSSE8_V_M4:
  case RISCV::PseudoVSSE8_V_M4_MASK:
  case RISCV::PseudoVSSE8_V_M8:
  case RISCV::PseudoVSSE8_V_M8_MASK:
  case RISCV::PseudoVSSE8_V_MF2:
  case RISCV::PseudoVSSE8_V_MF2_MASK:
  case RISCV::PseudoVSSE8_V_MF4:
  case RISCV::PseudoVSSE8_V_MF4_MASK:
  case RISCV::PseudoVSSE8_V_MF8:
  case RISCV::PseudoVSSE8_V_MF8_MASK:
    EEW = 8;
    break;
  case RISCV::PseudoVLE16_V_M1:
  case RISCV::PseudoVLE16_V_M1_MASK:
  case RISCV::PseudoVLE16_V_M2:
  case RISCV::PseudoVLE16_V_M2_MASK:
  case RISCV::PseudoVLE16_V_M4:
  case RISCV::PseudoVLE16_V_M4_MASK:
  case RISCV::PseudoVLE16_V_M8:
  case RISCV::PseudoVLE16_V_M8_MASK:
  case RISCV::PseudoVLE16_V_MF2:
  case RISCV::PseudoVLE16_V_MF2_MASK:
  case RISCV::PseudoVLE16_V_MF4:
  case RISCV::PseudoVLE16_V_MF4_MASK:
  case RISCV::PseudoVLSE16_V_M1:
  case RISCV::PseudoVLSE16_V_M1_MASK:
  case RISCV::PseudoVLSE16_V_M2:
  case RISCV::PseudoVLSE16_V_M2_MASK:
  case RISCV::PseudoVLSE16_V_M4:
  case RISCV::PseudoVLSE16_V_M4_MASK:
  case RISCV::PseudoVLSE16_V_M8:
  case RISCV::PseudoVLSE16_V_M8_MASK:
  case RISCV::PseudoVLSE16_V_MF2:
  case RISCV::PseudoVLSE16_V_MF2_MASK:
  case RISCV::PseudoVLSE16_V_MF4:
  case RISCV::PseudoVLSE16_V_MF4_MASK:
  case RISCV::PseudoVSE16_V_M1:
  case RISCV::PseudoVSE16_V_M1_MASK:
  case RISCV::PseudoVSE16_V_M2:
  case RISCV::PseudoVSE16_V_M2_MASK:
  case RISCV::PseudoVSE16_V_M4:
  case RISCV::PseudoVSE16_V_M4_MASK:
  case RISCV::PseudoVSE16_V_M8:
  case RISCV::PseudoVSE16_V_M8_MASK:
  case RISCV::PseudoVSE16_V_MF2:
  case RISCV::PseudoVSE16_V_MF2_MASK:
  case RISCV::PseudoVSE16_V_MF4:
  case RISCV::PseudoVSE16_V_MF4_MASK:
  case RISCV::PseudoVSSE16_V_M1:
  case RISCV::PseudoVSSE16_V_M1_MASK:
  case RISCV::PseudoVSSE16_V_M2:
  case RISCV::PseudoVSSE16_V_M2_MASK:
  case RISCV::PseudoVSSE16_V_M4:
  case RISCV::PseudoVSSE16_V_M4_MASK:
  case RISCV::PseudoVSSE16_V_M8:
  case RISCV::PseudoVSSE16_V_M8_MASK:
  case RISCV::PseudoVSSE16_V_MF2:
  case RISCV::PseudoVSSE16_V_MF2_MASK:
  case RISCV::PseudoVSSE16_V_MF4:
  case RISCV::PseudoVSSE16_V_MF4_MASK:
    EEW = 16;
    break;
  case RISCV::PseudoVLE32_V_M1:
  case RISCV::PseudoVLE32_V_M1_MASK:
  case RISCV::PseudoVLE32_V_M2:
  case RISCV::PseudoVLE32_V_M2_MASK:
  case RISCV::PseudoVLE32_V_M4:
  case RISCV::PseudoVLE32_V_M4_MASK:
  case RISCV::PseudoVLE32_V_M8:
  case RISCV::PseudoVLE32_V_M8_MASK:
  case RISCV::PseudoVLE32_V_MF2:
  case RISCV::PseudoVLE32_V_MF2_MASK:
  case RISCV::PseudoVLSE32_V_M1:
  case RISCV::PseudoVLSE32_V_M1_MASK:
  case RISCV::PseudoVLSE32_V_M2:
  case RISCV::PseudoVLSE32_V_M2_MASK:
  case RISCV::PseudoVLSE32_V_M4:
  case RISCV::PseudoVLSE32_V_M4_MASK:
  case RISCV::PseudoVLSE32_V_M8:
  case RISCV::PseudoVLSE32_V_M8_MASK:
  case RISCV::PseudoVLSE32_V_MF2:
  case RISCV::PseudoVLSE32_V_MF2_MASK:
  case RISCV::PseudoVSE32_V_M1:
  case RISCV::PseudoVSE32_V_M1_MASK:
  case RISCV::PseudoVSE32_V_M2:
  case RISCV::PseudoVSE32_V_M2_MASK:
  case RISCV::PseudoVSE32_V_M4:
  case RISCV::PseudoVSE32_V_M4_MASK:
  case RISCV::PseudoVSE32_V_M8:
  case RISCV::PseudoVSE32_V_M8_MASK:
  case RISCV::PseudoVSE32_V_MF2:
  case RISCV::PseudoVSE32_V_MF2_MASK:
  case RISCV::PseudoVSSE32_V_M1:
  case RISCV::PseudoVSSE32_V_M1_MASK:
  case RISCV::PseudoVSSE32_V_M2:
  case RISCV::PseudoVSSE32_V_M2_MASK:
  case RISCV::PseudoVSSE32_V_M4:
  case RISCV::PseudoVSSE32_V_M4_MASK:
  case RISCV::PseudoVSSE32_V_M8:
  case RISCV::PseudoVSSE32_V_M8_MASK:
  case RISCV::PseudoVSSE32_V_MF2:
  case RISCV::PseudoVSSE32_V_MF2_MASK:
    EEW = 32;
    break;
  case RISCV::PseudoVLE64_V_M1:
  case RISCV::PseudoVLE64_V_M1_MASK:
  case RISCV::PseudoVLE64_V_M2:
  case RISCV::PseudoVLE64_V_M2_MASK:
  case RISCV::PseudoVLE64_V_M4:
  case RISCV::PseudoVLE64_V_M4_MASK:
  case RISCV::PseudoVLE64_V_M8:
  case RISCV::PseudoVLE64_V_M8_MASK:
  case RISCV::PseudoVLSE64_V_M1:
  case RISCV::PseudoVLSE64_V_M1_MASK:
  case RISCV::PseudoVLSE64_V_M2:
  case RISCV::PseudoVLSE64_V_M2_MASK:
  case RISCV::PseudoVLSE64_V_M4:
  case RISCV::PseudoVLSE64_V_M4_MASK:
  case RISCV::PseudoVLSE64_V_M8:
  case RISCV::PseudoVLSE64_V_M8_MASK:
  case RISCV::PseudoVSE64_V_M1:
  case RISCV::PseudoVSE64_V_M1_MASK:
  case RISCV::PseudoVSE64_V_M2:
  case RISCV::PseudoVSE64_V_M2_MASK:
  case RISCV::PseudoVSE64_V_M4:
  case RISCV::PseudoVSE64_V_M4_MASK:
  case RISCV::PseudoVSE64_V_M8:
  case RISCV::PseudoVSE64_V_M8_MASK:
  case RISCV::PseudoVSSE64_V_M1:
  case RISCV::PseudoVSSE64_V_M1_MASK:
  case RISCV::PseudoVSSE64_V_M2:
  case RISCV::PseudoVSSE64_V_M2_MASK:
  case RISCV::PseudoVSSE64_V_M4:
  case RISCV::PseudoVSSE64_V_M4_MASK:
  case RISCV::PseudoVSSE64_V_M8:
  case RISCV::PseudoVSSE64_V_M8_MASK:
    EEW = 64;
    break;
  }

  return CurInfo.isCompatibleWithLoadStoreEEW(EEW, Require);
}

bool RISCVInsertVSETVLI::computeVLVTYPEChanges(const MachineBasicBlock &MBB) {
  bool HadVectorOp = false;

  BlockData &BBInfo = BlockInfo[MBB.getNumber()];
  for (const MachineInstr &MI : MBB) {
    // If this is an explicit VSETVLI or VSETIVLI, update our state.
    if (MI.getOpcode() == RISCV::PseudoVSETVLI ||
        MI.getOpcode() == RISCV::PseudoVSETVLIX0 ||
        MI.getOpcode() == RISCV::PseudoVSETIVLI) {
      HadVectorOp = true;
      BBInfo.Change = getInfoForVSETVLI(MI);
      continue;
    }

    uint64_t TSFlags = MI.getDesc().TSFlags;
    if (RISCVII::hasSEWOp(TSFlags)) {
      HadVectorOp = true;

      VSETVLIInfo NewInfo = computeInfoForInstr(MI, TSFlags, MRI);

      if (!BBInfo.Change.isValid()) {
        BBInfo.Change = NewInfo;
      } else {
        // If this instruction isn't compatible with the previous VL/VTYPE
        // we need to insert a VSETVLI.
        // If this is a unit-stride or strided load/store, we may be able to use
        // the EMUL=(EEW/SEW)*LMUL relationship to avoid changing vtype.
        // NOTE: We only do this if the vtype we're comparing against was
        // created in this block. We need the first and third phase to treat
        // the store the same way.
        if (!canSkipVSETVLIForLoadStore(MI, NewInfo, BBInfo.Change) &&
            needVSETVLI(NewInfo, BBInfo.Change))
          BBInfo.Change = NewInfo;
      }
    }

    // If this is something that updates VL/VTYPE that we don't know about, set
    // the state to unknown.
    if (MI.isCall() || MI.isInlineAsm() || MI.modifiesRegister(RISCV::VL) ||
        MI.modifiesRegister(RISCV::VTYPE)) {
      BBInfo.Change = VSETVLIInfo::getUnknown();
    }
  }

  // Initial exit state is whatever change we found in the block.
  BBInfo.Exit = BBInfo.Change;

  return HadVectorOp;
}

void RISCVInsertVSETVLI::computeIncomingVLVTYPE(const MachineBasicBlock &MBB) {
  BlockData &BBInfo = BlockInfo[MBB.getNumber()];

  BBInfo.InQueue = false;

  VSETVLIInfo InInfo;
  if (MBB.pred_empty()) {
    // There are no predecessors, so use the default starting status.
    InInfo.setUnknown();
  } else {
    for (MachineBasicBlock *P : MBB.predecessors())
      InInfo = InInfo.intersect(BlockInfo[P->getNumber()].Exit);
  }

  // If we don't have any valid predecessor value, wait until we do.
  if (!InInfo.isValid())
    return;

  BBInfo.Pred = InInfo;

  VSETVLIInfo TmpStatus = BBInfo.Pred.merge(BBInfo.Change);

  // If the new exit value matches the old exit value, we don't need to revisit
  // any blocks.
  if (BBInfo.Exit == TmpStatus)
    return;

  BBInfo.Exit = TmpStatus;

  // Add the successors to the work list so we can propagate the changed exit
  // status.
  for (MachineBasicBlock *S : MBB.successors())
    if (!BlockInfo[S->getNumber()].InQueue)
      WorkList.push(S);
}

// If we weren't able to prove a vsetvli was directly unneeded, it might still
// be/ unneeded if the AVL is a phi node where all incoming values are VL
// outputs from the last VSETVLI in their respective basic blocks.
bool RISCVInsertVSETVLI::needVSETVLIPHI(const VSETVLIInfo &Require,
                                        const MachineBasicBlock &MBB) {
  if (DisableInsertVSETVLPHIOpt)
    return true;

  if (!Require.hasAVLReg())
    return true;

  Register AVLReg = Require.getAVLReg();
  if (!AVLReg.isVirtual())
    return true;

  // We need the AVL to be produce by a PHI node in this basic block.
  MachineInstr *PHI = MRI->getVRegDef(AVLReg);
  if (!PHI || PHI->getOpcode() != RISCV::PHI || PHI->getParent() != &MBB)
    return true;

  for (unsigned PHIOp = 1, NumOps = PHI->getNumOperands(); PHIOp != NumOps;
       PHIOp += 2) {
    Register InReg = PHI->getOperand(PHIOp).getReg();
    MachineBasicBlock *PBB = PHI->getOperand(PHIOp + 1).getMBB();
    const BlockData &PBBInfo = BlockInfo[PBB->getNumber()];
    // If the exit from the predecessor has the VTYPE we are looking for
    // we might be able to avoid a VSETVLI.
    if (PBBInfo.Exit.isUnknown() || !PBBInfo.Exit.hasSameVTYPE(Require))
      return true;

    // We need the PHI input to the be the output of a VSET(I)VLI.
    MachineInstr *DefMI = MRI->getVRegDef(InReg);
    if (!DefMI || (DefMI->getOpcode() != RISCV::PseudoVSETVLI &&
                   DefMI->getOpcode() != RISCV::PseudoVSETVLIX0 &&
                   DefMI->getOpcode() != RISCV::PseudoVSETIVLI))
      return true;

    // We found a VSET(I)VLI make sure it matches the output of the
    // predecessor block.
    VSETVLIInfo DefInfo = getInfoForVSETVLI(*DefMI);
    if (!DefInfo.hasSameAVL(PBBInfo.Exit) ||
        !DefInfo.hasSameVTYPE(PBBInfo.Exit))
      return true;
  }

  // If all the incoming values to the PHI checked out, we don't need
  // to insert a VSETVLI.
  return false;
}

void RISCVInsertVSETVLI::emitVSETVLIs(MachineBasicBlock &MBB) {
  VSETVLIInfo CurInfo;
  // Only be set if current VSETVLIInfo is from an explicit VSET(I)VLI.
  MachineInstr *PrevVSETVLIMI = nullptr;

  for (MachineInstr &MI : MBB) {
    // If this is an explicit VSETVLI or VSETIVLI, update our state.
    if (MI.getOpcode() == RISCV::PseudoVSETVLI ||
        MI.getOpcode() == RISCV::PseudoVSETVLIX0 ||
        MI.getOpcode() == RISCV::PseudoVSETIVLI) {
      // Conservatively, mark the VL and VTYPE as live.
      assert(MI.getOperand(3).getReg() == RISCV::VL &&
             MI.getOperand(4).getReg() == RISCV::VTYPE &&
             "Unexpected operands where VL and VTYPE should be");
      MI.getOperand(3).setIsDead(false);
      MI.getOperand(4).setIsDead(false);
      CurInfo = getInfoForVSETVLI(MI);
      PrevVSETVLIMI = &MI;
      continue;
    }

    uint64_t TSFlags = MI.getDesc().TSFlags;
    if (RISCVII::hasSEWOp(TSFlags)) {
      VSETVLIInfo NewInfo = computeInfoForInstr(MI, TSFlags, MRI);
      if (RISCVII::hasVLOp(TSFlags)) {
        unsigned Offset = 2;
        if (RISCVII::hasVecPolicyOp(TSFlags))
          Offset = 3;
        MachineOperand &VLOp =
            MI.getOperand(MI.getNumExplicitOperands() - Offset);
        if (VLOp.isReg()) {
          // Erase the AVL operand from the instruction.
          VLOp.setReg(RISCV::NoRegister);
          VLOp.setIsKill(false);
        }
        MI.addOperand(MachineOperand::CreateReg(RISCV::VL, /*isDef*/ false,
                                                /*isImp*/ true));
      }
      MI.addOperand(MachineOperand::CreateReg(RISCV::VTYPE, /*isDef*/ false,
                                              /*isImp*/ true));

      if (!CurInfo.isValid()) {
        // We haven't found any vector instructions or VL/VTYPE changes yet,
        // use the predecessor information.
        assert(BlockInfo[MBB.getNumber()].Pred.isValid() &&
               "Expected a valid predecessor state.");
        if (needVSETVLI(NewInfo, BlockInfo[MBB.getNumber()].Pred) &&
            needVSETVLIPHI(NewInfo, MBB)) {
          insertVSETVLI(MBB, MI, NewInfo, BlockInfo[MBB.getNumber()].Pred);
          CurInfo = NewInfo;
        }
      } else {
        // If this instruction isn't compatible with the previous VL/VTYPE
        // we need to insert a VSETVLI.
        // If this is a unit-stride or strided load/store, we may be able to use
        // the EMUL=(EEW/SEW)*LMUL relationship to avoid changing vtype.
        // NOTE: We can't use predecessor information for the store. We must
        // treat it the same as the first phase so that we produce the correct
        // vl/vtype for succesor blocks.
        if (!canSkipVSETVLIForLoadStore(MI, NewInfo, CurInfo) &&
            needVSETVLI(NewInfo, CurInfo)) {
          // If the previous VL/VTYPE is set by VSETVLI and do not use, Merge it
          // with current VL/VTYPE.
          bool NeedInsertVSETVLI = true;
          if (PrevVSETVLIMI) {
            bool HasSameAVL =
                CurInfo.hasSameAVL(NewInfo) ||
                (NewInfo.hasAVLReg() && NewInfo.getAVLReg().isVirtual() &&
                 NewInfo.getAVLReg() == PrevVSETVLIMI->getOperand(0).getReg());
            // If these two VSETVLI have the same AVL and the same VLMAX,
            // we could merge these two VSETVLI.
            if (HasSameAVL &&
                CurInfo.getSEWLMULRatio() == NewInfo.getSEWLMULRatio()) {
              PrevVSETVLIMI->getOperand(2).setImm(NewInfo.encodeVTYPE());
              NeedInsertVSETVLI = false;
            }
          }
          if (NeedInsertVSETVLI)
            insertVSETVLI(MBB, MI, NewInfo, CurInfo);
          CurInfo = NewInfo;
        }
      }
      PrevVSETVLIMI = nullptr;
    }

    // If this is something updates VL/VTYPE that we don't know about, set
    // the state to unknown.
    if (MI.isCall() || MI.isInlineAsm() || MI.modifiesRegister(RISCV::VL) ||
        MI.modifiesRegister(RISCV::VTYPE)) {
      CurInfo = VSETVLIInfo::getUnknown();
      PrevVSETVLIMI = nullptr;
    }
  }
}

bool RISCVInsertVSETVLI::runOnMachineFunction(MachineFunction &MF) {
  // Skip if the vector extension is not enabled.
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasStdExtV())
    return false;

  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();

  assert(BlockInfo.empty() && "Expect empty block infos");
  BlockInfo.resize(MF.getNumBlockIDs());

  bool HaveVectorOp = false;

  // Phase 1 - determine how VL/VTYPE are affected by the each block.
  for (const MachineBasicBlock &MBB : MF)
    HaveVectorOp |= computeVLVTYPEChanges(MBB);

  // If we didn't find any instructions that need VSETVLI, we're done.
  if (HaveVectorOp) {
    // Phase 2 - determine the exit VL/VTYPE from each block. We add all
    // blocks to the list here, but will also add any that need to be revisited
    // during Phase 2 processing.
    for (const MachineBasicBlock &MBB : MF) {
      WorkList.push(&MBB);
      BlockInfo[MBB.getNumber()].InQueue = true;
    }
    while (!WorkList.empty()) {
      const MachineBasicBlock &MBB = *WorkList.front();
      WorkList.pop();
      computeIncomingVLVTYPE(MBB);
    }

    // Phase 3 - add any vsetvli instructions needed in the block. Use the
    // Phase 2 information to avoid adding vsetvlis before the first vector
    // instruction in the block if the VL/VTYPE is satisfied by its
    // predecessors.
    for (MachineBasicBlock &MBB : MF)
      emitVSETVLIs(MBB);
  }

  BlockInfo.clear();

  return HaveVectorOp;
}

/// Returns an instance of the Insert VSETVLI pass.
FunctionPass *llvm::createRISCVInsertVSETVLIPass() {
  return new RISCVInsertVSETVLI();
}
