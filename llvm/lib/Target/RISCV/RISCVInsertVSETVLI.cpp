//===- RISCVInsertVSETVLI.cpp - Insert VSETVLI instructions ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function pass that inserts VSETVLI instructions where
// needed and expands the vl outputs of VLEFF/VLSEGFF to PseudoReadVL
// instructions.
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
#define RISCV_INSERT_VSETVLI_NAME "RISC-V Insert VSETVLI pass"

static cl::opt<bool> DisableInsertVSETVLPHIOpt(
    "riscv-disable-insert-vsetvl-phi-opt", cl::init(false), cl::Hidden,
    cl::desc("Disable looking through phis when inserting vsetvlis."));

static cl::opt<bool> UseStrictAsserts(
    "riscv-insert-vsetvl-strict-asserts", cl::init(true), cl::Hidden,
    cl::desc("Enable strict assertion checking for the dataflow algorithm"));

namespace {

static unsigned getVLOpNum(const MachineInstr &MI) {
  return RISCVII::getVLOpNum(MI.getDesc());
}

static unsigned getSEWOpNum(const MachineInstr &MI) {
  return RISCVII::getSEWOpNum(MI.getDesc());
}

static bool isVectorConfigInstr(const MachineInstr &MI) {
  return MI.getOpcode() == RISCV::PseudoVSETVLI ||
         MI.getOpcode() == RISCV::PseudoVSETVLIX0 ||
         MI.getOpcode() == RISCV::PseudoVSETIVLI;
}

/// Return true if this is 'vsetvli x0, x0, vtype' which preserves
/// VL and only sets VTYPE.
static bool isVLPreservingConfig(const MachineInstr &MI) {
  if (MI.getOpcode() != RISCV::PseudoVSETVLIX0)
    return false;
  assert(RISCV::X0 == MI.getOperand(1).getReg());
  return RISCV::X0 == MI.getOperand(0).getReg();
}

static uint16_t getRVVMCOpcode(uint16_t RVVPseudoOpcode) {
  const RISCVVPseudosTable::PseudoInfo *RVV =
      RISCVVPseudosTable::getPseudoInfo(RVVPseudoOpcode);
  if (!RVV)
    return 0;
  return RVV->BaseInstr;
}

static bool isFloatScalarMoveOrScalarSplatInstr(const MachineInstr &MI) {
  switch (getRVVMCOpcode(MI.getOpcode())) {
  default:
    return false;
  case RISCV::VFMV_S_F:
  case RISCV::VFMV_V_F:
    return true;
  }
}

static bool isScalarExtractInstr(const MachineInstr &MI) {
  switch (getRVVMCOpcode(MI.getOpcode())) {
  default:
    return false;
  case RISCV::VMV_X_S:
  case RISCV::VFMV_F_S:
    return true;
  }
}

static bool isScalarInsertInstr(const MachineInstr &MI) {
  switch (getRVVMCOpcode(MI.getOpcode())) {
  default:
    return false;
  case RISCV::VMV_S_X:
  case RISCV::VFMV_S_F:
    return true;
  }
}

static bool isScalarSplatInstr(const MachineInstr &MI) {
  switch (getRVVMCOpcode(MI.getOpcode())) {
  default:
    return false;
  case RISCV::VMV_V_I:
  case RISCV::VMV_V_X:
  case RISCV::VFMV_V_F:
    return true;
  }
}

static bool isVSlideInstr(const MachineInstr &MI) {
  switch (getRVVMCOpcode(MI.getOpcode())) {
  default:
    return false;
  case RISCV::VSLIDEDOWN_VX:
  case RISCV::VSLIDEDOWN_VI:
  case RISCV::VSLIDEUP_VX:
  case RISCV::VSLIDEUP_VI:
    return true;
  }
}

/// Get the EEW for a load or store instruction.  Return std::nullopt if MI is
/// not a load or store which ignores SEW.
static std::optional<unsigned> getEEWForLoadStore(const MachineInstr &MI) {
  switch (getRVVMCOpcode(MI.getOpcode())) {
  default:
    return std::nullopt;
  case RISCV::VLE8_V:
  case RISCV::VLSE8_V:
  case RISCV::VSE8_V:
  case RISCV::VSSE8_V:
    return 8;
  case RISCV::VLE16_V:
  case RISCV::VLSE16_V:
  case RISCV::VSE16_V:
  case RISCV::VSSE16_V:
    return 16;
  case RISCV::VLE32_V:
  case RISCV::VLSE32_V:
  case RISCV::VSE32_V:
  case RISCV::VSSE32_V:
    return 32;
  case RISCV::VLE64_V:
  case RISCV::VLSE64_V:
  case RISCV::VSE64_V:
  case RISCV::VSSE64_V:
    return 64;
  }
}

/// Return true if this is an operation on mask registers.  Note that
/// this includes both arithmetic/logical ops and load/store (vlm/vsm).
static bool isMaskRegOp(const MachineInstr &MI) {
  if (!RISCVII::hasSEWOp(MI.getDesc().TSFlags))
    return false;
  const unsigned Log2SEW = MI.getOperand(getSEWOpNum(MI)).getImm();
  // A Log2SEW of 0 is an operation on mask registers only.
  return Log2SEW == 0;
}

/// Return true if the inactive elements in the result are entirely undefined.
/// Note that this is different from "agnostic" as defined by the vector
/// specification.  Agnostic requires each lane to either be undisturbed, or
/// take the value -1; no other value is allowed.
static bool hasUndefinedMergeOp(const MachineInstr &MI,
                                const MachineRegisterInfo &MRI) {

  unsigned UseOpIdx;
  if (!MI.isRegTiedToUseOperand(0, &UseOpIdx))
    // If there is no passthrough operand, then the pass through
    // lanes are undefined.
    return true;

  // If the tied operand is NoReg, an IMPLICIT_DEF, or a REG_SEQEUENCE whose
  // operands are solely IMPLICIT_DEFS, then the pass through lanes are
  // undefined.
  const MachineOperand &UseMO = MI.getOperand(UseOpIdx);
  if (UseMO.getReg() == RISCV::NoRegister)
    return true;

  if (MachineInstr *UseMI = MRI.getVRegDef(UseMO.getReg())) {
    if (UseMI->isImplicitDef())
      return true;

    if (UseMI->isRegSequence()) {
      for (unsigned i = 1, e = UseMI->getNumOperands(); i < e; i += 2) {
        MachineInstr *SourceMI = MRI.getVRegDef(UseMI->getOperand(i).getReg());
        if (!SourceMI || !SourceMI->isImplicitDef())
          return false;
      }
      return true;
    }
  }
  return false;
}

/// Which subfields of VL or VTYPE have values we need to preserve?
struct DemandedFields {
  // Some unknown property of VL is used.  If demanded, must preserve entire
  // value.
  bool VLAny = false;
  // Only zero vs non-zero is used. If demanded, can change non-zero values.
  bool VLZeroness = false;
  // What properties of SEW we need to preserve.
  enum : uint8_t {
    SEWEqual = 2,              // The exact value of SEW needs to be preserved.
    SEWGreaterThanOrEqual = 1, // SEW can be changed as long as it's greater
                               // than or equal to the original value.
    SEWNone = 0                // We don't need to preserve SEW at all.
  } SEW = SEWNone;
  bool LMUL = false;
  bool SEWLMULRatio = false;
  bool TailPolicy = false;
  bool MaskPolicy = false;

  // Return true if any part of VTYPE was used
  bool usedVTYPE() const {
    return SEW || LMUL || SEWLMULRatio || TailPolicy || MaskPolicy;
  }

  // Return true if any property of VL was used
  bool usedVL() {
    return VLAny || VLZeroness;
  }

  // Mark all VTYPE subfields and properties as demanded
  void demandVTYPE() {
    SEW = SEWEqual;
    LMUL = true;
    SEWLMULRatio = true;
    TailPolicy = true;
    MaskPolicy = true;
  }

  // Mark all VL properties as demanded
  void demandVL() {
    VLAny = true;
    VLZeroness = true;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Support for debugging, callable in GDB: V->dump()
  LLVM_DUMP_METHOD void dump() const {
    print(dbgs());
    dbgs() << "\n";
  }

  /// Implement operator<<.
  void print(raw_ostream &OS) const {
    OS << "{";
    OS << "VLAny=" << VLAny << ", ";
    OS << "VLZeroness=" << VLZeroness << ", ";
    OS << "SEW=";
    switch (SEW) {
    case SEWEqual:
      OS << "SEWEqual";
      break;
    case SEWGreaterThanOrEqual:
      OS << "SEWGreaterThanOrEqual";
      break;
    case SEWNone:
      OS << "SEWNone";
      break;
    };
    OS << ", ";
    OS << "LMUL=" << LMUL << ", ";
    OS << "SEWLMULRatio=" << SEWLMULRatio << ", ";
    OS << "TailPolicy=" << TailPolicy << ", ";
    OS << "MaskPolicy=" << MaskPolicy;
    OS << "}";
  }
#endif
};

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_ATTRIBUTE_USED
inline raw_ostream &operator<<(raw_ostream &OS, const DemandedFields &DF) {
  DF.print(OS);
  return OS;
}
#endif

/// Return true if moving from CurVType to NewVType is
/// indistinguishable from the perspective of an instruction (or set
/// of instructions) which use only the Used subfields and properties.
static bool areCompatibleVTYPEs(uint64_t CurVType, uint64_t NewVType,
                                const DemandedFields &Used) {
  if (Used.SEW == DemandedFields::SEWEqual &&
      RISCVVType::getSEW(CurVType) != RISCVVType::getSEW(NewVType))
    return false;

  if (Used.SEW == DemandedFields::SEWGreaterThanOrEqual &&
      RISCVVType::getSEW(NewVType) < RISCVVType::getSEW(CurVType))
    return false;

  if (Used.LMUL &&
      RISCVVType::getVLMUL(CurVType) != RISCVVType::getVLMUL(NewVType))
    return false;

  if (Used.SEWLMULRatio) {
    auto Ratio1 = RISCVVType::getSEWLMULRatio(RISCVVType::getSEW(CurVType),
                                              RISCVVType::getVLMUL(CurVType));
    auto Ratio2 = RISCVVType::getSEWLMULRatio(RISCVVType::getSEW(NewVType),
                                              RISCVVType::getVLMUL(NewVType));
    if (Ratio1 != Ratio2)
      return false;
  }

  if (Used.TailPolicy && RISCVVType::isTailAgnostic(CurVType) !=
                             RISCVVType::isTailAgnostic(NewVType))
    return false;
  if (Used.MaskPolicy && RISCVVType::isMaskAgnostic(CurVType) !=
                             RISCVVType::isMaskAgnostic(NewVType))
    return false;
  return true;
}

/// Return the fields and properties demanded by the provided instruction.
DemandedFields getDemanded(const MachineInstr &MI,
                           const MachineRegisterInfo *MRI) {
  // Warning: This function has to work on both the lowered (i.e. post
  // emitVSETVLIs) and pre-lowering forms.  The main implication of this is
  // that it can't use the value of a SEW, VL, or Policy operand as they might
  // be stale after lowering.
  bool HasVInstructionsF64 =
      MI.getMF()->getSubtarget<RISCVSubtarget>().hasVInstructionsF64();

  // Most instructions don't use any of these subfeilds.
  DemandedFields Res;
  // Start conservative if registers are used
  if (MI.isCall() || MI.isInlineAsm() || MI.readsRegister(RISCV::VL))
    Res.demandVL();
  if (MI.isCall() || MI.isInlineAsm() || MI.readsRegister(RISCV::VTYPE))
    Res.demandVTYPE();
  // Start conservative on the unlowered form too
  uint64_t TSFlags = MI.getDesc().TSFlags;
  if (RISCVII::hasSEWOp(TSFlags)) {
    Res.demandVTYPE();
    if (RISCVII::hasVLOp(TSFlags))
      Res.demandVL();

    // Behavior is independent of mask policy.
    if (!RISCVII::usesMaskPolicy(TSFlags))
      Res.MaskPolicy = false;
  }

  // Loads and stores with implicit EEW do not demand SEW or LMUL directly.
  // They instead demand the ratio of the two which is used in computing
  // EMUL, but which allows us the flexibility to change SEW and LMUL
  // provided we don't change the ratio.
  // Note: We assume that the instructions initial SEW is the EEW encoded
  // in the opcode.  This is asserted when constructing the VSETVLIInfo.
  if (getEEWForLoadStore(MI)) {
    Res.SEW = DemandedFields::SEWNone;
    Res.LMUL = false;
  }

  // Store instructions don't use the policy fields.
  if (RISCVII::hasSEWOp(TSFlags) && MI.getNumExplicitDefs() == 0) {
    Res.TailPolicy = false;
    Res.MaskPolicy = false;
  }

  // If this is a mask reg operation, it only cares about VLMAX.
  // TODO: Possible extensions to this logic
  // * Probably ok if available VLMax is larger than demanded
  // * The policy bits can probably be ignored..
  if (isMaskRegOp(MI)) {
    Res.SEW = DemandedFields::SEWNone;
    Res.LMUL = false;
  }

  // For vmv.s.x and vfmv.s.f, there are only two behaviors, VL = 0 and VL > 0.
  if (isScalarInsertInstr(MI)) {
    Res.LMUL = false;
    Res.SEWLMULRatio = false;
    Res.VLAny = false;
    // For vmv.s.x and vfmv.s.f, if the merge operand is *undefined*, we don't
    // need to preserve any other bits and are thus compatible with any larger,
    // etype and can disregard policy bits.  Warning: It's tempting to try doing
    // this for any tail agnostic operation, but we can't as TA requires
    // tail lanes to either be the original value or -1.  We are writing
    // unknown bits to the lanes here.
    if (hasUndefinedMergeOp(MI, *MRI)) {
      if (!isFloatScalarMoveOrScalarSplatInstr(MI) || HasVInstructionsF64)
        Res.SEW = DemandedFields::SEWGreaterThanOrEqual;
      Res.TailPolicy = false;
    }
  }

  // vmv.x.s, and vmv.f.s are unconditional and ignore everything except SEW.
  if (isScalarExtractInstr(MI)) {
    assert(!RISCVII::hasVLOp(TSFlags));
    Res.LMUL = false;
    Res.SEWLMULRatio = false;
    Res.TailPolicy = false;
    Res.MaskPolicy = false;
  }

  return Res;
}

/// Defines the abstract state with which the forward dataflow models the
/// values of the VL and VTYPE registers after insertion.
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
  uint8_t SEWLMULRatioOnly : 1;

public:
  VSETVLIInfo()
      : AVLImm(0), TailAgnostic(false), MaskAgnostic(false),
        SEWLMULRatioOnly(false) {}

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

  unsigned getSEW() const { return SEW; }
  RISCVII::VLMUL getVLMUL() const { return VLMul; }

  bool hasNonZeroAVL(const MachineRegisterInfo &MRI) const {
    if (hasAVLImm())
      return getAVLImm() > 0;
    if (hasAVLReg()) {
      if (getAVLReg() == RISCV::X0)
        return true;
      if (MachineInstr *MI = MRI.getVRegDef(getAVLReg());
          MI && MI->getOpcode() == RISCV::ADDI &&
          MI->getOperand(1).isReg() && MI->getOperand(2).isImm() &&
          MI->getOperand(1).getReg() == RISCV::X0 &&
          MI->getOperand(2).getImm() != 0)
        return true;
      return false;
    }
    return false;
  }

  bool hasEquallyZeroAVL(const VSETVLIInfo &Other,
                         const MachineRegisterInfo &MRI) const {
    if (hasSameAVL(Other))
      return true;
    return (hasNonZeroAVL(MRI) && Other.hasNonZeroAVL(MRI));
  }

  bool hasSameAVL(const VSETVLIInfo &Other) const {
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
  void setVTYPE(RISCVII::VLMUL L, unsigned S, bool TA, bool MA) {
    assert(isValid() && !isUnknown() &&
           "Can't set VTYPE for uninitialized or unknown");
    VLMul = L;
    SEW = S;
    TailAgnostic = TA;
    MaskAgnostic = MA;
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

  unsigned getSEWLMULRatio() const {
    assert(isValid() && !isUnknown() &&
           "Can't use VTYPE for uninitialized or unknown");
    return RISCVVType::getSEWLMULRatio(SEW, VLMul);
  }

  // Check if the VTYPE for these two VSETVLIInfos produce the same VLMAX.
  // Note that having the same VLMAX ensures that both share the same
  // function from AVL to VL; that is, they must produce the same VL value
  // for any given AVL value.
  bool hasSameVLMAX(const VSETVLIInfo &Other) const {
    assert(isValid() && Other.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!isUnknown() && !Other.isUnknown() &&
           "Can't compare VTYPE in unknown state");
    return getSEWLMULRatio() == Other.getSEWLMULRatio();
  }

  bool hasCompatibleVTYPE(const DemandedFields &Used,
                          const VSETVLIInfo &Require) const {
    return areCompatibleVTYPEs(Require.encodeVTYPE(), encodeVTYPE(), Used);
  }

  // Determine whether the vector instructions requirements represented by
  // Require are compatible with the previous vsetvli instruction represented
  // by this.  MI is the instruction whose requirements we're considering.
  bool isCompatible(const DemandedFields &Used, const VSETVLIInfo &Require,
                    const MachineRegisterInfo &MRI) const {
    assert(isValid() && Require.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!Require.SEWLMULRatioOnly &&
           "Expected a valid VTYPE for instruction!");
    // Nothing is compatible with Unknown.
    if (isUnknown() || Require.isUnknown())
      return false;

    // If only our VLMAX ratio is valid, then this isn't compatible.
    if (SEWLMULRatioOnly)
      return false;

    if (Used.VLAny && !hasSameAVL(Require))
      return false;

    if (Used.VLZeroness && !hasEquallyZeroAVL(Require, MRI))
      return false;

    return hasCompatibleVTYPE(Used, Require);
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

    // If the SEWLMULRatioOnly bits are different, then they aren't equal.
    if (SEWLMULRatioOnly != Other.SEWLMULRatioOnly)
      return false;

    // If only the VLMAX is valid, check that it is the same.
    if (SEWLMULRatioOnly)
      return hasSameVLMAX(Other);

    // If the full VTYPE is valid, check that it is the same.
    return hasSameVTYPE(Other);
  }

  bool operator!=(const VSETVLIInfo &Other) const {
    return !(*this == Other);
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

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Support for debugging, callable in GDB: V->dump()
  LLVM_DUMP_METHOD void dump() const {
    print(dbgs());
    dbgs() << "\n";
  }

  /// Implement operator<<.
  /// @{
  void print(raw_ostream &OS) const {
    OS << "{";
    if (!isValid())
      OS << "Uninitialized";
    if (isUnknown())
      OS << "unknown";
    if (hasAVLReg())
      OS << "AVLReg=" << (unsigned)AVLReg;
    if (hasAVLImm())
      OS << "AVLImm=" << (unsigned)AVLImm;
    OS << ", "
       << "VLMul=" << (unsigned)VLMul << ", "
       << "SEW=" << (unsigned)SEW << ", "
       << "TailAgnostic=" << (bool)TailAgnostic << ", "
       << "MaskAgnostic=" << (bool)MaskAgnostic << ", "
       << "SEWLMULRatioOnly=" << (bool)SEWLMULRatioOnly << "}";
  }
#endif
};

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_ATTRIBUTE_USED
inline raw_ostream &operator<<(raw_ostream &OS, const VSETVLIInfo &V) {
  V.print(OS);
  return OS;
}
#endif

struct BlockData {
  // The VSETVLIInfo that represents the VL/VTYPE settings on exit from this
  // block. Calculated in Phase 2.
  VSETVLIInfo Exit;

  // The VSETVLIInfo that represents the VL/VTYPE settings from all predecessor
  // blocks. Calculated in Phase 2, and used by Phase 3.
  VSETVLIInfo Pred;

  // Keeps track of whether the block is already in the queue.
  bool InQueue = false;

  BlockData() = default;
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
  bool needVSETVLI(const MachineInstr &MI, const VSETVLIInfo &Require,
                   const VSETVLIInfo &CurInfo) const;
  bool needVSETVLIPHI(const VSETVLIInfo &Require,
                      const MachineBasicBlock &MBB) const;
  void insertVSETVLI(MachineBasicBlock &MBB, MachineInstr &MI,
                     const VSETVLIInfo &Info, const VSETVLIInfo &PrevInfo);
  void insertVSETVLI(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator InsertPt, DebugLoc DL,
                     const VSETVLIInfo &Info, const VSETVLIInfo &PrevInfo);

  void transferBefore(VSETVLIInfo &Info, const MachineInstr &MI) const;
  void transferAfter(VSETVLIInfo &Info, const MachineInstr &MI) const;
  bool computeVLVTYPEChanges(const MachineBasicBlock &MBB,
                             VSETVLIInfo &Info) const;
  void computeIncomingVLVTYPE(const MachineBasicBlock &MBB);
  void emitVSETVLIs(MachineBasicBlock &MBB);
  void doLocalPostpass(MachineBasicBlock &MBB);
  void doPRE(MachineBasicBlock &MBB);
  void insertReadVL(MachineBasicBlock &MBB);
};

} // end anonymous namespace

char RISCVInsertVSETVLI::ID = 0;

INITIALIZE_PASS(RISCVInsertVSETVLI, DEBUG_TYPE, RISCV_INSERT_VSETVLI_NAME,
                false, false)

static VSETVLIInfo computeInfoForInstr(const MachineInstr &MI, uint64_t TSFlags,
                                       const MachineRegisterInfo *MRI) {
  VSETVLIInfo InstrInfo;

  bool TailAgnostic = true;
  bool MaskAgnostic = true;
  if (!hasUndefinedMergeOp(MI, *MRI)) {
    // Start with undisturbed.
    TailAgnostic = false;
    MaskAgnostic = false;

    // If there is a policy operand, use it.
    if (RISCVII::hasVecPolicyOp(TSFlags)) {
      const MachineOperand &Op = MI.getOperand(MI.getNumExplicitOperands() - 1);
      uint64_t Policy = Op.getImm();
      assert(Policy <= (RISCVII::TAIL_AGNOSTIC | RISCVII::MASK_AGNOSTIC) &&
             "Invalid Policy Value");
      TailAgnostic = Policy & RISCVII::TAIL_AGNOSTIC;
      MaskAgnostic = Policy & RISCVII::MASK_AGNOSTIC;
    }

    // Some pseudo instructions force a tail agnostic policy despite having a
    // tied def.
    if (RISCVII::doesForceTailAgnostic(TSFlags))
      TailAgnostic = true;

    if (!RISCVII::usesMaskPolicy(TSFlags))
      MaskAgnostic = true;
  }

  RISCVII::VLMUL VLMul = RISCVII::getLMul(TSFlags);

  unsigned Log2SEW = MI.getOperand(getSEWOpNum(MI)).getImm();
  // A Log2SEW of 0 is an operation on mask registers only.
  unsigned SEW = Log2SEW ? 1 << Log2SEW : 8;
  assert(RISCVVType::isValidSEW(SEW) && "Unexpected SEW");

  if (RISCVII::hasVLOp(TSFlags)) {
    const MachineOperand &VLOp = MI.getOperand(getVLOpNum(MI));
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
  } else {
    assert(isScalarExtractInstr(MI));
    InstrInfo.setAVLReg(RISCV::NoRegister);
  }
#ifndef NDEBUG
  if (std::optional<unsigned> EEW = getEEWForLoadStore(MI)) {
    assert(SEW == EEW && "Initial SEW doesn't match expected EEW");
  }
#endif
  InstrInfo.setVTYPE(VLMul, SEW, TailAgnostic, MaskAgnostic);

  return InstrInfo;
}

void RISCVInsertVSETVLI::insertVSETVLI(MachineBasicBlock &MBB, MachineInstr &MI,
                                       const VSETVLIInfo &Info,
                                       const VSETVLIInfo &PrevInfo) {
  DebugLoc DL = MI.getDebugLoc();
  insertVSETVLI(MBB, MachineBasicBlock::iterator(&MI), DL, Info, PrevInfo);
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

void RISCVInsertVSETVLI::insertVSETVLI(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator InsertPt, DebugLoc DL,
                     const VSETVLIInfo &Info, const VSETVLIInfo &PrevInfo) {

  if (PrevInfo.isValid() && !PrevInfo.isUnknown()) {
    // Use X0, X0 form if the AVL is the same and the SEW+LMUL gives the same
    // VLMAX.
    if (Info.hasSameAVL(PrevInfo) && Info.hasSameVLMAX(PrevInfo)) {
      BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETVLIX0))
          .addReg(RISCV::X0, RegState::Define | RegState::Dead)
          .addReg(RISCV::X0, RegState::Kill)
          .addImm(Info.encodeVTYPE())
          .addReg(RISCV::VL, RegState::Implicit);
      return;
    }

    // If our AVL is a virtual register, it might be defined by a VSET(I)VLI. If
    // it has the same VLMAX we want and the last VL/VTYPE we observed is the
    // same, we can use the X0, X0 form.
    if (Info.hasSameVLMAX(PrevInfo) && Info.hasAVLReg() &&
        Info.getAVLReg().isVirtual()) {
      if (MachineInstr *DefMI = MRI->getVRegDef(Info.getAVLReg())) {
        if (isVectorConfigInstr(*DefMI)) {
          VSETVLIInfo DefInfo = getInfoForVSETVLI(*DefMI);
          if (DefInfo.hasSameAVL(PrevInfo) && DefInfo.hasSameVLMAX(PrevInfo)) {
            BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETVLIX0))
                .addReg(RISCV::X0, RegState::Define | RegState::Dead)
                .addReg(RISCV::X0, RegState::Kill)
                .addImm(Info.encodeVTYPE())
                .addReg(RISCV::VL, RegState::Implicit);
            return;
          }
        }
      }
    }
  }

  if (Info.hasAVLImm()) {
    BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETIVLI))
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
      BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETVLIX0))
          .addReg(RISCV::X0, RegState::Define | RegState::Dead)
          .addReg(RISCV::X0, RegState::Kill)
          .addImm(Info.encodeVTYPE())
          .addReg(RISCV::VL, RegState::Implicit);
      return;
    }
    // Otherwise use an AVL of 1 to avoid depending on previous vl.
    BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETIVLI))
        .addReg(RISCV::X0, RegState::Define | RegState::Dead)
        .addImm(1)
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
  BuildMI(MBB, InsertPt, DL, TII->get(Opcode))
      .addReg(DestReg, RegState::Define | RegState::Dead)
      .addReg(AVLReg)
      .addImm(Info.encodeVTYPE());
}

static bool isLMUL1OrSmaller(RISCVII::VLMUL LMUL) {
  auto [LMul, Fractional] = RISCVVType::decodeVLMUL(LMUL);
  return Fractional || LMul == 1;
}

/// Return true if a VSETVLI is required to transition from CurInfo to Require
/// before MI.
bool RISCVInsertVSETVLI::needVSETVLI(const MachineInstr &MI,
                                     const VSETVLIInfo &Require,
                                     const VSETVLIInfo &CurInfo) const {
  assert(Require == computeInfoForInstr(MI, MI.getDesc().TSFlags, MRI));

  if (!CurInfo.isValid() || CurInfo.isUnknown() || CurInfo.hasSEWLMULRatioOnly())
    return true;

  DemandedFields Used = getDemanded(MI, MRI);
  bool HasVInstructionsF64 =
      MI.getMF()->getSubtarget<RISCVSubtarget>().hasVInstructionsF64();

  // A slidedown/slideup with an *undefined* merge op can freely clobber
  // elements not copied from the source vector (e.g. masked off, tail, or
  // slideup's prefix). Notes:
  // * We can't modify SEW here since the slide amount is in units of SEW.
  // * VL=1 is special only because we have existing support for zero vs
  //   non-zero VL.  We could generalize this if we had a VL > C predicate.
  // * The LMUL1 restriction is for machines whose latency may depend on VL.
  // * As above, this is only legal for tail "undefined" not "agnostic".
  if (isVSlideInstr(MI) && Require.hasAVLImm() && Require.getAVLImm() == 1 &&
      isLMUL1OrSmaller(CurInfo.getVLMUL()) && hasUndefinedMergeOp(MI, *MRI)) {
    Used.VLAny = false;
    Used.VLZeroness = true;
    Used.LMUL = false;
    Used.TailPolicy = false;
  }

  // A tail undefined vmv.v.i/x or vfmv.v.f with VL=1 can be treated in the same
  // semantically as vmv.s.x.  This is particularly useful since we don't have an
  // immediate form of vmv.s.x, and thus frequently use vmv.v.i in it's place.
  // Since a splat is non-constant time in LMUL, we do need to be careful to not
  // increase the number of active vector registers (unlike for vmv.s.x.)
  if (isScalarSplatInstr(MI) && Require.hasAVLImm() && Require.getAVLImm() == 1 &&
      isLMUL1OrSmaller(CurInfo.getVLMUL()) && hasUndefinedMergeOp(MI, *MRI)) {
    Used.LMUL = false;
    Used.SEWLMULRatio = false;
    Used.VLAny = false;
    if (!isFloatScalarMoveOrScalarSplatInstr(MI) || HasVInstructionsF64)
      Used.SEW = DemandedFields::SEWGreaterThanOrEqual;
    Used.TailPolicy = false;
  }

  if (CurInfo.isCompatible(Used, Require, *MRI))
    return false;

  // We didn't find a compatible value. If our AVL is a virtual register,
  // it might be defined by a VSET(I)VLI. If it has the same VLMAX we need
  // and the last VL/VTYPE we observed is the same, we don't need a
  // VSETVLI here.
  if (Require.hasAVLReg() && Require.getAVLReg().isVirtual() &&
      CurInfo.hasCompatibleVTYPE(Used, Require)) {
    if (MachineInstr *DefMI = MRI->getVRegDef(Require.getAVLReg())) {
      if (isVectorConfigInstr(*DefMI)) {
        VSETVLIInfo DefInfo = getInfoForVSETVLI(*DefMI);
        if (DefInfo.hasSameAVL(CurInfo) && DefInfo.hasSameVLMAX(CurInfo))
          return false;
      }
    }
  }

  return true;
}

// Given an incoming state reaching MI, modifies that state so that it is minimally
// compatible with MI.  The resulting state is guaranteed to be semantically legal
// for MI, but may not be the state requested by MI.
void RISCVInsertVSETVLI::transferBefore(VSETVLIInfo &Info,
                                        const MachineInstr &MI) const {
  uint64_t TSFlags = MI.getDesc().TSFlags;
  if (!RISCVII::hasSEWOp(TSFlags))
    return;

  const VSETVLIInfo NewInfo = computeInfoForInstr(MI, TSFlags, MRI);
  if (Info.isValid() && !needVSETVLI(MI, NewInfo, Info))
    return;

  const VSETVLIInfo PrevInfo = Info;
  Info = NewInfo;

  if (!RISCVII::hasVLOp(TSFlags))
    return;

  // For vmv.s.x and vfmv.s.f, there are only two behaviors, VL = 0 and
  // VL > 0. We can discard the user requested AVL and just use the last
  // one if we can prove it equally zero.  This removes a vsetvli entirely
  // if the types match or allows use of cheaper avl preserving variant
  // if VLMAX doesn't change.  If VLMAX might change, we couldn't use
  // the 'vsetvli x0, x0, vtype" variant, so we avoid the transform to
  // prevent extending live range of an avl register operand.
  // TODO: We can probably relax this for immediates.
  if (isScalarInsertInstr(MI) && PrevInfo.isValid() &&
      PrevInfo.hasEquallyZeroAVL(Info, *MRI) &&
      Info.hasSameVLMAX(PrevInfo)) {
    if (PrevInfo.hasAVLImm())
      Info.setAVLImm(PrevInfo.getAVLImm());
    else
      Info.setAVLReg(PrevInfo.getAVLReg());
    return;
  }

  // If AVL is defined by a vsetvli with the same VLMAX, we can
  // replace the AVL operand with the AVL of the defining vsetvli.
  // We avoid general register AVLs to avoid extending live ranges
  // without being sure we can kill the original source reg entirely.
  if (!Info.hasAVLReg() || !Info.getAVLReg().isVirtual())
    return;
  MachineInstr *DefMI = MRI->getVRegDef(Info.getAVLReg());
  if (!DefMI || !isVectorConfigInstr(*DefMI))
    return;

  VSETVLIInfo DefInfo = getInfoForVSETVLI(*DefMI);
  if (DefInfo.hasSameVLMAX(Info) &&
      (DefInfo.hasAVLImm() || DefInfo.getAVLReg() == RISCV::X0)) {
    if (DefInfo.hasAVLImm())
      Info.setAVLImm(DefInfo.getAVLImm());
    else
      Info.setAVLReg(DefInfo.getAVLReg());
    return;
  }
}

// Given a state with which we evaluated MI (see transferBefore above for why
// this might be different that the state MI requested), modify the state to
// reflect the changes MI might make.
void RISCVInsertVSETVLI::transferAfter(VSETVLIInfo &Info,
                                       const MachineInstr &MI) const {
  if (isVectorConfigInstr(MI)) {
    Info = getInfoForVSETVLI(MI);
    return;
  }

  if (RISCV::isFaultFirstLoad(MI)) {
    // Update AVL to vl-output of the fault first load.
    Info.setAVLReg(MI.getOperand(1).getReg());
    return;
  }

  // If this is something that updates VL/VTYPE that we don't know about, set
  // the state to unknown.
  if (MI.isCall() || MI.isInlineAsm() || MI.modifiesRegister(RISCV::VL) ||
      MI.modifiesRegister(RISCV::VTYPE))
    Info = VSETVLIInfo::getUnknown();
}

bool RISCVInsertVSETVLI::computeVLVTYPEChanges(const MachineBasicBlock &MBB,
                                               VSETVLIInfo &Info) const {
  bool HadVectorOp = false;

  Info = BlockInfo[MBB.getNumber()].Pred;
  for (const MachineInstr &MI : MBB) {
    transferBefore(Info, MI);

    if (isVectorConfigInstr(MI) || RISCVII::hasSEWOp(MI.getDesc().TSFlags))
      HadVectorOp = true;

    transferAfter(Info, MI);
  }

  return HadVectorOp;
}

void RISCVInsertVSETVLI::computeIncomingVLVTYPE(const MachineBasicBlock &MBB) {

  BlockData &BBInfo = BlockInfo[MBB.getNumber()];

  BBInfo.InQueue = false;

  // Start with the previous entry so that we keep the most conservative state
  // we have ever found.
  VSETVLIInfo InInfo = BBInfo.Pred;
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

  // If no change, no need to rerun block
  if (InInfo == BBInfo.Pred)
    return;

  BBInfo.Pred = InInfo;
  LLVM_DEBUG(dbgs() << "Entry state of " << printMBBReference(MBB)
                    << " changed to " << BBInfo.Pred << "\n");

  // Note: It's tempting to cache the state changes here, but due to the
  // compatibility checks performed a blocks output state can change based on
  // the input state.  To cache, we'd have to add logic for finding
  // never-compatible state changes.
  VSETVLIInfo TmpStatus;
  computeVLVTYPEChanges(MBB, TmpStatus);

  // If the new exit value matches the old exit value, we don't need to revisit
  // any blocks.
  if (BBInfo.Exit == TmpStatus)
    return;

  BBInfo.Exit = TmpStatus;
  LLVM_DEBUG(dbgs() << "Exit state of " << printMBBReference(MBB)
                    << " changed to " << BBInfo.Exit << "\n");

  // Add the successors to the work list so we can propagate the changed exit
  // status.
  for (MachineBasicBlock *S : MBB.successors())
    if (!BlockInfo[S->getNumber()].InQueue) {
      BlockInfo[S->getNumber()].InQueue = true;
      WorkList.push(S);
    }
}

// If we weren't able to prove a vsetvli was directly unneeded, it might still
// be unneeded if the AVL is a phi node where all incoming values are VL
// outputs from the last VSETVLI in their respective basic blocks.
bool RISCVInsertVSETVLI::needVSETVLIPHI(const VSETVLIInfo &Require,
                                        const MachineBasicBlock &MBB) const {
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
    if (!DefMI || !isVectorConfigInstr(*DefMI))
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
  VSETVLIInfo CurInfo = BlockInfo[MBB.getNumber()].Pred;
  // Track whether the prefix of the block we've scanned is transparent
  // (meaning has not yet changed the abstract state).
  bool PrefixTransparent = true;
  for (MachineInstr &MI : MBB) {
    const VSETVLIInfo PrevInfo = CurInfo;
    transferBefore(CurInfo, MI);

    // If this is an explicit VSETVLI or VSETIVLI, update our state.
    if (isVectorConfigInstr(MI)) {
      // Conservatively, mark the VL and VTYPE as live.
      assert(MI.getOperand(3).getReg() == RISCV::VL &&
             MI.getOperand(4).getReg() == RISCV::VTYPE &&
             "Unexpected operands where VL and VTYPE should be");
      MI.getOperand(3).setIsDead(false);
      MI.getOperand(4).setIsDead(false);
      PrefixTransparent = false;
    }

    uint64_t TSFlags = MI.getDesc().TSFlags;
    if (RISCVII::hasSEWOp(TSFlags)) {
      if (PrevInfo != CurInfo) {
        // If this is the first implicit state change, and the state change
        // requested can be proven to produce the same register contents, we
        // can skip emitting the actual state change and continue as if we
        // had since we know the GPR result of the implicit state change
        // wouldn't be used and VL/VTYPE registers are correct.  Note that
        // we *do* need to model the state as if it changed as while the
        // register contents are unchanged, the abstract model can change.
        if (!PrefixTransparent || needVSETVLIPHI(CurInfo, MBB))
          insertVSETVLI(MBB, MI, CurInfo, PrevInfo);
        PrefixTransparent = false;
      }

      if (RISCVII::hasVLOp(TSFlags)) {
        MachineOperand &VLOp = MI.getOperand(getVLOpNum(MI));
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
    }

    if (MI.isCall() || MI.isInlineAsm() || MI.modifiesRegister(RISCV::VL) ||
        MI.modifiesRegister(RISCV::VTYPE))
      PrefixTransparent = false;

    transferAfter(CurInfo, MI);
  }

  // If we reach the end of the block and our current info doesn't match the
  // expected info, insert a vsetvli to correct.
  if (!UseStrictAsserts) {
    const VSETVLIInfo &ExitInfo = BlockInfo[MBB.getNumber()].Exit;
    if (CurInfo.isValid() && ExitInfo.isValid() && !ExitInfo.isUnknown() &&
        CurInfo != ExitInfo) {
      // Note there's an implicit assumption here that terminators never use
      // or modify VL or VTYPE.  Also, fallthrough will return end().
      auto InsertPt = MBB.getFirstInstrTerminator();
      insertVSETVLI(MBB, InsertPt, MBB.findDebugLoc(InsertPt), ExitInfo,
                    CurInfo);
      CurInfo = ExitInfo;
    }
  }

  if (UseStrictAsserts && CurInfo.isValid()) {
    const auto &Info = BlockInfo[MBB.getNumber()];
    if (CurInfo != Info.Exit) {
      LLVM_DEBUG(dbgs() << "in block " << printMBBReference(MBB) << "\n");
      LLVM_DEBUG(dbgs() << "  begin        state: " << Info.Pred << "\n");
      LLVM_DEBUG(dbgs() << "  expected end state: " << Info.Exit << "\n");
      LLVM_DEBUG(dbgs() << "  actual   end state: " << CurInfo << "\n");
    }
    assert(CurInfo == Info.Exit &&
           "InsertVSETVLI dataflow invariant violated");
  }
}

/// Return true if the VL value configured must be equal to the requested one.
static bool hasFixedResult(const VSETVLIInfo &Info, const RISCVSubtarget &ST) {
  if (!Info.hasAVLImm())
    // VLMAX is always the same value.
    // TODO: Could extend to other registers by looking at the associated vreg
    // def placement.
    return RISCV::X0 == Info.getAVLReg();

  unsigned AVL = Info.getAVLImm();
  unsigned SEW = Info.getSEW();
  unsigned AVLInBits = AVL * SEW;

  unsigned LMul;
  bool Fractional;
  std::tie(LMul, Fractional) = RISCVVType::decodeVLMUL(Info.getVLMUL());

  if (Fractional)
    return ST.getRealMinVLen() / LMul >= AVLInBits;
  return ST.getRealMinVLen() * LMul >= AVLInBits;
}

/// Perform simple partial redundancy elimination of the VSETVLI instructions
/// we're about to insert by looking for cases where we can PRE from the
/// beginning of one block to the end of one of its predecessors.  Specifically,
/// this is geared to catch the common case of a fixed length vsetvl in a single
/// block loop when it could execute once in the preheader instead.
void RISCVInsertVSETVLI::doPRE(MachineBasicBlock &MBB) {
  const MachineFunction &MF = *MBB.getParent();
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();

  if (!BlockInfo[MBB.getNumber()].Pred.isUnknown())
    return;

  MachineBasicBlock *UnavailablePred = nullptr;
  VSETVLIInfo AvailableInfo;
  for (MachineBasicBlock *P : MBB.predecessors()) {
    const VSETVLIInfo &PredInfo = BlockInfo[P->getNumber()].Exit;
    if (PredInfo.isUnknown()) {
      if (UnavailablePred)
        return;
      UnavailablePred = P;
    } else if (!AvailableInfo.isValid()) {
      AvailableInfo = PredInfo;
    } else if (AvailableInfo != PredInfo) {
      return;
    }
  }

  // Unreachable, single pred, or full redundancy. Note that FRE is handled by
  // phase 3.
  if (!UnavailablePred || !AvailableInfo.isValid())
    return;

  // Critical edge - TODO: consider splitting?
  if (UnavailablePred->succ_size() != 1)
    return;

  // If VL can be less than AVL, then we can't reduce the frequency of exec.
  if (!hasFixedResult(AvailableInfo, ST))
    return;

  // Model the effect of changing the input state of the block MBB to
  // AvailableInfo.  We're looking for two issues here; one legality,
  // one profitability.
  // 1) If the block doesn't use some of the fields from VL or VTYPE, we
  //    may hit the end of the block with a different end state.  We can
  //    not make this change without reflowing later blocks as well.
  // 2) If we don't actually remove a transition, inserting a vsetvli
  //    into the predecessor block would be correct, but unprofitable.
  VSETVLIInfo OldInfo = BlockInfo[MBB.getNumber()].Pred;
  VSETVLIInfo CurInfo = AvailableInfo;
  int TransitionsRemoved = 0;
  for (const MachineInstr &MI : MBB) {
    const VSETVLIInfo LastInfo = CurInfo;
    const VSETVLIInfo LastOldInfo = OldInfo;
    transferBefore(CurInfo, MI);
    transferBefore(OldInfo, MI);
    if (CurInfo == LastInfo)
      TransitionsRemoved++;
    if (LastOldInfo == OldInfo)
      TransitionsRemoved--;
    transferAfter(CurInfo, MI);
    transferAfter(OldInfo, MI);
    if (CurInfo == OldInfo)
      // Convergence.  All transitions after this must match by construction.
      break;
  }
  if (CurInfo != OldInfo || TransitionsRemoved <= 0)
    // Issues 1 and 2 above
    return;

  // Finally, update both data flow state and insert the actual vsetvli.
  // Doing both keeps the code in sync with the dataflow results, which
  // is critical for correctness of phase 3.
  auto OldExit = BlockInfo[UnavailablePred->getNumber()].Exit;
  LLVM_DEBUG(dbgs() << "PRE VSETVLI from " << MBB.getName() << " to "
                    << UnavailablePred->getName() << " with state "
                    << AvailableInfo << "\n");
  BlockInfo[UnavailablePred->getNumber()].Exit = AvailableInfo;
  BlockInfo[MBB.getNumber()].Pred = AvailableInfo;

  // Note there's an implicit assumption here that terminators never use
  // or modify VL or VTYPE.  Also, fallthrough will return end().
  auto InsertPt = UnavailablePred->getFirstInstrTerminator();
  insertVSETVLI(*UnavailablePred, InsertPt,
                UnavailablePred->findDebugLoc(InsertPt),
                AvailableInfo, OldExit);
}

static void doUnion(DemandedFields &A, DemandedFields B) {
  A.VLAny |= B.VLAny;
  A.VLZeroness |= B.VLZeroness;
  A.SEW = std::max(A.SEW, B.SEW);
  A.LMUL |= B.LMUL;
  A.SEWLMULRatio |= B.SEWLMULRatio;
  A.TailPolicy |= B.TailPolicy;
  A.MaskPolicy |= B.MaskPolicy;
}

static bool isNonZeroAVL(const MachineOperand &MO) {
  if (MO.isReg())
    return RISCV::X0 == MO.getReg();
  assert(MO.isImm());
  return 0 != MO.getImm();
}

// Return true if we can mutate PrevMI to match MI without changing any the
// fields which would be observed.
static bool canMutatePriorConfig(const MachineInstr &PrevMI,
                                 const MachineInstr &MI,
                                 const DemandedFields &Used) {
  // If the VL values aren't equal, return false if either a) the former is
  // demanded, or b) we can't rewrite the former to be the later for
  // implementation reasons.
  if (!isVLPreservingConfig(MI)) {
    if (Used.VLAny)
      return false;

    // We don't bother to handle the equally zero case here as it's largely
    // uninteresting.
    if (Used.VLZeroness) {
      if (isVLPreservingConfig(PrevMI))
        return false;
      if (!isNonZeroAVL(MI.getOperand(1)) ||
          !isNonZeroAVL(PrevMI.getOperand(1)))
        return false;
    }

    // TODO: Track whether the register is defined between
    // PrevMI and MI.
    if (MI.getOperand(1).isReg() &&
        RISCV::X0 != MI.getOperand(1).getReg())
      return false;
  }

  if (!PrevMI.getOperand(2).isImm() || !MI.getOperand(2).isImm())
    return false;

  auto PriorVType = PrevMI.getOperand(2).getImm();
  auto VType = MI.getOperand(2).getImm();
  return areCompatibleVTYPEs(PriorVType, VType, Used);
}

void RISCVInsertVSETVLI::doLocalPostpass(MachineBasicBlock &MBB) {
  MachineInstr *NextMI = nullptr;
  // We can have arbitrary code in successors, so VL and VTYPE
  // must be considered demanded.
  DemandedFields Used;
  Used.demandVL();
  Used.demandVTYPE();
  SmallVector<MachineInstr*> ToDelete;
  for (MachineInstr &MI : make_range(MBB.rbegin(), MBB.rend())) {

    if (!isVectorConfigInstr(MI)) {
      doUnion(Used, getDemanded(MI, MRI));
      continue;
    }

    Register VRegDef = MI.getOperand(0).getReg();
    if (VRegDef != RISCV::X0 &&
        !(VRegDef.isVirtual() && MRI->use_nodbg_empty(VRegDef)))
      Used.demandVL();

    if (NextMI) {
      if (!Used.usedVL() && !Used.usedVTYPE()) {
        ToDelete.push_back(&MI);
        // Leave NextMI unchanged
        continue;
      } else if (canMutatePriorConfig(MI, *NextMI, Used)) {
        if (!isVLPreservingConfig(*NextMI)) {
          MI.getOperand(0).setReg(NextMI->getOperand(0).getReg());
          MI.getOperand(0).setIsDead(false);
          if (NextMI->getOperand(1).isImm())
            MI.getOperand(1).ChangeToImmediate(NextMI->getOperand(1).getImm());
          else
            MI.getOperand(1).ChangeToRegister(NextMI->getOperand(1).getReg(), false);
          MI.setDesc(NextMI->getDesc());
        }
        MI.getOperand(2).setImm(NextMI->getOperand(2).getImm());
        ToDelete.push_back(NextMI);
        // fallthrough
      }
    }
    NextMI = &MI;
    Used = getDemanded(MI, MRI);
  }

  for (auto *MI : ToDelete)
    MI->eraseFromParent();
}

void RISCVInsertVSETVLI::insertReadVL(MachineBasicBlock &MBB) {
  for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
    MachineInstr &MI = *I++;
    if (RISCV::isFaultFirstLoad(MI)) {
      Register VLOutput = MI.getOperand(1).getReg();
      if (!MRI->use_nodbg_empty(VLOutput))
        BuildMI(MBB, I, MI.getDebugLoc(), TII->get(RISCV::PseudoReadVL),
                VLOutput);
      // We don't use the vl output of the VLEFF/VLSEGFF anymore.
      MI.getOperand(1).setReg(RISCV::X0);
    }
  }
}

bool RISCVInsertVSETVLI::runOnMachineFunction(MachineFunction &MF) {
  // Skip if the vector extension is not enabled.
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasVInstructions())
    return false;

  LLVM_DEBUG(dbgs() << "Entering InsertVSETVLI for " << MF.getName() << "\n");

  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();

  assert(BlockInfo.empty() && "Expect empty block infos");
  BlockInfo.resize(MF.getNumBlockIDs());

  bool HaveVectorOp = false;

  // Phase 1 - determine how VL/VTYPE are affected by the each block.
  for (const MachineBasicBlock &MBB : MF) {
    VSETVLIInfo TmpStatus;
    HaveVectorOp |= computeVLVTYPEChanges(MBB, TmpStatus);
    // Initial exit state is whatever change we found in the block.
    BlockData &BBInfo = BlockInfo[MBB.getNumber()];
    BBInfo.Exit = TmpStatus;
    LLVM_DEBUG(dbgs() << "Initial exit state of " << printMBBReference(MBB)
                      << " is " << BBInfo.Exit << "\n");

  }

  // If we didn't find any instructions that need VSETVLI, we're done.
  if (!HaveVectorOp) {
    BlockInfo.clear();
    return false;
  }

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

  // Perform partial redundancy elimination of vsetvli transitions.
  for (MachineBasicBlock &MBB : MF)
    doPRE(MBB);

  // Phase 3 - add any vsetvli instructions needed in the block. Use the
  // Phase 2 information to avoid adding vsetvlis before the first vector
  // instruction in the block if the VL/VTYPE is satisfied by its
  // predecessors.
  for (MachineBasicBlock &MBB : MF)
    emitVSETVLIs(MBB);

  // Now that all vsetvlis are explicit, go through and do block local
  // DSE and peephole based demanded fields based transforms.  Note that
  // this *must* be done outside the main dataflow so long as we allow
  // any cross block analysis within the dataflow.  We can't have both
  // demanded fields based mutation and non-local analysis in the
  // dataflow at the same time without introducing inconsistencies.
  for (MachineBasicBlock &MBB : MF)
    doLocalPostpass(MBB);

  // Once we're fully done rewriting all the instructions, do a final pass
  // through to check for VSETVLIs which write to an unused destination.
  // For the non X0, X0 variant, we can replace the destination register
  // with X0 to reduce register pressure.  This is really a generic
  // optimization which can be applied to any dead def (TODO: generalize).
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() == RISCV::PseudoVSETVLI ||
          MI.getOpcode() == RISCV::PseudoVSETIVLI) {
        Register VRegDef = MI.getOperand(0).getReg();
        if (VRegDef != RISCV::X0 && MRI->use_nodbg_empty(VRegDef))
          MI.getOperand(0).setReg(RISCV::X0);
      }
    }
  }

  // Insert PseudoReadVL after VLEFF/VLSEGFF and replace it with the vl output
  // of VLEFF/VLSEGFF.
  for (MachineBasicBlock &MBB : MF)
    insertReadVL(MBB);

  BlockInfo.clear();
  return HaveVectorOp;
}

/// Returns an instance of the Insert VSETVLI pass.
FunctionPass *llvm::createRISCVInsertVSETVLIPass() {
  return new RISCVInsertVSETVLI();
}
