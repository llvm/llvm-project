//===- RISCVVectorConfigAnalysis ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This is the RISC-V analysis of vector unit config.
//===----------------------------------------------------------------------===//

#include "RISCVVectorConfigAnalysis.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveStacks.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-vconfig-analysis"

static bool isLMUL1OrSmaller(RISCVVType::VLMUL LMUL) {
  auto [LMul, Fractional] = RISCVVType::decodeVLMUL(LMUL);
  return Fractional || LMul == 1;
}

static unsigned getVLOpNum(const MachineInstr &MI) {
  return RISCVII::getVLOpNum(MI.getDesc());
}

static unsigned getSEWOpNum(const MachineInstr &MI) {
  return RISCVII::getSEWOpNum(MI.getDesc());
}

/// Given a virtual register \p Reg, return the corresponding VNInfo for it.
/// This will return nullptr if the virtual register is an implicit_def or
/// if LiveIntervals is not available.
static VNInfo *getVNInfoFromReg(Register Reg, const MachineInstr &MI,
                                const LiveIntervals *LIS) {
  assert(Reg.isVirtual());
  if (!LIS)
    return nullptr;
  auto &LI = LIS->getInterval(Reg);
  SlotIndex SI = LIS->getSlotIndexes()->getInstructionIndex(MI);
  return LI.getVNInfoBefore(SI);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
/// Implement operator<<.
void DemandedFields::print(raw_ostream &OS) const {
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
  case SEWGreaterThanOrEqualAndLessThan64:
    OS << "SEWGreaterThanOrEqualAndLessThan64";
    break;
  case SEWNone:
    OS << "SEWNone";
    break;
  };
  OS << ", ";
  OS << "LMUL=";
  switch (LMUL) {
  case LMULEqual:
    OS << "LMULEqual";
    break;
  case LMULLessThanOrEqualToM1:
    OS << "LMULLessThanOrEqualToM1";
    break;
  case LMULNone:
    OS << "LMULNone";
    break;
  };
  OS << ", ";
  OS << "SEWLMULRatio=" << SEWLMULRatio << ", ";
  OS << "TailPolicy=" << TailPolicy << ", ";
  OS << "MaskPolicy=" << MaskPolicy << ", ";
  OS << "VILL=" << VILL;
  OS << "}";
}

void VSETVLIInfo::print(raw_ostream &OS) const {
  OS << "{";
  if (!isValid())
    OS << "Uninitialized";
  if (isUnknown())
    OS << "unknown";
  if (hasAVLReg())
    OS << "AVLReg=" << llvm::printReg(getAVLReg());
  if (hasAVLImm())
    OS << "AVLImm=" << (unsigned)AVLImm;
  if (hasAVLVLMAX())
    OS << "AVLVLMAX";
  OS << ", ";

  unsigned LMul;
  bool Fractional;
  std::tie(LMul, Fractional) = decodeVLMUL(VLMul);

  OS << "VLMul=";
  if (Fractional)
    OS << "mf";
  else
    OS << "m";
  OS << LMul << ", "
     << "SEW=e" << (unsigned)SEW << ", "
     << "TailAgnostic=" << (bool)TailAgnostic << ", "
     << "MaskAgnostic=" << (bool)MaskAgnostic << ", "
     << "SEWLMULRatioOnly=" << (bool)SEWLMULRatioOnly << "}";
}
#endif

void DemandedFields::demandVTYPE() {
  SEW = SEWEqual;
  LMUL = LMULEqual;
  SEWLMULRatio = true;
  TailPolicy = true;
  MaskPolicy = true;
  VILL = true;
}

void DemandedFields::doUnion(const DemandedFields &B) {
  VLAny |= B.VLAny;
  VLZeroness |= B.VLZeroness;
  SEW = std::max(SEW, B.SEW);
  LMUL = std::max(LMUL, B.LMUL);
  SEWLMULRatio |= B.SEWLMULRatio;
  TailPolicy |= B.TailPolicy;
  MaskPolicy |= B.MaskPolicy;
  VILL |= B.VILL;
}

const MachineInstr *VSETVLIInfo::getAVLDefMI(const LiveIntervals *LIS) const {
  assert(hasAVLReg());
  if (!LIS || getAVLVNInfo()->isPHIDef())
    return nullptr;
  auto *MI = LIS->getInstructionFromIndex(getAVLVNInfo()->def);
  assert(MI);
  return MI;
}

void VSETVLIInfo::setAVL(const VSETVLIInfo &Info) {
  assert(Info.isValid());
  if (Info.isUnknown())
    setUnknown();
  else if (Info.hasAVLReg())
    setAVLRegDef(Info.getAVLVNInfo(), Info.getAVLReg());
  else if (Info.hasAVLVLMAX())
    setAVLVLMAX();
  else {
    assert(Info.hasAVLImm());
    setAVLImm(Info.getAVLImm());
  }
}

bool VSETVLIInfo::hasNonZeroAVL(const LiveIntervals *LIS) const {
  if (hasAVLImm())
    return getAVLImm() > 0;
  if (hasAVLReg()) {
    if (auto *DefMI = getAVLDefMI(LIS))
      return RISCVInstrInfo::isNonZeroLoadImmediate(*DefMI);
  }
  if (hasAVLVLMAX())
    return true;
  return false;
}

bool VSETVLIInfo::hasSameAVL(const VSETVLIInfo &Other) const {
  // Without LiveIntervals, we don't know which instruction defines a
  // register.  Since a register may be redefined, this means all AVLIsReg
  // states must be treated as possibly distinct.
  if (hasAVLReg() && Other.hasAVLReg()) {
    assert(!getAVLVNInfo() == !Other.getAVLVNInfo() &&
           "we either have intervals or we don't");
    if (!getAVLVNInfo())
      return false;
  }
  return hasSameAVLLatticeValue(Other);
}

bool VSETVLIInfo::hasSameAVLLatticeValue(const VSETVLIInfo &Other) const {
  if (hasAVLReg() && Other.hasAVLReg()) {
    assert(!getAVLVNInfo() == !Other.getAVLVNInfo() &&
           "we either have intervals or we don't");
    if (!getAVLVNInfo())
      return getAVLReg() == Other.getAVLReg();
    return getAVLVNInfo()->id == Other.getAVLVNInfo()->id &&
           getAVLReg() == Other.getAVLReg();
  }

  if (hasAVLImm() && Other.hasAVLImm())
    return getAVLImm() == Other.getAVLImm();

  if (hasAVLVLMAX())
    return Other.hasAVLVLMAX() && hasSameVLMAX(Other);

  return false;
}

void VSETVLIInfo::setVTYPE(RISCVVType::VLMUL L, unsigned S, bool TA, bool MA) {
  assert(isValid() && !isUnknown() &&
         "Can't set VTYPE for uninitialized or unknown");
  VLMul = L;
  SEW = S;
  TailAgnostic = TA;
  MaskAgnostic = MA;
}

void VSETVLIInfo::setVTYPE(unsigned VType) {
  assert(isValid() && !isUnknown() &&
         "Can't set VTYPE for uninitialized or unknown");
  VLMul = RISCVVType::getVLMUL(VType);
  SEW = RISCVVType::getSEW(VType);
  TailAgnostic = RISCVVType::isTailAgnostic(VType);
  MaskAgnostic = RISCVVType::isMaskAgnostic(VType);
}

bool VSETVLIInfo::hasSameVTYPE(const VSETVLIInfo &Other) const {
  assert(isValid() && Other.isValid() && "Can't compare invalid VSETVLIInfos");
  assert(!isUnknown() && !Other.isUnknown() &&
         "Can't compare VTYPE in unknown state");
  assert(!SEWLMULRatioOnly && !Other.SEWLMULRatioOnly &&
         "Can't compare when only LMUL/SEW ratio is valid.");
  return std::tie(VLMul, SEW, TailAgnostic, MaskAgnostic) ==
         std::tie(Other.VLMul, Other.SEW, Other.TailAgnostic,
                  Other.MaskAgnostic);
}

bool VSETVLIInfo::isCompatible(const DemandedFields &Used,
                               const VSETVLIInfo &Require,
                               const LiveIntervals *LIS) const {
  assert(isValid() && Require.isValid() &&
         "Can't compare invalid VSETVLIInfos");
  // Nothing is compatible with Unknown.
  if (isUnknown() || Require.isUnknown())
    return false;

  // If only our VLMAX ratio is valid, then this isn't compatible.
  if (SEWLMULRatioOnly || Require.SEWLMULRatioOnly)
    return false;

  if (Used.VLAny && !(hasSameAVL(Require) && hasSameVLMAX(Require)))
    return false;

  if (Used.VLZeroness && !hasEquallyZeroAVL(Require, LIS))
    return false;

  return hasCompatibleVTYPE(Used, Require);
}

bool VSETVLIInfo::operator==(const VSETVLIInfo &Other) const {
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

  if (!hasSameAVLLatticeValue(Other))
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

VSETVLIInfo VSETVLIInfo::intersect(const VSETVLIInfo &Other) const {
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

bool RISCVVectorConfigInfo::areCompatibleVTYPEs(uint64_t CurVType,
                                                uint64_t NewVType,
                                                const DemandedFields &Used) {
  switch (Used.SEW) {
  case DemandedFields::SEWNone:
    break;
  case DemandedFields::SEWEqual:
    if (RISCVVType::getSEW(CurVType) != RISCVVType::getSEW(NewVType))
      return false;
    break;
  case DemandedFields::SEWGreaterThanOrEqual:
    if (RISCVVType::getSEW(NewVType) < RISCVVType::getSEW(CurVType))
      return false;
    break;
  case DemandedFields::SEWGreaterThanOrEqualAndLessThan64:
    if (RISCVVType::getSEW(NewVType) < RISCVVType::getSEW(CurVType) ||
        RISCVVType::getSEW(NewVType) >= 64)
      return false;
    break;
  }

  switch (Used.LMUL) {
  case DemandedFields::LMULNone:
    break;
  case DemandedFields::LMULEqual:
    if (RISCVVType::getVLMUL(CurVType) != RISCVVType::getVLMUL(NewVType))
      return false;
    break;
  case DemandedFields::LMULLessThanOrEqualToM1:
    if (!isLMUL1OrSmaller(RISCVVType::getVLMUL(NewVType)))
      return false;
    break;
  }

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

bool VSETVLIInfo::hasCompatibleVTYPE(const DemandedFields &Used,
                                     const VSETVLIInfo &Require) const {
  return RISCVVectorConfigInfo::areCompatibleVTYPEs(Require.encodeVTYPE(),
                                                    encodeVTYPE(), Used);
}

bool RISCVVectorConfigInfo::haveVectorOp() { return HaveVectorOp; }

/// Return the fields and properties demanded by the provided instruction.
DemandedFields RISCVVectorConfigInfo::getDemanded(const MachineInstr &MI,
                                                  const RISCVSubtarget *ST) {
  // This function works in coalesceVSETVLI too. We can still use the value of a
  // SEW, VL, or Policy operand even though it might not be the exact value in
  // the VL or VTYPE, since we only care about what the instruction originally
  // demanded.

  // Most instructions don't use any of these subfeilds.
  DemandedFields Res;
  // Start conservative if registers are used
  if (MI.isCall() || MI.isInlineAsm() ||
      MI.readsRegister(RISCV::VL, /*TRI=*/nullptr))
    Res.demandVL();
  if (MI.isCall() || MI.isInlineAsm() ||
      MI.readsRegister(RISCV::VTYPE, /*TRI=*/nullptr))
    Res.demandVTYPE();
  // Start conservative on the unlowered form too
  uint64_t TSFlags = MI.getDesc().TSFlags;
  if (RISCVII::hasSEWOp(TSFlags)) {
    Res.demandVTYPE();
    if (RISCVII::hasVLOp(TSFlags))
      if (const MachineOperand &VLOp = MI.getOperand(getVLOpNum(MI));
          !VLOp.isReg() || !VLOp.isUndef())
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
  if (RISCVInstrInfo::getEEWForLoadStore(MI)) {
    Res.SEW = DemandedFields::SEWNone;
    Res.LMUL = DemandedFields::LMULNone;
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
  if (RISCVInstrInfo::isMaskRegOp(MI)) {
    Res.SEW = DemandedFields::SEWNone;
    Res.LMUL = DemandedFields::LMULNone;
  }

  // For vmv.s.x and vfmv.s.f, there are only two behaviors, VL = 0 and VL > 0.
  if (RISCVInstrInfo::isScalarInsertInstr(MI)) {
    Res.LMUL = DemandedFields::LMULNone;
    Res.SEWLMULRatio = false;
    Res.VLAny = false;
    // For vmv.s.x and vfmv.s.f, if the passthru is *undefined*, we don't
    // need to preserve any other bits and are thus compatible with any larger,
    // etype and can disregard policy bits.  Warning: It's tempting to try doing
    // this for any tail agnostic operation, but we can't as TA requires
    // tail lanes to either be the original value or -1.  We are writing
    // unknown bits to the lanes here.
    if (RISCVInstrInfo::hasUndefinedPassthru(MI)) {
      if (RISCVInstrInfo::isFloatScalarMoveOrScalarSplatInstr(MI) &&
          !ST->hasVInstructionsF64())
        Res.SEW = DemandedFields::SEWGreaterThanOrEqualAndLessThan64;
      else
        Res.SEW = DemandedFields::SEWGreaterThanOrEqual;
      Res.TailPolicy = false;
    }
  }

  // vmv.x.s, and vfmv.f.s are unconditional and ignore everything except SEW.
  if (RISCVInstrInfo::isScalarExtractInstr(MI)) {
    assert(!RISCVII::hasVLOp(TSFlags));
    Res.LMUL = DemandedFields::LMULNone;
    Res.SEWLMULRatio = false;
    Res.TailPolicy = false;
    Res.MaskPolicy = false;
  }

  if (RISCVII::hasVLOp(MI.getDesc().TSFlags)) {
    const MachineOperand &VLOp = MI.getOperand(getVLOpNum(MI));
    // A slidedown/slideup with an *undefined* passthru can freely clobber
    // elements not copied from the source vector (e.g. masked off, tail, or
    // slideup's prefix). Notes:
    // * We can't modify SEW here since the slide amount is in units of SEW.
    // * VL=1 is special only because we have existing support for zero vs
    //   non-zero VL.  We could generalize this if we had a VL > C predicate.
    // * The LMUL1 restriction is for machines whose latency may depend on LMUL.
    // * As above, this is only legal for tail "undefined" not "agnostic".
    // * We avoid increasing vl if the subtarget has +vl-dependent-latency
    if (RISCVInstrInfo::isVSlideInstr(MI) && VLOp.isImm() &&
        VLOp.getImm() == 1 && RISCVInstrInfo::hasUndefinedPassthru(MI) &&
        !ST->hasVLDependentLatency()) {
      Res.VLAny = false;
      Res.VLZeroness = true;
      Res.LMUL = DemandedFields::LMULLessThanOrEqualToM1;
      Res.TailPolicy = false;
    }

    // A tail undefined vmv.v.i/x or vfmv.v.f with VL=1 can be treated in the
    // same semantically as vmv.s.x.  This is particularly useful since we don't
    // have an immediate form of vmv.s.x, and thus frequently use vmv.v.i in
    // it's place. Since a splat is non-constant time in LMUL, we do need to be
    // careful to not increase the number of active vector registers (unlike for
    // vmv.s.x.)
    if (RISCVInstrInfo::isScalarSplatInstr(MI) && VLOp.isImm() &&
        VLOp.getImm() == 1 && RISCVInstrInfo::hasUndefinedPassthru(MI) &&
        !ST->hasVLDependentLatency()) {
      Res.LMUL = DemandedFields::LMULLessThanOrEqualToM1;
      Res.SEWLMULRatio = false;
      Res.VLAny = false;
      if (RISCVInstrInfo::isFloatScalarMoveOrScalarSplatInstr(MI) &&
          !ST->hasVInstructionsF64())
        Res.SEW = DemandedFields::SEWGreaterThanOrEqualAndLessThan64;
      else
        Res.SEW = DemandedFields::SEWGreaterThanOrEqual;
      Res.TailPolicy = false;
    }
  }

  // In §32.16.6, whole vector register moves have a dependency on SEW. At the
  // MIR level though we don't encode the element type, and it gives the same
  // result whatever the SEW may be.
  //
  // However it does need valid SEW, i.e. vill must be cleared. The entry to a
  // function, calls and inline assembly may all set it, so make sure we clear
  // it for whole register copies. Do this by leaving VILL demanded.
  if (RISCVInstrInfo::isVectorCopy(ST->getRegisterInfo(), MI)) {
    Res.LMUL = DemandedFields::LMULNone;
    Res.SEW = DemandedFields::SEWNone;
    Res.SEWLMULRatio = false;
    Res.TailPolicy = false;
    Res.MaskPolicy = false;
  }

  if (RISCVInstrInfo::isVExtractInstr(MI)) {
    assert(!RISCVII::hasVLOp(TSFlags));
    // TODO: LMUL can be any larger value (without cost)
    Res.TailPolicy = false;
  }

  return Res;
}

// Given an incoming state reaching MI, minimally modifies that state so that it
// is compatible with MI. The resulting state is guaranteed to be semantically
// legal for MI, but may not be the state requested by MI.
void RISCVVectorConfigInfo::transferBefore(VSETVLIInfo &Info,
                                           const MachineInstr &MI) const {
  if (RISCVInstrInfo::isVectorCopy(ST->getRegisterInfo(), MI) &&
      (Info.isUnknown() || !Info.isValid() || Info.hasSEWLMULRatioOnly())) {
    // Use an arbitrary but valid AVL and VTYPE so vill will be cleared. It may
    // be coalesced into another vsetvli since we won't demand any fields.
    VSETVLIInfo NewInfo; // Need a new VSETVLIInfo to clear SEWLMULRatioOnly
    NewInfo.setAVLImm(1);
    NewInfo.setVTYPE(RISCVVType::LMUL_1, /*sew*/ 8, /*ta*/ true, /*ma*/ true);
    Info = NewInfo;
    return;
  }

  if (!RISCVII::hasSEWOp(MI.getDesc().TSFlags))
    return;

  DemandedFields Demanded = getDemanded(MI, ST);

  const VSETVLIInfo NewInfo = computeInfoForInstr(MI);
  assert(NewInfo.isValid() && !NewInfo.isUnknown());
  if (Info.isValid() && !needVSETVLI(Demanded, NewInfo, Info))
    return;

  const VSETVLIInfo PrevInfo = Info;
  if (!Info.isValid() || Info.isUnknown())
    Info = NewInfo;

  const VSETVLIInfo IncomingInfo = adjustIncoming(PrevInfo, NewInfo, Demanded);

  // If MI only demands that VL has the same zeroness, we only need to set the
  // AVL if the zeroness differs.  This removes a vsetvli entirely if the types
  // match or allows use of cheaper avl preserving variant if VLMAX doesn't
  // change. If VLMAX might change, we couldn't use the 'vsetvli x0, x0, vtype"
  // variant, so we avoid the transform to prevent extending live range of an
  // avl register operand.
  // TODO: We can probably relax this for immediates.
  bool EquallyZero = IncomingInfo.hasEquallyZeroAVL(PrevInfo, LIS) &&
                     IncomingInfo.hasSameVLMAX(PrevInfo);
  if (Demanded.VLAny || (Demanded.VLZeroness && !EquallyZero))
    Info.setAVL(IncomingInfo);

  Info.setVTYPE(
      ((Demanded.LMUL || Demanded.SEWLMULRatio) ? IncomingInfo : Info)
          .getVLMUL(),
      ((Demanded.SEW || Demanded.SEWLMULRatio) ? IncomingInfo : Info).getSEW(),
      // Prefer tail/mask agnostic since it can be relaxed to undisturbed later
      // if needed.
      (Demanded.TailPolicy ? IncomingInfo : Info).getTailAgnostic() ||
          IncomingInfo.getTailAgnostic(),
      (Demanded.MaskPolicy ? IncomingInfo : Info).getMaskAgnostic() ||
          IncomingInfo.getMaskAgnostic());

  // If we only knew the sew/lmul ratio previously, replace the VTYPE but keep
  // the AVL.
  if (Info.hasSEWLMULRatioOnly()) {
    VSETVLIInfo RatiolessInfo = IncomingInfo;
    RatiolessInfo.setAVL(Info);
    Info = RatiolessInfo;
  }
}

// Given a state with which we evaluated MI (see transferBefore above for why
// this might be different that the state MI requested), modify the state to
// reflect the changes MI might make.
void RISCVVectorConfigInfo::transferAfter(VSETVLIInfo &Info,
                                          const MachineInstr &MI) const {
  if (RISCVInstrInfo::isVectorConfigInstr(MI)) {
    Info = getInfoForVSETVLI(MI);
    return;
  }

  if (RISCVInstrInfo::isFaultOnlyFirstLoad(MI)) {
    // Update AVL to vl-output of the fault first load.
    assert(MI.getOperand(1).getReg().isVirtual());
    if (LIS) {
      auto &LI = LIS->getInterval(MI.getOperand(1).getReg());
      SlotIndex SI =
          LIS->getSlotIndexes()->getInstructionIndex(MI).getRegSlot();
      VNInfo *VNI = LI.getVNInfoAt(SI);
      Info.setAVLRegDef(VNI, MI.getOperand(1).getReg());
    } else
      Info.setAVLRegDef(nullptr, MI.getOperand(1).getReg());
    return;
  }

  // If this is something that updates VL/VTYPE that we don't know about, set
  // the state to unknown.
  if (MI.isCall() || MI.isInlineAsm() ||
      MI.modifiesRegister(RISCV::VL, /*TRI=*/nullptr) ||
      MI.modifiesRegister(RISCV::VTYPE, /*TRI=*/nullptr))
    Info = VSETVLIInfo::getUnknown();
}

unsigned RISCVVectorConfigInfo::computeVLMAX(unsigned VLEN, unsigned SEW,
                                             RISCVVType::VLMUL VLMul) {
  auto [LMul, Fractional] = RISCVVType::decodeVLMUL(VLMul);
  if (Fractional)
    VLEN = VLEN / LMul;
  else
    VLEN = VLEN * LMul;
  return VLEN / SEW;
}

// If we don't use LMUL or the SEW/LMUL ratio, then adjust LMUL so that we
// maintain the SEW/LMUL ratio. This allows us to eliminate VL toggles in more
// places.
VSETVLIInfo RISCVVectorConfigInfo::adjustIncoming(const VSETVLIInfo &PrevInfo,
                                                  const VSETVLIInfo &NewInfo,
                                                  DemandedFields &Demanded) {
  VSETVLIInfo Info = NewInfo;

  if (!Demanded.LMUL && !Demanded.SEWLMULRatio && PrevInfo.isValid() &&
      !PrevInfo.isUnknown()) {
    if (auto NewVLMul = RISCVVType::getSameRatioLMUL(
            PrevInfo.getSEW(), PrevInfo.getVLMUL(), Info.getSEW()))
      Info.setVLMul(*NewVLMul);
    Demanded.LMUL = DemandedFields::LMULEqual;
  }

  return Info;
}

bool RISCVVectorConfigInfo::needVSETVLI(const DemandedFields &Used,
                                        const VSETVLIInfo &Require,
                                        const VSETVLIInfo &CurInfo) const {
  if (!CurInfo.isValid() || CurInfo.isUnknown() ||
      CurInfo.hasSEWLMULRatioOnly())
    return true;

  if (CurInfo.isCompatible(Used, Require, LIS))
    return false;

  return true;
}

VSETVLIInfo
RISCVVectorConfigInfo::getInfoForVSETVLI(const MachineInstr &MI) const {
  VSETVLIInfo NewInfo;
  if (MI.getOpcode() == RISCV::PseudoVSETIVLI) {
    NewInfo.setAVLImm(MI.getOperand(1).getImm());
  } else {
    assert(MI.getOpcode() == RISCV::PseudoVSETVLI ||
           MI.getOpcode() == RISCV::PseudoVSETVLIX0);
    if (MI.getOpcode() == RISCV::PseudoVSETVLIX0)
      NewInfo.setAVLVLMAX();
    else if (MI.getOperand(1).isUndef())
      // Otherwise use an AVL of 1 to avoid depending on previous vl.
      NewInfo.setAVLImm(1);
    else {
      Register AVLReg = MI.getOperand(1).getReg();
      VNInfo *VNI = getVNInfoFromReg(AVLReg, MI, LIS);
      NewInfo.setAVLRegDef(VNI, AVLReg);
    }
  }
  NewInfo.setVTYPE(MI.getOperand(2).getImm());

  forwardVSETVLIAVL(NewInfo);

  return NewInfo;
}

bool RISCVVectorConfigInfo::canMutatePriorConfig(
    const MachineInstr &PrevMI, const MachineInstr &MI,
    const DemandedFields &Used) const {
  // If the VL values aren't equal, return false if either a) the former is
  // demanded, or b) we can't rewrite the former to be the later for
  // implementation reasons.
  if (!RISCVInstrInfo::isVLPreservingConfig(MI)) {
    if (Used.VLAny)
      return false;

    if (Used.VLZeroness) {
      if (RISCVInstrInfo::isVLPreservingConfig(PrevMI))
        return false;
      if (!getInfoForVSETVLI(PrevMI).hasEquallyZeroAVL(getInfoForVSETVLI(MI),
                                                       LIS))
        return false;
    }

    auto &AVL = MI.getOperand(1);

    // If the AVL is a register, we need to make sure its definition is the same
    // at PrevMI as it was at MI.
    if (AVL.isReg() && AVL.getReg() != RISCV::X0) {
      VNInfo *VNI = getVNInfoFromReg(AVL.getReg(), MI, LIS);
      VNInfo *PrevVNI = getVNInfoFromReg(AVL.getReg(), PrevMI, LIS);
      if (!VNI || !PrevVNI || VNI != PrevVNI)
        return false;
    }
  }

  assert(PrevMI.getOperand(2).isImm() && MI.getOperand(2).isImm());
  auto PriorVType = PrevMI.getOperand(2).getImm();
  auto VType = MI.getOperand(2).getImm();
  return areCompatibleVTYPEs(PriorVType, VType, Used);
}

VSETVLIInfo
RISCVVectorConfigInfo::computeInfoForInstr(const MachineInstr &MI) const {
  VSETVLIInfo InstrInfo;
  const uint64_t TSFlags = MI.getDesc().TSFlags;

  bool TailAgnostic = true;
  bool MaskAgnostic = true;
  if (!RISCVInstrInfo::hasUndefinedPassthru(MI)) {
    // Start with undisturbed.
    TailAgnostic = false;
    MaskAgnostic = false;

    // If there is a policy operand, use it.
    if (RISCVII::hasVecPolicyOp(TSFlags)) {
      const MachineOperand &Op = MI.getOperand(MI.getNumExplicitOperands() - 1);
      uint64_t Policy = Op.getImm();
      assert(Policy <=
                 (RISCVVType::TAIL_AGNOSTIC | RISCVVType::MASK_AGNOSTIC) &&
             "Invalid Policy Value");
      TailAgnostic = Policy & RISCVVType::TAIL_AGNOSTIC;
      MaskAgnostic = Policy & RISCVVType::MASK_AGNOSTIC;
    }

    if (!RISCVII::usesMaskPolicy(TSFlags))
      MaskAgnostic = true;
  }

  RISCVVType::VLMUL VLMul = RISCVII::getLMul(TSFlags);

  unsigned Log2SEW = MI.getOperand(getSEWOpNum(MI)).getImm();
  // A Log2SEW of 0 is an operation on mask registers only.
  unsigned SEW = Log2SEW ? 1 << Log2SEW : 8;
  assert(RISCVVType::isValidSEW(SEW) && "Unexpected SEW");

  if (RISCVII::hasVLOp(TSFlags)) {
    const MachineOperand &VLOp = MI.getOperand(getVLOpNum(MI));
    if (VLOp.isImm()) {
      int64_t Imm = VLOp.getImm();
      // Convert the VLMax sentintel to X0 register.
      if (Imm == RISCV::VLMaxSentinel) {
        // If we know the exact VLEN, see if we can use the constant encoding
        // for the VLMAX instead.  This reduces register pressure slightly.
        const unsigned VLMAX = computeVLMAX(ST->getRealMaxVLen(), SEW, VLMul);
        if (ST->getRealMinVLen() == ST->getRealMaxVLen() && VLMAX <= 31)
          InstrInfo.setAVLImm(VLMAX);
        else
          InstrInfo.setAVLVLMAX();
      } else
        InstrInfo.setAVLImm(Imm);
    } else if (VLOp.isUndef()) {
      // Otherwise use an AVL of 1 to avoid depending on previous vl.
      InstrInfo.setAVLImm(1);
    } else {
      VNInfo *VNI = getVNInfoFromReg(VLOp.getReg(), MI, LIS);
      InstrInfo.setAVLRegDef(VNI, VLOp.getReg());
    }
  } else {
    assert(RISCVInstrInfo::isScalarExtractInstr(MI) ||
           RISCVInstrInfo::isVExtractInstr(MI));
    // Pick a random value for state tracking purposes, will be ignored via
    // the demanded fields mechanism
    InstrInfo.setAVLImm(1);
  }
#ifndef NDEBUG
  if (std::optional<unsigned> EEW = RISCVInstrInfo::getEEWForLoadStore(MI)) {
    assert(SEW == EEW && "Initial SEW doesn't match expected EEW");
  }
#endif
  InstrInfo.setVTYPE(VLMul, SEW, TailAgnostic, MaskAgnostic);

  forwardVSETVLIAVL(InstrInfo);

  return InstrInfo;
}

bool RISCVVectorConfigInfo::computeVLVTYPEChanges(const MachineBasicBlock &MBB,
                                                  VSETVLIInfo &Info) const {
  bool HadVectorOp = false;

  Info = BlockInfo[MBB.getNumber()].Pred;
  for (const MachineInstr &MI : MBB) {
    transferBefore(Info, MI);

    if (RISCVInstrInfo::isVectorConfigInstr(MI) ||
        RISCVII::hasSEWOp(MI.getDesc().TSFlags) ||
        RISCVInstrInfo::isVectorCopy(ST->getRegisterInfo(), MI))
      HadVectorOp = true;

    transferAfter(Info, MI);
  }

  return HadVectorOp;
}

void RISCVVectorConfigInfo::forwardVSETVLIAVL(VSETVLIInfo &Info) const {
  if (!Info.hasAVLReg())
    return;
  const MachineInstr *DefMI = Info.getAVLDefMI(LIS);
  if (!DefMI || !RISCVInstrInfo::isVectorConfigInstr(*DefMI))
    return;
  VSETVLIInfo DefInstrInfo = getInfoForVSETVLI(*DefMI);
  if (!DefInstrInfo.hasSameVLMAX(Info))
    return;
  Info.setAVL(DefInstrInfo);
}

void RISCVVectorConfigInfo::computeIncomingVLVTYPE(
    const MachineBasicBlock &MBB) {

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
void RISCVVectorConfigInfo::compute(const MachineFunction &MF) {
  assert(BlockInfo.empty() && "Expect empty block infos");
  BlockInfo.resize(MF.getNumBlockIDs());

  HaveVectorOp = false;

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
    return;
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
}

void RISCVVectorConfigInfo::clear() { BlockInfo.clear(); }

char RISCVVectorConfigWrapperPass::ID = 0;

INITIALIZE_PASS_BEGIN(RISCVVectorConfigWrapperPass, DEBUG_TYPE,
                      "RISC-V Vector Config Analysis", false, true)
INITIALIZE_PASS_END(RISCVVectorConfigWrapperPass, DEBUG_TYPE,
                    "RISC-V Vector Config Analysis", false, true)

RISCVVectorConfigWrapperPass::RISCVVectorConfigWrapperPass()
    : MachineFunctionPass(ID) {}

void RISCVVectorConfigWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool RISCVVectorConfigWrapperPass::runOnMachineFunction(MachineFunction &MF) {
  auto *LISWrapper = getAnalysisIfAvailable<LiveIntervalsWrapperPass>();
  LiveIntervals *LIS = LISWrapper ? &LISWrapper->getLIS() : nullptr;
  Result = RISCVVectorConfigInfo(&MF.getSubtarget<RISCVSubtarget>(), LIS);
  Result.compute(MF);
  return false;
}
