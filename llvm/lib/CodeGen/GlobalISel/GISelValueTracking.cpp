//===- lib/CodeGen/GlobalISel/GISelValueTracking.cpp --------------*- C++
//*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Provides analysis for querying information about KnownBits during GISel
/// passes.
//
//===----------------------------------------------------------------------===//
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineFloatingPointPredicateUtils.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/LowLevelTypeUtils.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/FMF.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/KnownFPClass.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "gisel-known-bits"

using namespace llvm;
using namespace MIPatternMatch;

char llvm::GISelValueTrackingAnalysisLegacy::ID = 0;

INITIALIZE_PASS(GISelValueTrackingAnalysisLegacy, DEBUG_TYPE,
                "Analysis for ComputingKnownBits", false, true)

GISelValueTracking::GISelValueTracking(MachineFunction &MF, unsigned MaxDepth)
    : MF(MF), MRI(MF.getRegInfo()), TL(*MF.getSubtarget().getTargetLowering()),
      DL(MF.getFunction().getDataLayout()), MaxDepth(MaxDepth) {}

Align GISelValueTracking::computeKnownAlignment(Register R, unsigned Depth) {
  const MachineInstr *MI = MRI.getVRegDef(R);
  switch (MI->getOpcode()) {
  case TargetOpcode::COPY:
    return computeKnownAlignment(MI->getOperand(1).getReg(), Depth);
  case TargetOpcode::G_ASSERT_ALIGN: {
    // TODO: Min with source
    return Align(MI->getOperand(2).getImm());
  }
  case TargetOpcode::G_FRAME_INDEX: {
    int FrameIdx = MI->getOperand(1).getIndex();
    return MF.getFrameInfo().getObjectAlign(FrameIdx);
  }
  case TargetOpcode::G_INTRINSIC:
  case TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS:
  case TargetOpcode::G_INTRINSIC_CONVERGENT:
  case TargetOpcode::G_INTRINSIC_CONVERGENT_W_SIDE_EFFECTS:
  default:
    return TL.computeKnownAlignForTargetInstr(*this, R, MRI, Depth + 1);
  }
}

KnownBits GISelValueTracking::getKnownBits(MachineInstr &MI) {
  assert(MI.getNumExplicitDefs() == 1 &&
         "expected single return generic instruction");
  return getKnownBits(MI.getOperand(0).getReg());
}

KnownBits GISelValueTracking::getKnownBits(Register R) {
  const LLT Ty = MRI.getType(R);
  // Since the number of lanes in a scalable vector is unknown at compile time,
  // we track one bit which is implicitly broadcast to all lanes.  This means
  // that all lanes in a scalable vector are considered demanded.
  APInt DemandedElts =
      Ty.isFixedVector() ? APInt::getAllOnes(Ty.getNumElements()) : APInt(1, 1);
  return getKnownBits(R, DemandedElts);
}

KnownBits GISelValueTracking::getKnownBits(Register R,
                                           const APInt &DemandedElts,
                                           unsigned Depth) {
  KnownBits Known;
  computeKnownBitsImpl(R, Known, DemandedElts, Depth);
  return Known;
}

bool GISelValueTracking::signBitIsZero(Register R) {
  LLT Ty = MRI.getType(R);
  unsigned BitWidth = Ty.getScalarSizeInBits();
  return maskedValueIsZero(R, APInt::getSignMask(BitWidth));
}

APInt GISelValueTracking::getKnownZeroes(Register R) {
  return getKnownBits(R).Zero;
}

APInt GISelValueTracking::getKnownOnes(Register R) {
  return getKnownBits(R).One;
}

[[maybe_unused]] static void
dumpResult(const MachineInstr &MI, const KnownBits &Known, unsigned Depth) {
  dbgs() << "[" << Depth << "] Compute known bits: " << MI << "[" << Depth
         << "] Computed for: " << MI << "[" << Depth << "] Known: 0x"
         << toString(Known.Zero | Known.One, 16, false) << "\n"
         << "[" << Depth << "] Zero: 0x" << toString(Known.Zero, 16, false)
         << "\n"
         << "[" << Depth << "] One:  0x" << toString(Known.One, 16, false)
         << "\n";
}

/// Compute known bits for the intersection of \p Src0 and \p Src1
void GISelValueTracking::computeKnownBitsMin(Register Src0, Register Src1,
                                             KnownBits &Known,
                                             const APInt &DemandedElts,
                                             unsigned Depth) {
  // Test src1 first, since we canonicalize simpler expressions to the RHS.
  computeKnownBitsImpl(Src1, Known, DemandedElts, Depth);

  // If we don't know any bits, early out.
  if (Known.isUnknown())
    return;

  KnownBits Known2;
  computeKnownBitsImpl(Src0, Known2, DemandedElts, Depth);

  // Only known if known in both the LHS and RHS.
  Known = Known.intersectWith(Known2);
}

// Bitfield extract is computed as (Src >> Offset) & Mask, where Mask is
// created using Width. Use this function when the inputs are KnownBits
// objects. TODO: Move this KnownBits.h if this is usable in more cases.
static KnownBits extractBits(unsigned BitWidth, const KnownBits &SrcOpKnown,
                             const KnownBits &OffsetKnown,
                             const KnownBits &WidthKnown) {
  KnownBits Mask(BitWidth);
  Mask.Zero = APInt::getBitsSetFrom(
      BitWidth, WidthKnown.getMaxValue().getLimitedValue(BitWidth));
  Mask.One = APInt::getLowBitsSet(
      BitWidth, WidthKnown.getMinValue().getLimitedValue(BitWidth));
  return KnownBits::lshr(SrcOpKnown, OffsetKnown) & Mask;
}

void GISelValueTracking::computeKnownBitsImpl(Register R, KnownBits &Known,
                                              const APInt &DemandedElts,
                                              unsigned Depth) {
  MachineInstr &MI = *MRI.getVRegDef(R);
  unsigned Opcode = MI.getOpcode();
  LLT DstTy = MRI.getType(R);

  // Handle the case where this is called on a register that does not have a
  // type constraint. For example, it may be post-ISel or this target might not
  // preserve the type when early-selecting instructions.
  if (!DstTy.isValid()) {
    Known = KnownBits();
    return;
  }

#ifndef NDEBUG
  if (DstTy.isFixedVector()) {
    assert(
        DstTy.getNumElements() == DemandedElts.getBitWidth() &&
        "DemandedElt width should equal the fixed vector number of elements");
  } else {
    assert(DemandedElts.getBitWidth() == 1 && DemandedElts == APInt(1, 1) &&
           "DemandedElt width should be 1 for scalars or scalable vectors");
  }
#endif

  unsigned BitWidth = DstTy.getScalarSizeInBits();
  Known = KnownBits(BitWidth); // Don't know anything

  // Depth may get bigger than max depth if it gets passed to a different
  // GISelValueTracking object.
  // This may happen when say a generic part uses a GISelValueTracking object
  // with some max depth, but then we hit TL.computeKnownBitsForTargetInstr
  // which creates a new GISelValueTracking object with a different and smaller
  // depth. If we just check for equality, we would never exit if the depth
  // that is passed down to the target specific GISelValueTracking object is
  // already bigger than its max depth.
  if (Depth >= getMaxDepth())
    return;

  if (!DemandedElts)
    return; // No demanded elts, better to assume we don't know anything.

  KnownBits Known2;

  switch (Opcode) {
  default:
    TL.computeKnownBitsForTargetInstr(*this, R, Known, DemandedElts, MRI,
                                      Depth);
    break;
  case TargetOpcode::G_BUILD_VECTOR: {
    // Collect the known bits that are shared by every demanded vector element.
    Known.Zero.setAllBits();
    Known.One.setAllBits();
    for (const auto &[I, MO] : enumerate(drop_begin(MI.operands()))) {
      if (!DemandedElts[I])
        continue;

      computeKnownBitsImpl(MO.getReg(), Known2, APInt(1, 1), Depth + 1);

      // Known bits are the values that are shared by every demanded element.
      Known = Known.intersectWith(Known2);

      // If we don't know any bits, early out.
      if (Known.isUnknown())
        break;
    }
    break;
  }
  case TargetOpcode::G_SPLAT_VECTOR: {
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known, APInt(1, 1),
                         Depth + 1);
    // Implicitly truncate the bits to match the official semantics of
    // G_SPLAT_VECTOR.
    Known = Known.trunc(BitWidth);
    break;
  }
  case TargetOpcode::COPY:
  case TargetOpcode::G_PHI:
  case TargetOpcode::PHI: {
    Known.One = APInt::getAllOnes(BitWidth);
    Known.Zero = APInt::getAllOnes(BitWidth);
    // Destination registers should not have subregisters at this
    // point of the pipeline, otherwise the main live-range will be
    // defined more than once, which is against SSA.
    assert(MI.getOperand(0).getSubReg() == 0 && "Is this code in SSA?");
    // PHI's operand are a mix of registers and basic blocks interleaved.
    // We only care about the register ones.
    for (unsigned Idx = 1; Idx < MI.getNumOperands(); Idx += 2) {
      const MachineOperand &Src = MI.getOperand(Idx);
      Register SrcReg = Src.getReg();
      LLT SrcTy = MRI.getType(SrcReg);
      // Look through trivial copies and phis but don't look through trivial
      // copies or phis of the form `%1:(s32) = OP %0:gpr32`, known-bits
      // analysis is currently unable to determine the bit width of a
      // register class.
      //
      // We can't use NoSubRegister by name as it's defined by each target but
      // it's always defined to be 0 by tablegen.
      if (SrcReg.isVirtual() && Src.getSubReg() == 0 /*NoSubRegister*/ &&
          SrcTy.isValid()) {
        APInt NowDemandedElts;
        if (!SrcTy.isFixedVector()) {
          NowDemandedElts = APInt(1, 1);
        } else if (DstTy.isFixedVector() &&
                   SrcTy.getNumElements() == DstTy.getNumElements()) {
          NowDemandedElts = DemandedElts;
        } else {
          NowDemandedElts = APInt::getAllOnes(SrcTy.getNumElements());
        }

        // For COPYs we don't do anything, don't increase the depth.
        computeKnownBitsImpl(SrcReg, Known2, NowDemandedElts,
                             Depth + (Opcode != TargetOpcode::COPY));
        Known2 = Known2.anyextOrTrunc(BitWidth);
        Known = Known.intersectWith(Known2);
        // If we reach a point where we don't know anything
        // just stop looking through the operands.
        if (Known.isUnknown())
          break;
      } else {
        // We know nothing.
        Known = KnownBits(BitWidth);
        break;
      }
    }
    break;
  }
  case TargetOpcode::G_STEP_VECTOR: {
    APInt Step = MI.getOperand(1).getCImm()->getValue();

    if (Step.isPowerOf2())
      Known.Zero.setLowBits(Step.logBase2());

    if (!isUIntN(BitWidth, DstTy.getElementCount().getKnownMinValue()))
      break;

    const APInt MinNumElts =
        APInt(BitWidth, DstTy.getElementCount().getKnownMinValue());
    const Function &F = getMachineFunction().getFunction();
    bool Overflow;
    const APInt MaxNumElts = getVScaleRange(&F, BitWidth)
                                 .getUnsignedMax()
                                 .umul_ov(MinNumElts, Overflow);
    if (Overflow)
      break;
    const APInt MaxValue = (MaxNumElts - 1).umul_ov(Step, Overflow);
    if (Overflow)
      break;
    Known.Zero.setHighBits(MaxValue.countl_zero());
    break;
  }
  case TargetOpcode::G_CONSTANT: {
    Known = KnownBits::makeConstant(MI.getOperand(1).getCImm()->getValue());
    break;
  }
  case TargetOpcode::G_FRAME_INDEX: {
    int FrameIdx = MI.getOperand(1).getIndex();
    TL.computeKnownBitsForFrameIndex(FrameIdx, Known, MF);
    break;
  }
  case TargetOpcode::G_SUB: {
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known2, DemandedElts,
                         Depth + 1);
    Known = KnownBits::sub(Known, Known2, MI.getFlag(MachineInstr::NoSWrap),
                           MI.getFlag(MachineInstr::NoUWrap));
    break;
  }
  case TargetOpcode::G_XOR: {
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known2, DemandedElts,
                         Depth + 1);

    Known ^= Known2;
    break;
  }
  case TargetOpcode::G_PTR_ADD: {
    if (DstTy.isVector())
      break;
    // G_PTR_ADD is like G_ADD. FIXME: Is this true for all targets?
    LLT Ty = MRI.getType(MI.getOperand(1).getReg());
    if (DL.isNonIntegralAddressSpace(Ty.getAddressSpace()))
      break;
    [[fallthrough]];
  }
  case TargetOpcode::G_ADD: {
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known2, DemandedElts,
                         Depth + 1);
    Known = KnownBits::add(Known, Known2);
    break;
  }
  case TargetOpcode::G_AND: {
    // If either the LHS or the RHS are Zero, the result is zero.
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known2, DemandedElts,
                         Depth + 1);

    Known &= Known2;
    break;
  }
  case TargetOpcode::G_OR: {
    // If either the LHS or the RHS are Zero, the result is zero.
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known2, DemandedElts,
                         Depth + 1);

    Known |= Known2;
    break;
  }
  case TargetOpcode::G_MUL: {
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known2, DemandedElts,
                         Depth + 1);
    Known = KnownBits::mul(Known, Known2);
    break;
  }
  case TargetOpcode::G_UMULH: {
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known2, DemandedElts,
                         Depth + 1);
    Known = KnownBits::mulhu(Known, Known2);
    break;
  }
  case TargetOpcode::G_SMULH: {
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known2, DemandedElts,
                         Depth + 1);
    Known = KnownBits::mulhs(Known, Known2);
    break;
  }
  case TargetOpcode::G_ABDU: {
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known2, DemandedElts,
                         Depth + 1);
    Known = KnownBits::abdu(Known, Known2);
    break;
  }
  case TargetOpcode::G_ABDS: {
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known2, DemandedElts,
                         Depth + 1);
    Known = KnownBits::abds(Known, Known2);

    unsigned SignBits1 =
        computeNumSignBits(MI.getOperand(2).getReg(), DemandedElts, Depth + 1);
    if (SignBits1 == 1) {
      break;
    }
    unsigned SignBits0 =
        computeNumSignBits(MI.getOperand(1).getReg(), DemandedElts, Depth + 1);

    Known.Zero.setHighBits(std::min(SignBits0, SignBits1) - 1);
    break;
  }
  case TargetOpcode::G_UDIV: {
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known2, DemandedElts,
                         Depth + 1);
    Known = KnownBits::udiv(Known, Known2,
                            MI.getFlag(MachineInstr::MIFlag::IsExact));
    break;
  }
  case TargetOpcode::G_SDIV: {
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known2, DemandedElts,
                         Depth + 1);
    Known = KnownBits::sdiv(Known, Known2,
                            MI.getFlag(MachineInstr::MIFlag::IsExact));
    break;
  }
  case TargetOpcode::G_SELECT: {
    computeKnownBitsMin(MI.getOperand(2).getReg(), MI.getOperand(3).getReg(),
                        Known, DemandedElts, Depth + 1);
    break;
  }
  case TargetOpcode::G_SMIN: {
    // TODO: Handle clamp pattern with number of sign bits
    KnownBits KnownRHS;
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(2).getReg(), KnownRHS, DemandedElts,
                         Depth + 1);
    Known = KnownBits::smin(Known, KnownRHS);
    break;
  }
  case TargetOpcode::G_SMAX: {
    // TODO: Handle clamp pattern with number of sign bits
    KnownBits KnownRHS;
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(2).getReg(), KnownRHS, DemandedElts,
                         Depth + 1);
    Known = KnownBits::smax(Known, KnownRHS);
    break;
  }
  case TargetOpcode::G_UMIN: {
    KnownBits KnownRHS;
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(2).getReg(), KnownRHS, DemandedElts,
                         Depth + 1);
    Known = KnownBits::umin(Known, KnownRHS);
    break;
  }
  case TargetOpcode::G_UMAX: {
    KnownBits KnownRHS;
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(2).getReg(), KnownRHS, DemandedElts,
                         Depth + 1);
    Known = KnownBits::umax(Known, KnownRHS);
    break;
  }
  case TargetOpcode::G_FCMP:
  case TargetOpcode::G_ICMP: {
    if (DstTy.isVector())
      break;
    if (TL.getBooleanContents(DstTy.isVector(),
                              Opcode == TargetOpcode::G_FCMP) ==
            TargetLowering::ZeroOrOneBooleanContent &&
        BitWidth > 1)
      Known.Zero.setBitsFrom(1);
    break;
  }
  case TargetOpcode::G_SEXT: {
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known, DemandedElts,
                         Depth + 1);
    // If the sign bit is known to be zero or one, then sext will extend
    // it to the top bits, else it will just zext.
    Known = Known.sext(BitWidth);
    break;
  }
  case TargetOpcode::G_ASSERT_SEXT:
  case TargetOpcode::G_SEXT_INREG: {
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known, DemandedElts,
                         Depth + 1);
    Known = Known.sextInReg(MI.getOperand(2).getImm());
    break;
  }
  case TargetOpcode::G_ANYEXT: {
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known, DemandedElts,
                         Depth + 1);
    Known = Known.anyext(BitWidth);
    break;
  }
  case TargetOpcode::G_LOAD: {
    const MachineMemOperand *MMO = *MI.memoperands_begin();
    KnownBits KnownRange(MMO->getMemoryType().getScalarSizeInBits());
    if (const MDNode *Ranges = MMO->getRanges())
      computeKnownBitsFromRangeMetadata(*Ranges, KnownRange);
    Known = KnownRange.anyext(Known.getBitWidth());
    break;
  }
  case TargetOpcode::G_SEXTLOAD:
  case TargetOpcode::G_ZEXTLOAD: {
    if (DstTy.isVector())
      break;
    const MachineMemOperand *MMO = *MI.memoperands_begin();
    KnownBits KnownRange(MMO->getMemoryType().getScalarSizeInBits());
    if (const MDNode *Ranges = MMO->getRanges())
      computeKnownBitsFromRangeMetadata(*Ranges, KnownRange);
    Known = Opcode == TargetOpcode::G_SEXTLOAD
                ? KnownRange.sext(Known.getBitWidth())
                : KnownRange.zext(Known.getBitWidth());
    break;
  }
  case TargetOpcode::G_ASHR: {
    KnownBits LHSKnown, RHSKnown;
    computeKnownBitsImpl(MI.getOperand(1).getReg(), LHSKnown, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(2).getReg(), RHSKnown, DemandedElts,
                         Depth + 1);
    Known = KnownBits::ashr(LHSKnown, RHSKnown);
    break;
  }
  case TargetOpcode::G_LSHR: {
    KnownBits LHSKnown, RHSKnown;
    computeKnownBitsImpl(MI.getOperand(1).getReg(), LHSKnown, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(2).getReg(), RHSKnown, DemandedElts,
                         Depth + 1);
    Known = KnownBits::lshr(LHSKnown, RHSKnown);
    break;
  }
  case TargetOpcode::G_SHL: {
    KnownBits LHSKnown, RHSKnown;
    computeKnownBitsImpl(MI.getOperand(1).getReg(), LHSKnown, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(2).getReg(), RHSKnown, DemandedElts,
                         Depth + 1);
    Known = KnownBits::shl(LHSKnown, RHSKnown);
    break;
  }
  case TargetOpcode::G_ROTL:
  case TargetOpcode::G_ROTR: {
    MachineInstr *AmtOpMI = MRI.getVRegDef(MI.getOperand(2).getReg());
    auto MaybeAmtOp = isConstantOrConstantSplatVector(*AmtOpMI, MRI);
    if (!MaybeAmtOp)
      break;

    Register SrcReg = MI.getOperand(1).getReg();
    computeKnownBitsImpl(SrcReg, Known, DemandedElts, Depth + 1);

    unsigned Amt = MaybeAmtOp->urem(BitWidth);

    // Canonicalize to ROTR.
    if (Opcode == TargetOpcode::G_ROTL)
      Amt = BitWidth - Amt;

    Known.Zero = Known.Zero.rotr(Amt);
    Known.One = Known.One.rotr(Amt);
    break;
  }
  case TargetOpcode::G_INTTOPTR:
  case TargetOpcode::G_PTRTOINT:
    if (DstTy.isVector())
      break;
    // Fall through and handle them the same as zext/trunc.
    [[fallthrough]];
  case TargetOpcode::G_ZEXT:
  case TargetOpcode::G_TRUNC: {
    Register SrcReg = MI.getOperand(1).getReg();
    computeKnownBitsImpl(SrcReg, Known, DemandedElts, Depth + 1);
    Known = Known.zextOrTrunc(BitWidth);
    break;
  }
  case TargetOpcode::G_ASSERT_ZEXT: {
    Register SrcReg = MI.getOperand(1).getReg();
    computeKnownBitsImpl(SrcReg, Known, DemandedElts, Depth + 1);

    unsigned SrcBitWidth = MI.getOperand(2).getImm();
    assert(SrcBitWidth && "SrcBitWidth can't be zero");
    APInt InMask = APInt::getLowBitsSet(BitWidth, SrcBitWidth);
    Known.Zero |= (~InMask);
    Known.One &= (~Known.Zero);
    break;
  }
  case TargetOpcode::G_ASSERT_ALIGN: {
    int64_t LogOfAlign = Log2_64(MI.getOperand(2).getImm());

    // TODO: Should use maximum with source
    // If a node is guaranteed to be aligned, set low zero bits accordingly as
    // well as clearing one bits.
    Known.Zero.setLowBits(LogOfAlign);
    Known.One.clearLowBits(LogOfAlign);
    break;
  }
  case TargetOpcode::G_MERGE_VALUES: {
    unsigned NumOps = MI.getNumOperands();
    unsigned OpSize = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();

    for (unsigned I = 0; I != NumOps - 1; ++I) {
      KnownBits SrcOpKnown;
      computeKnownBitsImpl(MI.getOperand(I + 1).getReg(), SrcOpKnown,
                           DemandedElts, Depth + 1);
      Known.insertBits(SrcOpKnown, I * OpSize);
    }
    break;
  }
  case TargetOpcode::G_UNMERGE_VALUES: {
    unsigned NumOps = MI.getNumOperands();
    Register SrcReg = MI.getOperand(NumOps - 1).getReg();
    LLT SrcTy = MRI.getType(SrcReg);

    if (SrcTy.isVector() && SrcTy.getScalarType() != DstTy.getScalarType())
      return; // TODO: Handle vector->subelement unmerges

    // Figure out the result operand index
    unsigned DstIdx = 0;
    for (; DstIdx != NumOps - 1 && MI.getOperand(DstIdx).getReg() != R;
         ++DstIdx)
      ;

    APInt SubDemandedElts = DemandedElts;
    if (SrcTy.isVector()) {
      unsigned DstLanes = DstTy.isVector() ? DstTy.getNumElements() : 1;
      SubDemandedElts =
          DemandedElts.zext(SrcTy.getNumElements()).shl(DstIdx * DstLanes);
    }

    KnownBits SrcOpKnown;
    computeKnownBitsImpl(SrcReg, SrcOpKnown, SubDemandedElts, Depth + 1);

    if (SrcTy.isVector())
      Known = std::move(SrcOpKnown);
    else
      Known = SrcOpKnown.extractBits(BitWidth, BitWidth * DstIdx);
    break;
  }
  case TargetOpcode::G_BSWAP: {
    Register SrcReg = MI.getOperand(1).getReg();
    computeKnownBitsImpl(SrcReg, Known, DemandedElts, Depth + 1);
    Known = Known.byteSwap();
    break;
  }
  case TargetOpcode::G_BITREVERSE: {
    Register SrcReg = MI.getOperand(1).getReg();
    computeKnownBitsImpl(SrcReg, Known, DemandedElts, Depth + 1);
    Known = Known.reverseBits();
    break;
  }
  case TargetOpcode::G_CTPOP: {
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known2, DemandedElts,
                         Depth + 1);
    // We can bound the space the count needs.  Also, bits known to be zero
    // can't contribute to the population.
    unsigned BitsPossiblySet = Known2.countMaxPopulation();
    unsigned LowBits = llvm::bit_width(BitsPossiblySet);
    Known.Zero.setBitsFrom(LowBits);
    // TODO: we could bound Known.One using the lower bound on the number of
    // bits which might be set provided by popcnt KnownOne2.
    break;
  }
  case TargetOpcode::G_UBFX: {
    KnownBits SrcOpKnown, OffsetKnown, WidthKnown;
    computeKnownBitsImpl(MI.getOperand(1).getReg(), SrcOpKnown, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(2).getReg(), OffsetKnown, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(3).getReg(), WidthKnown, DemandedElts,
                         Depth + 1);
    Known = extractBits(BitWidth, SrcOpKnown, OffsetKnown, WidthKnown);
    break;
  }
  case TargetOpcode::G_SBFX: {
    KnownBits SrcOpKnown, OffsetKnown, WidthKnown;
    computeKnownBitsImpl(MI.getOperand(1).getReg(), SrcOpKnown, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(2).getReg(), OffsetKnown, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(3).getReg(), WidthKnown, DemandedElts,
                         Depth + 1);
    OffsetKnown = OffsetKnown.sext(BitWidth);
    WidthKnown = WidthKnown.sext(BitWidth);
    Known = extractBits(BitWidth, SrcOpKnown, OffsetKnown, WidthKnown);
    // Sign extend the extracted value using shift left and arithmetic shift
    // right.
    KnownBits ExtKnown = KnownBits::makeConstant(APInt(BitWidth, BitWidth));
    KnownBits ShiftKnown = KnownBits::sub(ExtKnown, WidthKnown);
    Known = KnownBits::ashr(KnownBits::shl(Known, ShiftKnown), ShiftKnown);
    break;
  }
  case TargetOpcode::G_UADDO:
  case TargetOpcode::G_UADDE:
  case TargetOpcode::G_SADDO:
  case TargetOpcode::G_SADDE: {
    if (MI.getOperand(1).getReg() == R) {
      // If we know the result of a compare has the top bits zero, use this
      // info.
      if (TL.getBooleanContents(DstTy.isVector(), false) ==
              TargetLowering::ZeroOrOneBooleanContent &&
          BitWidth > 1)
        Known.Zero.setBitsFrom(1);
      break;
    }

    assert(MI.getOperand(0).getReg() == R &&
           "We only compute knownbits for the sum here.");
    // With [US]ADDE, a carry bit may be added in.
    KnownBits Carry(1);
    if (Opcode == TargetOpcode::G_UADDE || Opcode == TargetOpcode::G_SADDE) {
      computeKnownBitsImpl(MI.getOperand(4).getReg(), Carry, DemandedElts,
                           Depth + 1);
      // Carry has bit width 1
      Carry = Carry.trunc(1);
    } else {
      Carry.setAllZero();
    }

    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(3).getReg(), Known2, DemandedElts,
                         Depth + 1);
    Known = KnownBits::computeForAddCarry(Known, Known2, Carry);
    break;
  }
  case TargetOpcode::G_USUBO:
  case TargetOpcode::G_USUBE:
  case TargetOpcode::G_SSUBO:
  case TargetOpcode::G_SSUBE:
  case TargetOpcode::G_UMULO:
  case TargetOpcode::G_SMULO: {
    if (MI.getOperand(1).getReg() == R) {
      // If we know the result of a compare has the top bits zero, use this
      // info.
      if (TL.getBooleanContents(DstTy.isVector(), false) ==
              TargetLowering::ZeroOrOneBooleanContent &&
          BitWidth > 1)
        Known.Zero.setBitsFrom(1);
    }
    break;
  }
  case TargetOpcode::G_CTTZ:
  case TargetOpcode::G_CTTZ_ZERO_UNDEF: {
    KnownBits SrcOpKnown;
    computeKnownBitsImpl(MI.getOperand(1).getReg(), SrcOpKnown, DemandedElts,
                         Depth + 1);
    // If we have a known 1, its position is our upper bound
    unsigned PossibleTZ = SrcOpKnown.countMaxTrailingZeros();
    unsigned LowBits = llvm::bit_width(PossibleTZ);
    Known.Zero.setBitsFrom(LowBits);
    break;
  }
  case TargetOpcode::G_CTLZ:
  case TargetOpcode::G_CTLZ_ZERO_UNDEF: {
    KnownBits SrcOpKnown;
    computeKnownBitsImpl(MI.getOperand(1).getReg(), SrcOpKnown, DemandedElts,
                         Depth + 1);
    // If we have a known 1, its position is our upper bound.
    unsigned PossibleLZ = SrcOpKnown.countMaxLeadingZeros();
    unsigned LowBits = llvm::bit_width(PossibleLZ);
    Known.Zero.setBitsFrom(LowBits);
    break;
  }
  case TargetOpcode::G_CTLS: {
    Register Reg = MI.getOperand(1).getReg();
    unsigned MinRedundantSignBits = computeNumSignBits(Reg, Depth + 1) - 1;

    unsigned MaxUpperRedundantSignBits = MRI.getType(Reg).getScalarSizeInBits();

    ConstantRange Range(APInt(BitWidth, MinRedundantSignBits),
                        APInt(BitWidth, MaxUpperRedundantSignBits));

    Known = Range.toKnownBits();
    break;
  }
  case TargetOpcode::G_EXTRACT_VECTOR_ELT: {
    GExtractVectorElement &Extract = cast<GExtractVectorElement>(MI);
    Register InVec = Extract.getVectorReg();
    Register EltNo = Extract.getIndexReg();

    auto ConstEltNo = getIConstantVRegVal(EltNo, MRI);

    LLT VecVT = MRI.getType(InVec);
    // computeKnownBits not yet implemented for scalable vectors.
    if (VecVT.isScalableVector())
      break;

    const unsigned EltBitWidth = VecVT.getScalarSizeInBits();
    const unsigned NumSrcElts = VecVT.getNumElements();
    // A return type different from the vector's element type may lead to
    // issues with pattern selection. Bail out to avoid that.
    if (BitWidth > EltBitWidth)
      break;

    Known.Zero.setAllBits();
    Known.One.setAllBits();

    // If we know the element index, just demand that vector element, else for
    // an unknown element index, ignore DemandedElts and demand them all.
    APInt DemandedSrcElts = APInt::getAllOnes(NumSrcElts);
    if (ConstEltNo && ConstEltNo->ult(NumSrcElts))
      DemandedSrcElts =
          APInt::getOneBitSet(NumSrcElts, ConstEltNo->getZExtValue());

    computeKnownBitsImpl(InVec, Known, DemandedSrcElts, Depth + 1);
    break;
  }
  case TargetOpcode::G_SHUFFLE_VECTOR: {
    APInt DemandedLHS, DemandedRHS;
    // Collect the known bits that are shared by every vector element referenced
    // by the shuffle.
    unsigned NumElts = MRI.getType(MI.getOperand(1).getReg()).getNumElements();
    if (!getShuffleDemandedElts(NumElts, MI.getOperand(3).getShuffleMask(),
                                DemandedElts, DemandedLHS, DemandedRHS))
      break;

    // Known bits are the values that are shared by every demanded element.
    Known.Zero.setAllBits();
    Known.One.setAllBits();
    if (!!DemandedLHS) {
      computeKnownBitsImpl(MI.getOperand(1).getReg(), Known2, DemandedLHS,
                           Depth + 1);
      Known = Known.intersectWith(Known2);
    }
    // If we don't know any bits, early out.
    if (Known.isUnknown())
      break;
    if (!!DemandedRHS) {
      computeKnownBitsImpl(MI.getOperand(2).getReg(), Known2, DemandedRHS,
                           Depth + 1);
      Known = Known.intersectWith(Known2);
    }
    break;
  }
  case TargetOpcode::G_CONCAT_VECTORS: {
    if (MRI.getType(MI.getOperand(0).getReg()).isScalableVector())
      break;
    // Split DemandedElts and test each of the demanded subvectors.
    Known.Zero.setAllBits();
    Known.One.setAllBits();
    unsigned NumSubVectorElts =
        MRI.getType(MI.getOperand(1).getReg()).getNumElements();

    for (const auto &[I, MO] : enumerate(drop_begin(MI.operands()))) {
      APInt DemandedSub =
          DemandedElts.extractBits(NumSubVectorElts, I * NumSubVectorElts);
      if (!!DemandedSub) {
        computeKnownBitsImpl(MO.getReg(), Known2, DemandedSub, Depth + 1);

        Known = Known.intersectWith(Known2);
      }
      // If we don't know any bits, early out.
      if (Known.isUnknown())
        break;
    }
    break;
  }
  case TargetOpcode::G_ABS: {
    Register SrcReg = MI.getOperand(1).getReg();
    computeKnownBitsImpl(SrcReg, Known, DemandedElts, Depth + 1);
    Known = Known.abs();
    Known.Zero.setHighBits(computeNumSignBits(SrcReg, DemandedElts, Depth + 1) -
                           1);
    break;
  }
  }

  LLVM_DEBUG(dumpResult(MI, Known, Depth));
}

void GISelValueTracking::computeKnownFPClass(Register R, KnownFPClass &Known,
                                             FPClassTest InterestedClasses,
                                             unsigned Depth) {
  LLT Ty = MRI.getType(R);
  APInt DemandedElts =
      Ty.isFixedVector() ? APInt::getAllOnes(Ty.getNumElements()) : APInt(1, 1);
  computeKnownFPClass(R, DemandedElts, InterestedClasses, Known, Depth);
}

/// Return true if this value is known to be the fractional part x - floor(x),
/// which lies in [0, 1). This implies the value cannot introduce overflow in a
/// fmul when the other operand is known finite.
static bool isAbsoluteValueULEOne(Register R, const MachineRegisterInfo &MRI) {
  using namespace MIPatternMatch;
  Register SubX;
  return mi_match(R, MRI, m_GFSub(m_Reg(SubX), m_GFFloor(m_DeferredReg(SubX))));
}

void GISelValueTracking::computeKnownFPClassForFPTrunc(
    const MachineInstr &MI, const APInt &DemandedElts,
    FPClassTest InterestedClasses, KnownFPClass &Known, unsigned Depth) {
  if ((InterestedClasses & (KnownFPClass::OrderedLessThanZeroMask | fcNan)) ==
      fcNone)
    return;

  Register Val = MI.getOperand(1).getReg();
  KnownFPClass KnownSrc;
  computeKnownFPClass(Val, DemandedElts, InterestedClasses, KnownSrc,
                      Depth + 1);
  Known = KnownFPClass::fptrunc(KnownSrc);
}

void GISelValueTracking::computeKnownFPClass(Register R,
                                             const APInt &DemandedElts,
                                             FPClassTest InterestedClasses,
                                             KnownFPClass &Known,
                                             unsigned Depth) {
  assert(Known.isUnknown() && "should not be called with known information");

  if (!DemandedElts) {
    // No demanded elts, better to assume we don't know anything.
    Known.resetAll();
    return;
  }

  assert(Depth <= MaxAnalysisRecursionDepth && "Limit Search Depth");

  MachineInstr &MI = *MRI.getVRegDef(R);
  unsigned Opcode = MI.getOpcode();
  LLT DstTy = MRI.getType(R);

  if (!DstTy.isValid()) {
    Known.resetAll();
    return;
  }

  if (auto Cst = GFConstant::getConstant(R, MRI)) {
    switch (Cst->getKind()) {
    case GFConstant::GFConstantKind::Scalar: {
      auto APF = Cst->getScalarValue();
      Known.KnownFPClasses = APF.classify();
      Known.SignBit = APF.isNegative();
      break;
    }
    case GFConstant::GFConstantKind::FixedVector: {
      Known.KnownFPClasses = fcNone;
      bool SignBitAllZero = true;
      bool SignBitAllOne = true;

      for (auto C : *Cst) {
        Known.KnownFPClasses |= C.classify();
        if (C.isNegative())
          SignBitAllZero = false;
        else
          SignBitAllOne = false;
      }

      if (SignBitAllOne != SignBitAllZero)
        Known.SignBit = SignBitAllOne;

      break;
    }
    case GFConstant::GFConstantKind::ScalableVector: {
      Known.resetAll();
      break;
    }
    }

    return;
  }

  FPClassTest KnownNotFromFlags = fcNone;
  if (MI.getFlag(MachineInstr::MIFlag::FmNoNans))
    KnownNotFromFlags |= fcNan;
  if (MI.getFlag(MachineInstr::MIFlag::FmNoInfs))
    KnownNotFromFlags |= fcInf;

  // We no longer need to find out about these bits from inputs if we can
  // assume this from flags/attributes.
  InterestedClasses &= ~KnownNotFromFlags;

  llvm::scope_exit ClearClassesFromFlags(
      [=, &Known] { Known.knownNot(KnownNotFromFlags); });

  // All recursive calls that increase depth must come after this.
  if (Depth == MaxAnalysisRecursionDepth)
    return;

  const MachineFunction *MF = MI.getMF();

  switch (Opcode) {
  default:
    TL.computeKnownFPClassForTargetInstr(*this, R, Known, DemandedElts, MRI,
                                         Depth);
    break;
  case TargetOpcode::G_FNEG: {
    Register Val = MI.getOperand(1).getReg();
    computeKnownFPClass(Val, DemandedElts, InterestedClasses, Known, Depth + 1);
    Known.fneg();
    break;
  }
  case TargetOpcode::G_SELECT: {
    GSelect &SelMI = cast<GSelect>(MI);
    Register Cond = SelMI.getCondReg();
    Register LHS = SelMI.getTrueReg();
    Register RHS = SelMI.getFalseReg();

    FPClassTest FilterLHS = fcAllFlags;
    FPClassTest FilterRHS = fcAllFlags;

    Register TestedValue;
    FPClassTest MaskIfTrue = fcAllFlags;
    FPClassTest MaskIfFalse = fcAllFlags;
    FPClassTest ClassVal = fcNone;

    CmpInst::Predicate Pred;
    Register CmpLHS, CmpRHS;
    if (mi_match(Cond, MRI,
                 m_GFCmp(m_Pred(Pred), m_Reg(CmpLHS), m_Reg(CmpRHS)))) {
      // If the select filters out a value based on the class, it no longer
      // participates in the class of the result

      // TODO: In some degenerate cases we can infer something if we try again
      // without looking through sign operations.
      bool LookThroughFAbsFNeg = CmpLHS != LHS && CmpLHS != RHS;
      std::tie(TestedValue, MaskIfTrue, MaskIfFalse) =
          fcmpImpliesClass(Pred, *MF, CmpLHS, CmpRHS, LookThroughFAbsFNeg);
    } else if (mi_match(
                   Cond, MRI,
                   m_GIsFPClass(m_Reg(TestedValue), m_FPClassTest(ClassVal)))) {
      FPClassTest TestedMask = ClassVal;
      MaskIfTrue = TestedMask;
      MaskIfFalse = ~TestedMask;
    }

    if (TestedValue == LHS) {
      // match !isnan(x) ? x : y
      FilterLHS = MaskIfTrue;
    } else if (TestedValue == RHS) { // && IsExactClass
      // match !isnan(x) ? y : x
      FilterRHS = MaskIfFalse;
    }

    KnownFPClass Known2;
    computeKnownFPClass(LHS, DemandedElts, InterestedClasses & FilterLHS, Known,
                        Depth + 1);
    Known.KnownFPClasses &= FilterLHS;

    computeKnownFPClass(RHS, DemandedElts, InterestedClasses & FilterRHS,
                        Known2, Depth + 1);
    Known2.KnownFPClasses &= FilterRHS;

    Known |= Known2;
    break;
  }
  case TargetOpcode::G_FCOPYSIGN: {
    Register Magnitude = MI.getOperand(1).getReg();
    Register Sign = MI.getOperand(2).getReg();

    KnownFPClass KnownSign;

    computeKnownFPClass(Magnitude, DemandedElts, InterestedClasses, Known,
                        Depth + 1);
    computeKnownFPClass(Sign, DemandedElts, InterestedClasses, KnownSign,
                        Depth + 1);
    Known.copysign(KnownSign);
    break;
  }
  case TargetOpcode::G_FMA:
  case TargetOpcode::G_STRICT_FMA:
  case TargetOpcode::G_FMAD: {
    if ((InterestedClasses & fcNegative) == fcNone)
      break;

    Register A = MI.getOperand(1).getReg();
    Register B = MI.getOperand(2).getReg();
    Register C = MI.getOperand(3).getReg();

    DenormalMode Mode =
        MF->getDenormalMode(getFltSemanticForLLT(DstTy.getScalarType()));

    if (A == B && isGuaranteedNotToBeUndef(A, MRI, Depth + 1)) {
      // x * x + y
      KnownFPClass KnownSrc, KnownAddend;
      computeKnownFPClass(C, DemandedElts, InterestedClasses, KnownAddend,
                          Depth + 1);
      computeKnownFPClass(A, DemandedElts, InterestedClasses, KnownSrc,
                          Depth + 1);
      if (KnownNotFromFlags) {
        KnownSrc.knownNot(KnownNotFromFlags);
        KnownAddend.knownNot(KnownNotFromFlags);
      }
      Known = KnownFPClass::fma_square(KnownSrc, KnownAddend, Mode);
    } else {
      KnownFPClass KnownSrc[3];
      computeKnownFPClass(A, DemandedElts, InterestedClasses, KnownSrc[0],
                          Depth + 1);
      if (KnownSrc[0].isUnknown())
        break;
      computeKnownFPClass(B, DemandedElts, InterestedClasses, KnownSrc[1],
                          Depth + 1);
      if (KnownSrc[1].isUnknown())
        break;
      computeKnownFPClass(C, DemandedElts, InterestedClasses, KnownSrc[2],
                          Depth + 1);
      if (KnownSrc[2].isUnknown())
        break;
      if (KnownNotFromFlags) {
        KnownSrc[0].knownNot(KnownNotFromFlags);
        KnownSrc[1].knownNot(KnownNotFromFlags);
        KnownSrc[2].knownNot(KnownNotFromFlags);
      }
      Known = KnownFPClass::fma(KnownSrc[0], KnownSrc[1], KnownSrc[2], Mode);
    }
    break;
  }
  case TargetOpcode::G_FSQRT:
  case TargetOpcode::G_STRICT_FSQRT: {
    KnownFPClass KnownSrc;
    FPClassTest InterestedSrcs = InterestedClasses;
    if (InterestedClasses & fcNan)
      InterestedSrcs |= KnownFPClass::OrderedLessThanZeroMask;

    Register Val = MI.getOperand(1).getReg();
    computeKnownFPClass(Val, DemandedElts, InterestedSrcs, KnownSrc, Depth + 1);

    DenormalMode Mode =
        MF->getDenormalMode(getFltSemanticForLLT(DstTy.getScalarType()));
    Known = KnownFPClass::sqrt(KnownSrc, Mode);
    if (MI.getFlag(MachineInstr::MIFlag::FmNsz))
      Known.knownNot(fcNegZero);
    break;
  }
  case TargetOpcode::G_FABS: {
    if ((InterestedClasses & (fcNan | fcPositive)) != fcNone) {
      Register Val = MI.getOperand(1).getReg();
      // If we only care about the sign bit we don't need to inspect the
      // operand.
      computeKnownFPClass(Val, DemandedElts, InterestedClasses, Known,
                          Depth + 1);
    }
    Known.fabs();
    break;
  }
  case TargetOpcode::G_FATAN2: {
    Register Y = MI.getOperand(1).getReg();
    Register X = MI.getOperand(2).getReg();
    KnownFPClass KnownY, KnownX;
    computeKnownFPClass(Y, DemandedElts, InterestedClasses, KnownY, Depth + 1);
    computeKnownFPClass(X, DemandedElts, InterestedClasses, KnownX, Depth + 1);
    Known = KnownFPClass::atan2(KnownY, KnownX);
    break;
  }
  case TargetOpcode::G_FSINH: {
    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Val, DemandedElts, InterestedClasses, KnownSrc,
                        Depth + 1);
    Known = KnownFPClass::sinh(KnownSrc);
    break;
  }
  case TargetOpcode::G_FCOSH: {
    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Val, DemandedElts, InterestedClasses, KnownSrc,
                        Depth + 1);
    Known = KnownFPClass::cosh(KnownSrc);
    break;
  }
  case TargetOpcode::G_FTANH: {
    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Val, DemandedElts, InterestedClasses, KnownSrc,
                        Depth + 1);
    Known = KnownFPClass::tanh(KnownSrc);
    break;
  }
  case TargetOpcode::G_FASIN: {
    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Val, DemandedElts, InterestedClasses, KnownSrc,
                        Depth + 1);
    Known = KnownFPClass::asin(KnownSrc);
    break;
  }
  case TargetOpcode::G_FACOS: {
    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Val, DemandedElts, InterestedClasses, KnownSrc,
                        Depth + 1);
    Known = KnownFPClass::acos(KnownSrc);
    break;
  }
  case TargetOpcode::G_FATAN: {
    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Val, DemandedElts, InterestedClasses, KnownSrc,
                        Depth + 1);
    Known = KnownFPClass::atan(KnownSrc);
    break;
  }
  case TargetOpcode::G_FTAN: {
    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Val, DemandedElts, InterestedClasses, KnownSrc,
                        Depth + 1);
    Known = KnownFPClass::tan(KnownSrc);
    break;
  }
  case TargetOpcode::G_FSIN:
  case TargetOpcode::G_FCOS: {
    // Return NaN on infinite inputs.
    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Val, DemandedElts, InterestedClasses, KnownSrc,
                        Depth + 1);
    Known = Opcode == TargetOpcode::G_FCOS ? KnownFPClass::cos(KnownSrc)
                                           : KnownFPClass::sin(KnownSrc);
    break;
  }
  case TargetOpcode::G_FSINCOS: {
    // Operand layout: (sin_dst, cos_dst, src)
    Register Src = MI.getOperand(2).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Src, DemandedElts, InterestedClasses, KnownSrc,
                        Depth + 1);
    if (R == MI.getOperand(0).getReg())
      Known = KnownFPClass::sin(KnownSrc);
    else
      Known = KnownFPClass::cos(KnownSrc);
    break;
  }
  case TargetOpcode::G_FMAXNUM:
  case TargetOpcode::G_FMINNUM:
  case TargetOpcode::G_FMINNUM_IEEE:
  case TargetOpcode::G_FMAXIMUM:
  case TargetOpcode::G_FMINIMUM:
  case TargetOpcode::G_FMAXNUM_IEEE:
  case TargetOpcode::G_FMAXIMUMNUM:
  case TargetOpcode::G_FMINIMUMNUM: {
    Register LHS = MI.getOperand(1).getReg();
    Register RHS = MI.getOperand(2).getReg();
    KnownFPClass KnownLHS, KnownRHS;

    computeKnownFPClass(LHS, DemandedElts, InterestedClasses, KnownLHS,
                        Depth + 1);
    computeKnownFPClass(RHS, DemandedElts, InterestedClasses, KnownRHS,
                        Depth + 1);

    KnownFPClass::MinMaxKind Kind;
    switch (Opcode) {
    case TargetOpcode::G_FMINIMUM:
      Kind = KnownFPClass::MinMaxKind::minimum;
      break;
    case TargetOpcode::G_FMAXIMUM:
      Kind = KnownFPClass::MinMaxKind::maximum;
      break;
    case TargetOpcode::G_FMINIMUMNUM:
      Kind = KnownFPClass::MinMaxKind::minimumnum;
      break;
    case TargetOpcode::G_FMAXIMUMNUM:
      Kind = KnownFPClass::MinMaxKind::maximumnum;
      break;
    case TargetOpcode::G_FMINNUM:
    case TargetOpcode::G_FMINNUM_IEEE:
      Kind = KnownFPClass::MinMaxKind::minnum;
      break;
    case TargetOpcode::G_FMAXNUM:
    case TargetOpcode::G_FMAXNUM_IEEE:
      Kind = KnownFPClass::MinMaxKind::maxnum;
      break;
    default:
      llvm_unreachable("unhandled min/max opcode");
    }

    DenormalMode Mode =
        MF->getDenormalMode(getFltSemanticForLLT(DstTy.getScalarType()));
    Known = KnownFPClass::minMaxLike(KnownLHS, KnownRHS, Kind, Mode);
    break;
  }
  case TargetOpcode::G_FCANONICALIZE: {
    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Val, DemandedElts, InterestedClasses, KnownSrc,
                        Depth + 1);

    LLT Ty = MRI.getType(Val).getScalarType();
    const fltSemantics &FPType = getFltSemanticForLLT(Ty);
    DenormalMode DenormMode = MF->getDenormalMode(FPType);
    Known = KnownFPClass::canonicalize(KnownSrc, DenormMode);
    break;
  }
  case TargetOpcode::G_VECREDUCE_FMAX:
  case TargetOpcode::G_VECREDUCE_FMIN:
  case TargetOpcode::G_VECREDUCE_FMAXIMUM:
  case TargetOpcode::G_VECREDUCE_FMINIMUM: {
    Register Val = MI.getOperand(1).getReg();
    // reduce min/max will choose an element from one of the vector elements,
    // so we can infer and class information that is common to all elements.

    Known =
        computeKnownFPClass(Val, MI.getFlags(), InterestedClasses, Depth + 1);
    // Can only propagate sign if output is never NaN.
    if (!Known.isKnownNeverNaN())
      Known.SignBit.reset();
    break;
  }
  case TargetOpcode::G_FFLOOR:
  case TargetOpcode::G_FCEIL:
  case TargetOpcode::G_FRINT:
  case TargetOpcode::G_FNEARBYINT:
  case TargetOpcode::G_INTRINSIC_FPTRUNC_ROUND:
  case TargetOpcode::G_INTRINSIC_ROUND:
  case TargetOpcode::G_INTRINSIC_ROUNDEVEN:
  case TargetOpcode::G_INTRINSIC_TRUNC: {
    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    FPClassTest InterestedSrcs = InterestedClasses;
    if (InterestedSrcs & fcPosFinite)
      InterestedSrcs |= fcPosFinite;
    if (InterestedSrcs & fcNegFinite)
      InterestedSrcs |= fcNegFinite;
    computeKnownFPClass(Val, DemandedElts, InterestedSrcs, KnownSrc, Depth + 1);

    // TODO: handle multi unit FPTypes once LLT FPInfo lands
    bool IsTrunc = Opcode == TargetOpcode::G_INTRINSIC_TRUNC;
    Known = KnownFPClass::roundToIntegral(KnownSrc, IsTrunc,
                                          /*IsMultiUnitFPType=*/false);
    break;
  }
  case TargetOpcode::G_FEXP:
  case TargetOpcode::G_FEXP2:
  case TargetOpcode::G_FEXP10: {
    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Val, DemandedElts, InterestedClasses, KnownSrc,
                        Depth + 1);
    Known = KnownFPClass::exp(KnownSrc);
    break;
  }
  case TargetOpcode::G_FLOG:
  case TargetOpcode::G_FLOG2:
  case TargetOpcode::G_FLOG10: {
    // log(+inf) -> +inf
    // log([+-]0.0) -> -inf
    // log(-inf) -> nan
    // log(-x) -> nan
    if ((InterestedClasses & (fcNan | fcInf)) == fcNone)
      break;

    FPClassTest InterestedSrcs = InterestedClasses;
    if ((InterestedClasses & fcNegInf) != fcNone)
      InterestedSrcs |= fcZero | fcSubnormal;
    if ((InterestedClasses & fcNan) != fcNone)
      InterestedSrcs |= fcNan | fcNegative;

    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Val, DemandedElts, InterestedSrcs, KnownSrc, Depth + 1);

    LLT Ty = MRI.getType(Val).getScalarType();
    const fltSemantics &FltSem = getFltSemanticForLLT(Ty);
    DenormalMode Mode = MF->getDenormalMode(FltSem);
    Known = KnownFPClass::log(KnownSrc, Mode);
    break;
  }
  case TargetOpcode::G_FPOWI: {
    if ((InterestedClasses & (fcNan | fcInf | fcNegative)) == fcNone)
      break;

    Register Exp = MI.getOperand(2).getReg();
    LLT ExpTy = MRI.getType(Exp);
    KnownBits ExponentKnownBits = getKnownBits(
        Exp, ExpTy.isVector() ? DemandedElts : APInt(1, 1), Depth + 1);

    FPClassTest InterestedSrcs = fcNone;
    if (InterestedClasses & fcNan)
      InterestedSrcs |= fcNan;
    if (!ExponentKnownBits.isZero()) {
      if (InterestedClasses & fcInf)
        InterestedSrcs |= fcFinite | fcInf;
      if ((InterestedClasses & fcNegative) && !ExponentKnownBits.isEven())
        InterestedSrcs |= fcNegative;
    }

    KnownFPClass KnownSrc;
    if (InterestedSrcs != fcNone) {
      Register Val = MI.getOperand(1).getReg();
      computeKnownFPClass(Val, DemandedElts, InterestedSrcs, KnownSrc,
                          Depth + 1);
    }

    Known = KnownFPClass::powi(KnownSrc, ExponentKnownBits);
    break;
  }
  case TargetOpcode::G_FLDEXP:
  case TargetOpcode::G_STRICT_FLDEXP: {
    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Val, DemandedElts, InterestedClasses, KnownSrc,
                        Depth + 1);

    // Can refine inf/zero handling based on the exponent operand.
    const FPClassTest ExpInfoMask = fcZero | fcSubnormal | fcInf;
    KnownBits ExpBits;
    if ((KnownSrc.KnownFPClasses & ExpInfoMask) != fcNone) {
      Register ExpReg = MI.getOperand(2).getReg();
      LLT ExpTy = MRI.getType(ExpReg);
      ExpBits = getKnownBits(
          ExpReg, ExpTy.isVector() ? DemandedElts : APInt(1, 1), Depth + 1);
    }

    LLT ScalarTy = DstTy.getScalarType();
    const fltSemantics &Flt = getFltSemanticForLLT(ScalarTy);
    DenormalMode Mode = MF->getDenormalMode(Flt);
    Known = KnownFPClass::ldexp(KnownSrc, ExpBits, Flt, Mode);
    break;
  }
  case TargetOpcode::G_FADD:
  case TargetOpcode::G_STRICT_FADD:
  case TargetOpcode::G_FSUB:
  case TargetOpcode::G_STRICT_FSUB: {
    Register LHS = MI.getOperand(1).getReg();
    Register RHS = MI.getOperand(2).getReg();
    bool IsAdd = (Opcode == TargetOpcode::G_FADD ||
                  Opcode == TargetOpcode::G_STRICT_FADD);
    bool WantNegative =
        IsAdd &&
        (InterestedClasses & KnownFPClass::OrderedLessThanZeroMask) != fcNone;
    bool WantNaN = (InterestedClasses & fcNan) != fcNone;
    bool WantNegZero = (InterestedClasses & fcNegZero) != fcNone;

    if (!WantNaN && !WantNegative && !WantNegZero) {
      break;
    }

    DenormalMode Mode =
        MF->getDenormalMode(getFltSemanticForLLT(DstTy.getScalarType()));

    FPClassTest InterestedSrcs = InterestedClasses;
    if (WantNegative)
      InterestedSrcs |= KnownFPClass::OrderedLessThanZeroMask;
    if (InterestedClasses & fcNan)
      InterestedSrcs |= fcInf;

    // Special case fadd x, x (canonical form of fmul x, 2).
    if (IsAdd && LHS == RHS && isGuaranteedNotToBeUndef(LHS, MRI, Depth + 1)) {
      KnownFPClass KnownSelf;
      computeKnownFPClass(LHS, DemandedElts, InterestedSrcs, KnownSelf,
                          Depth + 1);
      Known = KnownFPClass::fadd_self(KnownSelf, Mode);
      break;
    }

    KnownFPClass KnownLHS, KnownRHS;
    computeKnownFPClass(RHS, DemandedElts, InterestedSrcs, KnownRHS, Depth + 1);

    if ((WantNaN && KnownRHS.isKnownNeverNaN()) ||
        (WantNegative && KnownRHS.cannotBeOrderedLessThanZero()) ||
        WantNegZero || !IsAdd) {
      // RHS is canonically cheaper to compute. Skip inspecting the LHS if
      // there's no point.
      computeKnownFPClass(LHS, DemandedElts, InterestedSrcs, KnownLHS,
                          Depth + 1);
    }

    if (IsAdd)
      Known = KnownFPClass::fadd(KnownLHS, KnownRHS, Mode);
    else
      Known = KnownFPClass::fsub(KnownLHS, KnownRHS, Mode);
    break;
  }
  case TargetOpcode::G_FMUL:
  case TargetOpcode::G_STRICT_FMUL: {
    Register LHS = MI.getOperand(1).getReg();
    Register RHS = MI.getOperand(2).getReg();
    DenormalMode Mode =
        MF->getDenormalMode(getFltSemanticForLLT(DstTy.getScalarType()));

    // X * X is always non-negative or a NaN (use square() for precision).
    if (LHS == RHS && isGuaranteedNotToBeUndef(LHS, MRI, Depth + 1)) {
      KnownFPClass KnownSrc;
      computeKnownFPClass(LHS, DemandedElts, fcAllFlags, KnownSrc, Depth + 1);
      Known = KnownFPClass::square(KnownSrc, Mode);
    } else {
      // If RHS is a scalar constant, use the more precise APFloat overload.
      auto RHSCst = GFConstant::getConstant(RHS, MRI);
      if (RHSCst && RHSCst->getKind() == GFConstant::GFConstantKind::Scalar) {
        KnownFPClass KnownLHS;
        computeKnownFPClass(LHS, DemandedElts, fcAllFlags, KnownLHS, Depth + 1);
        Known = KnownFPClass::fmul(KnownLHS, RHSCst->getScalarValue(), Mode);
      } else {
        KnownFPClass KnownLHS, KnownRHS;
        computeKnownFPClass(RHS, DemandedElts, fcAllFlags, KnownRHS, Depth + 1);
        computeKnownFPClass(LHS, DemandedElts, fcAllFlags, KnownLHS, Depth + 1);
        Known = KnownFPClass::fmul(KnownLHS, KnownRHS, Mode);

        // If one operand is known |x| <= 1 and the other is finite, the
        // product cannot overflow to infinity.
        if (KnownLHS.isKnownNever(fcInf) && isAbsoluteValueULEOne(RHS, MRI))
          Known.knownNot(fcInf);
        else if (KnownRHS.isKnownNever(fcInf) &&
                 isAbsoluteValueULEOne(LHS, MRI))
          Known.knownNot(fcInf);
      }
    }
    break;
  }
  case TargetOpcode::G_FDIV:
  case TargetOpcode::G_FREM: {
    Register LHS = MI.getOperand(1).getReg();
    Register RHS = MI.getOperand(2).getReg();

    if (Opcode == TargetOpcode::G_FREM)
      Known.knownNot(fcInf);

    DenormalMode Mode =
        MF->getDenormalMode(getFltSemanticForLLT(DstTy.getScalarType()));

    if (LHS == RHS && isGuaranteedNotToBeUndef(LHS, MRI, Depth + 1)) {
      if (Opcode == TargetOpcode::G_FDIV) {
        const bool WantNan = (InterestedClasses & fcNan) != fcNone;
        if (!WantNan) {
          // X / X is always exactly 1.0 or a NaN.
          Known.KnownFPClasses = fcPosNormal | fcNan;
          break;
        }
        KnownFPClass KnownSrc;
        computeKnownFPClass(LHS, DemandedElts,
                            fcNan | fcInf | fcZero | fcSubnormal, KnownSrc,
                            Depth + 1);
        Known = KnownFPClass::fdiv_self(KnownSrc, Mode);
      } else {
        const bool WantNan = (InterestedClasses & fcNan) != fcNone;
        if (!WantNan) {
          // X % X is always exactly [+-]0.0 or a NaN.
          Known.KnownFPClasses = fcZero | fcNan;
          break;
        }
        KnownFPClass KnownSrc;
        computeKnownFPClass(LHS, DemandedElts,
                            fcNan | fcInf | fcZero | fcSubnormal, KnownSrc,
                            Depth + 1);
        Known = KnownFPClass::frem_self(KnownSrc, Mode);
      }
      break;
    }

    const bool WantNan = (InterestedClasses & fcNan) != fcNone;
    const bool WantNegative = (InterestedClasses & fcNegative) != fcNone;
    const bool WantPositive = Opcode == TargetOpcode::G_FREM &&
                              (InterestedClasses & fcPositive) != fcNone;
    if (!WantNan && !WantNegative && !WantPositive) {
      break;
    }

    KnownFPClass KnownLHS, KnownRHS;

    computeKnownFPClass(RHS, DemandedElts, fcNan | fcInf | fcZero | fcNegative,
                        KnownRHS, Depth + 1);

    bool KnowSomethingUseful = KnownRHS.isKnownNeverNaN() ||
                               KnownRHS.isKnownNever(fcNegative) ||
                               KnownRHS.isKnownNever(fcPositive);

    if (KnowSomethingUseful || WantPositive) {
      computeKnownFPClass(LHS, DemandedElts, fcAllFlags, KnownLHS, Depth + 1);
    }

    if (Opcode == TargetOpcode::G_FDIV) {
      Known = KnownFPClass::fdiv(KnownLHS, KnownRHS, Mode);
    } else {
      // Inf REM x and x REM 0 produce NaN.
      if (KnownLHS.isKnownNeverNaN() && KnownRHS.isKnownNeverNaN() &&
          KnownLHS.isKnownNeverInfinity() &&
          KnownRHS.isKnownNeverLogicalZero(Mode)) {
        Known.knownNot(fcNan);
      }

      // The sign for frem is the same as the first operand.
      if (KnownLHS.cannotBeOrderedLessThanZero())
        Known.knownNot(KnownFPClass::OrderedLessThanZeroMask);
      if (KnownLHS.cannotBeOrderedGreaterThanZero())
        Known.knownNot(KnownFPClass::OrderedGreaterThanZeroMask);

      // See if we can be more aggressive about the sign of 0.
      if (KnownLHS.isKnownNever(fcNegative))
        Known.knownNot(fcNegative);
      if (KnownLHS.isKnownNever(fcPositive))
        Known.knownNot(fcPositive);
    }
    break;
  }
  case TargetOpcode::G_FFREXP: {
    // Only handle the mantissa output (operand 0); the exponent is an integer.
    if (R != MI.getOperand(0).getReg())
      break;
    Register Src = MI.getOperand(2).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Src, DemandedElts, InterestedClasses, KnownSrc,
                        Depth + 1);
    DenormalMode Mode =
        MF->getDenormalMode(getFltSemanticForLLT(DstTy.getScalarType()));
    Known = KnownFPClass::frexp_mant(KnownSrc, Mode);
    break;
  }
  case TargetOpcode::G_FPEXT: {
    Register Src = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Src, DemandedElts, InterestedClasses, KnownSrc,
                        Depth + 1);

    LLT DstScalarTy = DstTy.getScalarType();
    const fltSemantics &DstSem = getFltSemanticForLLT(DstScalarTy);
    LLT SrcTy = MRI.getType(Src).getScalarType();
    const fltSemantics &SrcSem = getFltSemanticForLLT(SrcTy);

    Known = KnownFPClass::fpext(KnownSrc, DstSem, SrcSem);
    break;
  }
  case TargetOpcode::G_FPTRUNC: {
    computeKnownFPClassForFPTrunc(MI, DemandedElts, InterestedClasses, Known,
                                  Depth);
    break;
  }
  case TargetOpcode::G_SITOFP:
  case TargetOpcode::G_UITOFP: {
    // Cannot produce nan
    Known.knownNot(fcNan);

    // Integers cannot be subnormal
    Known.knownNot(fcSubnormal);

    // sitofp and uitofp turn into +0.0 for zero.
    Known.knownNot(fcNegZero);

    // UIToFP is always non-negative regardless of known bits.
    if (Opcode == TargetOpcode::G_UITOFP)
      Known.signBitMustBeZero();

    // Only compute known bits if we can learn something useful from them.
    if (!(InterestedClasses & (fcPosZero | fcNormal | fcInf)))
      break;

    Register Val = MI.getOperand(1).getReg();
    LLT Ty = MRI.getType(Val);
    KnownBits IntKnown = getKnownBits(
        Val, Ty.isVector() ? DemandedElts : APInt(1, 1), Depth + 1);

    // If the integer is non-zero, the result cannot be +0.0.
    if (IntKnown.isNonZero())
      Known.knownNot(fcPosZero);

    if (Opcode == TargetOpcode::G_SITOFP) {
      // If the signed integer is known non-negative, the result is
      // non-negative. If the signed integer is known negative, the result is
      // negative.
      if (IntKnown.isNonNegative())
        Known.signBitMustBeZero();
      else if (IntKnown.isNegative())
        Known.signBitMustBeOne();
    }

    if (InterestedClasses & fcInf) {
      LLT FPTy = DstTy.getScalarType();
      const fltSemantics &FltSem = getFltSemanticForLLT(FPTy);

      // Compute the effective integer width after removing known-zero leading
      // bits, to check if the result can overflow to infinity.
      int IntSize = IntKnown.getBitWidth();
      if (Opcode == TargetOpcode::G_UITOFP)
        IntSize -= IntKnown.countMinLeadingZeros();
      else
        IntSize -= IntKnown.countMinSignBits();

      // If the exponent of the largest finite FP value can hold the largest
      // integer, the result of the cast must be finite.
      if (ilogb(APFloat::getLargest(FltSem)) >= IntSize)
        Known.knownNot(fcInf);
    }

    break;
  }
  // case TargetOpcode::G_MERGE_VALUES:
  case TargetOpcode::G_BUILD_VECTOR:
  case TargetOpcode::G_CONCAT_VECTORS: {
    GMergeLikeInstr &Merge = cast<GMergeLikeInstr>(MI);

    if (!DstTy.isFixedVector())
      break;

    bool First = true;
    for (unsigned Idx = 0; Idx < Merge.getNumSources(); ++Idx) {
      // We know the index we are inserting to, so clear it from Vec check.
      bool NeedsElt = DemandedElts[Idx];

      // Do we demand the inserted element?
      if (NeedsElt) {
        Register Src = Merge.getSourceReg(Idx);
        if (First) {
          computeKnownFPClass(Src, Known, InterestedClasses, Depth + 1);
          First = false;
        } else {
          KnownFPClass Known2;
          computeKnownFPClass(Src, Known2, InterestedClasses, Depth + 1);
          Known |= Known2;
        }

        // If we don't know any bits, early out.
        if (Known.isUnknown())
          break;
      }
    }

    break;
  }
  case TargetOpcode::G_EXTRACT_VECTOR_ELT: {
    // Look through extract element. If the index is non-constant or
    // out-of-range demand all elements, otherwise just the extracted
    // element.
    GExtractVectorElement &Extract = cast<GExtractVectorElement>(MI);
    Register Vec = Extract.getVectorReg();
    Register Idx = Extract.getIndexReg();

    auto CIdx = getIConstantVRegVal(Idx, MRI);

    LLT VecTy = MRI.getType(Vec);

    if (VecTy.isFixedVector()) {
      unsigned NumElts = VecTy.getNumElements();
      APInt DemandedVecElts = APInt::getAllOnes(NumElts);
      if (CIdx && CIdx->ult(NumElts))
        DemandedVecElts = APInt::getOneBitSet(NumElts, CIdx->getZExtValue());
      return computeKnownFPClass(Vec, DemandedVecElts, InterestedClasses, Known,
                                 Depth + 1);
    }

    break;
  }
  case TargetOpcode::G_INSERT_VECTOR_ELT: {
    GInsertVectorElement &Insert = cast<GInsertVectorElement>(MI);
    Register Vec = Insert.getVectorReg();
    Register Elt = Insert.getElementReg();
    Register Idx = Insert.getIndexReg();

    LLT VecTy = MRI.getType(Vec);

    if (VecTy.isScalableVector())
      return;

    auto CIdx = getIConstantVRegVal(Idx, MRI);

    unsigned NumElts = DemandedElts.getBitWidth();
    APInt DemandedVecElts = DemandedElts;
    bool NeedsElt = true;
    // If we know the index we are inserting to, clear it from Vec check.
    if (CIdx && CIdx->ult(NumElts)) {
      DemandedVecElts.clearBit(CIdx->getZExtValue());
      NeedsElt = DemandedElts[CIdx->getZExtValue()];
    }

    // Do we demand the inserted element?
    if (NeedsElt) {
      computeKnownFPClass(Elt, Known, InterestedClasses, Depth + 1);
      // If we don't know any bits, early out.
      if (Known.isUnknown())
        break;
    } else {
      Known.KnownFPClasses = fcNone;
    }

    // Do we need anymore elements from Vec?
    if (!DemandedVecElts.isZero()) {
      KnownFPClass Known2;
      computeKnownFPClass(Vec, DemandedVecElts, InterestedClasses, Known2,
                          Depth + 1);
      Known |= Known2;
    }

    break;
  }
  case TargetOpcode::G_SHUFFLE_VECTOR: {
    // For undef elements, we don't know anything about the common state of
    // the shuffle result.
    GShuffleVector &Shuf = cast<GShuffleVector>(MI);
    APInt DemandedLHS, DemandedRHS;
    if (DstTy.isScalableVector()) {
      assert(DemandedElts == APInt(1, 1));
      DemandedLHS = DemandedRHS = DemandedElts;
    } else {
      unsigned NumElts = MRI.getType(Shuf.getSrc1Reg()).getNumElements();
      if (!llvm::getShuffleDemandedElts(NumElts, Shuf.getMask(), DemandedElts,
                                        DemandedLHS, DemandedRHS)) {
        Known.resetAll();
        return;
      }
    }

    if (!!DemandedLHS) {
      Register LHS = Shuf.getSrc1Reg();
      computeKnownFPClass(LHS, DemandedLHS, InterestedClasses, Known,
                          Depth + 1);

      // If we don't know any bits, early out.
      if (Known.isUnknown())
        break;
    } else {
      Known.KnownFPClasses = fcNone;
    }

    if (!!DemandedRHS) {
      KnownFPClass Known2;
      Register RHS = Shuf.getSrc2Reg();
      computeKnownFPClass(RHS, DemandedRHS, InterestedClasses, Known2,
                          Depth + 1);
      Known |= Known2;
    }
    break;
  }
  case TargetOpcode::G_PHI: {
    // Cap PHI recursion below the global limit to avoid spending the entire
    // budget chasing loop back-edges (matches ValueTracking's
    // PhiRecursionLimit).
    if (Depth + 2 > MaxAnalysisRecursionDepth)
      break;
    // PHI's operands are a mix of registers and basic blocks interleaved.
    // We only care about the register ones.
    bool First = true;
    for (unsigned Idx = 1; Idx < MI.getNumOperands(); Idx += 2) {
      const MachineOperand &Src = MI.getOperand(Idx);
      Register SrcReg = Src.getReg();
      if (First) {
        computeKnownFPClass(SrcReg, DemandedElts, InterestedClasses, Known,
                            Depth + 1);
        First = false;
      } else {
        KnownFPClass Known2;
        computeKnownFPClass(SrcReg, DemandedElts, InterestedClasses, Known2,
                            Depth + 1);
        Known = Known.intersectWith(Known2);
      }
      if (Known.isUnknown())
        break;
    }
    break;
  }
  case TargetOpcode::COPY: {
    Register Src = MI.getOperand(1).getReg();

    if (!Src.isVirtual())
      return;

    computeKnownFPClass(Src, DemandedElts, InterestedClasses, Known, Depth + 1);
    break;
  }
  }
}

KnownFPClass
GISelValueTracking::computeKnownFPClass(Register R, const APInt &DemandedElts,
                                        FPClassTest InterestedClasses,
                                        unsigned Depth) {
  KnownFPClass KnownClasses;
  computeKnownFPClass(R, DemandedElts, InterestedClasses, KnownClasses, Depth);
  return KnownClasses;
}

KnownFPClass GISelValueTracking::computeKnownFPClass(
    Register R, FPClassTest InterestedClasses, unsigned Depth) {
  KnownFPClass Known;
  computeKnownFPClass(R, Known, InterestedClasses, Depth);
  return Known;
}

KnownFPClass GISelValueTracking::computeKnownFPClass(
    Register R, const APInt &DemandedElts, uint32_t Flags,
    FPClassTest InterestedClasses, unsigned Depth) {
  if (Flags & MachineInstr::MIFlag::FmNoNans)
    InterestedClasses &= ~fcNan;
  if (Flags & MachineInstr::MIFlag::FmNoInfs)
    InterestedClasses &= ~fcInf;

  KnownFPClass Result =
      computeKnownFPClass(R, DemandedElts, InterestedClasses, Depth);

  if (Flags & MachineInstr::MIFlag::FmNoNans)
    Result.KnownFPClasses &= ~fcNan;
  if (Flags & MachineInstr::MIFlag::FmNoInfs)
    Result.KnownFPClasses &= ~fcInf;
  return Result;
}

KnownFPClass GISelValueTracking::computeKnownFPClass(
    Register R, uint32_t Flags, FPClassTest InterestedClasses, unsigned Depth) {
  LLT Ty = MRI.getType(R);
  APInt DemandedElts =
      Ty.isFixedVector() ? APInt::getAllOnes(Ty.getNumElements()) : APInt(1, 1);
  return computeKnownFPClass(R, DemandedElts, Flags, InterestedClasses, Depth);
}

bool GISelValueTracking::isKnownNeverNaN(Register Val, bool SNaN) {
  const MachineInstr *DefMI = MRI.getVRegDef(Val);
  if (!DefMI)
    return false;

  if (DefMI->getFlag(MachineInstr::FmNoNans))
    return true;

  // IEEE 754 arithmetic operations always quiet signaling NaNs. Short-circuit
  // the value-tracking analysis for the SNaN-only case: if the defining op is
  // known to quiet sNaN, the output can never be an sNaN.
  if (SNaN) {
    switch (DefMI->getOpcode()) {
    default:
      break;
    case TargetOpcode::G_FADD:
    case TargetOpcode::G_STRICT_FADD:
    case TargetOpcode::G_FSUB:
    case TargetOpcode::G_STRICT_FSUB:
    case TargetOpcode::G_FMUL:
    case TargetOpcode::G_STRICT_FMUL:
    case TargetOpcode::G_FDIV:
    case TargetOpcode::G_FREM:
    case TargetOpcode::G_FMA:
    case TargetOpcode::G_STRICT_FMA:
    case TargetOpcode::G_FMAD:
    case TargetOpcode::G_FSQRT:
    case TargetOpcode::G_STRICT_FSQRT:
    // Note: G_FABS and G_FNEG are bit-manipulation ops that preserve sNaN
    // exactly (LLVM LangRef: "never change anything except possibly the sign
    // bit"). They must NOT be listed here.
    case TargetOpcode::G_FSIN:
    case TargetOpcode::G_FCOS:
    case TargetOpcode::G_FSINCOS:
    case TargetOpcode::G_FTAN:
    case TargetOpcode::G_FASIN:
    case TargetOpcode::G_FACOS:
    case TargetOpcode::G_FATAN:
    case TargetOpcode::G_FATAN2:
    case TargetOpcode::G_FSINH:
    case TargetOpcode::G_FCOSH:
    case TargetOpcode::G_FTANH:
    case TargetOpcode::G_FEXP:
    case TargetOpcode::G_FEXP2:
    case TargetOpcode::G_FEXP10:
    case TargetOpcode::G_FLOG:
    case TargetOpcode::G_FLOG2:
    case TargetOpcode::G_FLOG10:
    case TargetOpcode::G_FPOWI:
    case TargetOpcode::G_FLDEXP:
    case TargetOpcode::G_STRICT_FLDEXP:
    case TargetOpcode::G_FFREXP:
    case TargetOpcode::G_INTRINSIC_TRUNC:
    case TargetOpcode::G_INTRINSIC_ROUND:
    case TargetOpcode::G_INTRINSIC_ROUNDEVEN:
    case TargetOpcode::G_FFLOOR:
    case TargetOpcode::G_FCEIL:
    case TargetOpcode::G_FRINT:
    case TargetOpcode::G_FNEARBYINT:
    case TargetOpcode::G_FPEXT:
    case TargetOpcode::G_FPTRUNC:
    case TargetOpcode::G_FCANONICALIZE:
    case TargetOpcode::G_FMINNUM:
    case TargetOpcode::G_FMAXNUM:
    case TargetOpcode::G_FMINNUM_IEEE:
    case TargetOpcode::G_FMAXNUM_IEEE:
    case TargetOpcode::G_FMINIMUM:
    case TargetOpcode::G_FMAXIMUM:
    case TargetOpcode::G_FMINIMUMNUM:
    case TargetOpcode::G_FMAXIMUMNUM:
      return true;
    }
  }

  KnownFPClass FPClass = computeKnownFPClass(Val, SNaN ? fcSNan : fcNan);

  if (SNaN)
    return FPClass.isKnownNever(fcSNan);

  return FPClass.isKnownNeverNaN();
}

/// Compute number of sign bits for the intersection of \p Src0 and \p Src1
unsigned GISelValueTracking::computeNumSignBitsMin(Register Src0, Register Src1,
                                                   const APInt &DemandedElts,
                                                   unsigned Depth) {
  // Test src1 first, since we canonicalize simpler expressions to the RHS.
  unsigned Src1SignBits = computeNumSignBits(Src1, DemandedElts, Depth);
  if (Src1SignBits == 1)
    return 1;
  return std::min(computeNumSignBits(Src0, DemandedElts, Depth), Src1SignBits);
}

/// Compute the known number of sign bits with attached range metadata in the
/// memory operand. If this is an extending load, accounts for the behavior of
/// the high bits.
static unsigned computeNumSignBitsFromRangeMetadata(const GAnyLoad *Ld,
                                                    unsigned TyBits) {
  const MDNode *Ranges = Ld->getRanges();
  if (!Ranges)
    return 1;

  ConstantRange CR = getConstantRangeFromMetadata(*Ranges);
  if (TyBits > CR.getBitWidth()) {
    switch (Ld->getOpcode()) {
    case TargetOpcode::G_SEXTLOAD:
      CR = CR.signExtend(TyBits);
      break;
    case TargetOpcode::G_ZEXTLOAD:
      CR = CR.zeroExtend(TyBits);
      break;
    default:
      break;
    }
  }

  return std::min(CR.getSignedMin().getNumSignBits(),
                  CR.getSignedMax().getNumSignBits());
}

unsigned GISelValueTracking::computeNumSignBits(Register R,
                                                const APInt &DemandedElts,
                                                unsigned Depth) {
  MachineInstr &MI = *MRI.getVRegDef(R);
  unsigned Opcode = MI.getOpcode();

  if (Opcode == TargetOpcode::G_CONSTANT)
    return MI.getOperand(1).getCImm()->getValue().getNumSignBits();

  if (Depth == getMaxDepth())
    return 1;

  if (!DemandedElts)
    return 1; // No demanded elts, better to assume we don't know anything.

  LLT DstTy = MRI.getType(R);
  const unsigned TyBits = DstTy.getScalarSizeInBits();

  // Handle the case where this is called on a register that does not have a
  // type constraint. This is unlikely to occur except by looking through copies
  // but it is possible for the initial register being queried to be in this
  // state.
  if (!DstTy.isValid())
    return 1;

  unsigned FirstAnswer = 1;
  switch (Opcode) {
  case TargetOpcode::COPY: {
    MachineOperand &Src = MI.getOperand(1);
    if (Src.getReg().isVirtual() && Src.getSubReg() == 0 &&
        MRI.getType(Src.getReg()).isValid()) {
      // Don't increment Depth for this one since we didn't do any work.
      return computeNumSignBits(Src.getReg(), DemandedElts, Depth);
    }

    return 1;
  }
  case TargetOpcode::G_SEXT: {
    Register Src = MI.getOperand(1).getReg();
    LLT SrcTy = MRI.getType(Src);
    unsigned Tmp = DstTy.getScalarSizeInBits() - SrcTy.getScalarSizeInBits();
    return computeNumSignBits(Src, DemandedElts, Depth + 1) + Tmp;
  }
  case TargetOpcode::G_ASSERT_SEXT:
  case TargetOpcode::G_SEXT_INREG: {
    // Max of the input and what this extends.
    Register Src = MI.getOperand(1).getReg();
    unsigned SrcBits = MI.getOperand(2).getImm();
    unsigned InRegBits = TyBits - SrcBits + 1;
    return std::max(computeNumSignBits(Src, DemandedElts, Depth + 1),
                    InRegBits);
  }
  case TargetOpcode::G_LOAD: {
    GLoad *Ld = cast<GLoad>(&MI);
    if (DemandedElts != 1 || !getDataLayout().isLittleEndian())
      break;

    return computeNumSignBitsFromRangeMetadata(Ld, TyBits);
  }
  case TargetOpcode::G_SEXTLOAD: {
    GSExtLoad *Ld = cast<GSExtLoad>(&MI);

    // FIXME: We need an in-memory type representation.
    if (DstTy.isVector())
      return 1;

    unsigned NumBits = computeNumSignBitsFromRangeMetadata(Ld, TyBits);
    if (NumBits != 1)
      return NumBits;

    // e.g. i16->i32 = '17' bits known.
    const MachineMemOperand *MMO = *MI.memoperands_begin();
    return TyBits - MMO->getSizeInBits().getValue() + 1;
  }
  case TargetOpcode::G_ZEXTLOAD: {
    GZExtLoad *Ld = cast<GZExtLoad>(&MI);

    // FIXME: We need an in-memory type representation.
    if (DstTy.isVector())
      return 1;

    unsigned NumBits = computeNumSignBitsFromRangeMetadata(Ld, TyBits);
    if (NumBits != 1)
      return NumBits;

    // e.g. i16->i32 = '16' bits known.
    const MachineMemOperand *MMO = *MI.memoperands_begin();
    return TyBits - MMO->getSizeInBits().getValue();
  }
  case TargetOpcode::G_AND:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR: {
    Register Src1 = MI.getOperand(1).getReg();
    unsigned Src1NumSignBits =
        computeNumSignBits(Src1, DemandedElts, Depth + 1);
    if (Src1NumSignBits != 1) {
      Register Src2 = MI.getOperand(2).getReg();
      unsigned Src2NumSignBits =
          computeNumSignBits(Src2, DemandedElts, Depth + 1);
      FirstAnswer = std::min(Src1NumSignBits, Src2NumSignBits);
    }
    break;
  }
  case TargetOpcode::G_ASHR: {
    Register Src1 = MI.getOperand(1).getReg();
    Register Src2 = MI.getOperand(2).getReg();
    FirstAnswer = computeNumSignBits(Src1, DemandedElts, Depth + 1);
    if (auto C = getValidMinimumShiftAmount(Src2, DemandedElts, Depth + 1))
      FirstAnswer = std::min<uint64_t>(FirstAnswer + *C, TyBits);
    break;
  }
  case TargetOpcode::G_SHL: {
    Register Src1 = MI.getOperand(1).getReg();
    Register Src2 = MI.getOperand(2).getReg();
    if (std::optional<ConstantRange> ShAmtRange =
            getValidShiftAmountRange(Src2, DemandedElts, Depth + 1)) {
      uint64_t MaxShAmt = ShAmtRange->getUnsignedMax().getZExtValue();
      uint64_t MinShAmt = ShAmtRange->getUnsignedMin().getZExtValue();

      MachineInstr &ExtMI = *MRI.getVRegDef(Src1);
      unsigned ExtOpc = ExtMI.getOpcode();

      // Try to look through ZERO/SIGN/ANY_EXTEND. If all extended bits are
      // shifted out, then we can compute the number of sign bits for the
      // operand being extended. A future improvement could be to pass along the
      // "shifted left by" information in the recursive calls to
      // ComputeKnownSignBits. Allowing us to handle this more generically.
      if (ExtOpc == TargetOpcode::G_SEXT || ExtOpc == TargetOpcode::G_ZEXT ||
          ExtOpc == TargetOpcode::G_ANYEXT) {
        LLT ExtTy = MRI.getType(Src1);
        Register Extendee = ExtMI.getOperand(1).getReg();
        LLT ExtendeeTy = MRI.getType(Extendee);
        uint64_t SizeDiff =
            ExtTy.getScalarSizeInBits() - ExtendeeTy.getScalarSizeInBits();

        if (SizeDiff <= MinShAmt) {
          unsigned Tmp =
              SizeDiff + computeNumSignBits(Extendee, DemandedElts, Depth + 1);
          if (MaxShAmt < Tmp)
            return Tmp - MaxShAmt;
        }
      }
      // shl destroys sign bits, ensure it doesn't shift out all sign bits.
      unsigned Tmp = computeNumSignBits(Src1, DemandedElts, Depth + 1);
      if (MaxShAmt < Tmp)
        return Tmp - MaxShAmt;
    }
    break;
  }
  case TargetOpcode::G_TRUNC: {
    Register Src = MI.getOperand(1).getReg();
    LLT SrcTy = MRI.getType(Src);

    // Check if the sign bits of source go down as far as the truncated value.
    unsigned DstTyBits = DstTy.getScalarSizeInBits();
    unsigned NumSrcBits = SrcTy.getScalarSizeInBits();
    unsigned NumSrcSignBits = computeNumSignBits(Src, DemandedElts, Depth + 1);
    if (NumSrcSignBits > (NumSrcBits - DstTyBits))
      return NumSrcSignBits - (NumSrcBits - DstTyBits);
    break;
  }
  case TargetOpcode::G_SELECT: {
    return computeNumSignBitsMin(MI.getOperand(2).getReg(),
                                 MI.getOperand(3).getReg(), DemandedElts,
                                 Depth + 1);
  }
  case TargetOpcode::G_SMIN:
  case TargetOpcode::G_SMAX:
  case TargetOpcode::G_UMIN:
  case TargetOpcode::G_UMAX:
    // TODO: Handle clamp pattern with number of sign bits for SMIN/SMAX.
    return computeNumSignBitsMin(MI.getOperand(1).getReg(),
                                 MI.getOperand(2).getReg(), DemandedElts,
                                 Depth + 1);
  case TargetOpcode::G_SADDO:
  case TargetOpcode::G_SADDE:
  case TargetOpcode::G_UADDO:
  case TargetOpcode::G_UADDE:
  case TargetOpcode::G_SSUBO:
  case TargetOpcode::G_SSUBE:
  case TargetOpcode::G_USUBO:
  case TargetOpcode::G_USUBE:
  case TargetOpcode::G_SMULO:
  case TargetOpcode::G_UMULO: {
    // If compares returns 0/-1, all bits are sign bits.
    // We know that we have an integer-based boolean since these operations
    // are only available for integer.
    if (MI.getOperand(1).getReg() == R) {
      if (TL.getBooleanContents(DstTy.isVector(), false) ==
          TargetLowering::ZeroOrNegativeOneBooleanContent)
        return TyBits;
    }

    break;
  }
  case TargetOpcode::G_SUB: {
    Register Src2 = MI.getOperand(2).getReg();
    unsigned Src2NumSignBits =
        computeNumSignBits(Src2, DemandedElts, Depth + 1);
    if (Src2NumSignBits == 1)
      return 1; // Early out.

    // Handle NEG.
    Register Src1 = MI.getOperand(1).getReg();
    KnownBits Known1 = getKnownBits(Src1, DemandedElts, Depth);
    if (Known1.isZero()) {
      KnownBits Known2 = getKnownBits(Src2, DemandedElts, Depth);
      // If the input is known to be 0 or 1, the output is 0/-1, which is all
      // sign bits set.
      if ((Known2.Zero | 1).isAllOnes())
        return TyBits;

      // If the input is known to be positive (the sign bit is known clear),
      // the output of the NEG has, at worst, the same number of sign bits as
      // the input.
      if (Known2.isNonNegative()) {
        FirstAnswer = Src2NumSignBits;
        break;
      }

      // Otherwise, we treat this like a SUB.
    }

    unsigned Src1NumSignBits =
        computeNumSignBits(Src1, DemandedElts, Depth + 1);
    if (Src1NumSignBits == 1)
      return 1; // Early Out.

    // Sub can have at most one carry bit.  Thus we know that the output
    // is, at worst, one more bit than the inputs.
    FirstAnswer = std::min(Src1NumSignBits, Src2NumSignBits) - 1;
    break;
  }
  case TargetOpcode::G_ADD: {
    Register Src2 = MI.getOperand(2).getReg();
    unsigned Src2NumSignBits =
        computeNumSignBits(Src2, DemandedElts, Depth + 1);
    if (Src2NumSignBits <= 2)
      return 1; // Early out.

    Register Src1 = MI.getOperand(1).getReg();
    unsigned Src1NumSignBits =
        computeNumSignBits(Src1, DemandedElts, Depth + 1);
    if (Src1NumSignBits == 1)
      return 1; // Early Out.

    // Special case decrementing a value (ADD X, -1):
    KnownBits Known2 = getKnownBits(Src2, DemandedElts, Depth);
    if (Known2.isAllOnes()) {
      KnownBits Known1 = getKnownBits(Src1, DemandedElts, Depth);
      // If the input is known to be 0 or 1, the output is 0/-1, which is all
      // sign bits set.
      if ((Known1.Zero | 1).isAllOnes())
        return TyBits;

      // If we are subtracting one from a positive number, there is no carry
      // out of the result.
      if (Known1.isNonNegative()) {
        FirstAnswer = Src1NumSignBits;
        break;
      }

      // Otherwise, we treat this like an ADD.
    }

    // Add can have at most one carry bit.  Thus we know that the output
    // is, at worst, one more bit than the inputs.
    FirstAnswer = std::min(Src1NumSignBits, Src2NumSignBits) - 1;
    break;
  }
  case TargetOpcode::G_FCMP:
  case TargetOpcode::G_ICMP: {
    bool IsFP = Opcode == TargetOpcode::G_FCMP;
    if (TyBits == 1)
      break;
    auto BC = TL.getBooleanContents(DstTy.isVector(), IsFP);
    if (BC == TargetLoweringBase::ZeroOrNegativeOneBooleanContent)
      return TyBits; // All bits are sign bits.
    if (BC == TargetLowering::ZeroOrOneBooleanContent)
      return TyBits - 1; // Every always-zero bit is a sign bit.
    break;
  }
  case TargetOpcode::G_BUILD_VECTOR: {
    // Collect the known bits that are shared by every demanded vector element.
    FirstAnswer = TyBits;
    APInt SingleDemandedElt(1, 1);
    for (const auto &[I, MO] : enumerate(drop_begin(MI.operands()))) {
      if (!DemandedElts[I])
        continue;

      unsigned Tmp2 =
          computeNumSignBits(MO.getReg(), SingleDemandedElt, Depth + 1);
      FirstAnswer = std::min(FirstAnswer, Tmp2);

      // If we don't know any bits, early out.
      if (FirstAnswer == 1)
        break;
    }
    break;
  }
  case TargetOpcode::G_CONCAT_VECTORS: {
    if (MRI.getType(MI.getOperand(0).getReg()).isScalableVector())
      break;
    FirstAnswer = TyBits;
    // Determine the minimum number of sign bits across all demanded
    // elts of the input vectors. Early out if the result is already 1.
    unsigned NumSubVectorElts =
        MRI.getType(MI.getOperand(1).getReg()).getNumElements();
    for (const auto &[I, MO] : enumerate(drop_begin(MI.operands()))) {
      APInt DemandedSub =
          DemandedElts.extractBits(NumSubVectorElts, I * NumSubVectorElts);
      if (!DemandedSub)
        continue;
      unsigned Tmp2 = computeNumSignBits(MO.getReg(), DemandedSub, Depth + 1);

      FirstAnswer = std::min(FirstAnswer, Tmp2);

      // If we don't know any bits, early out.
      if (FirstAnswer == 1)
        break;
    }
    break;
  }
  case TargetOpcode::G_SHUFFLE_VECTOR: {
    // Collect the minimum number of sign bits that are shared by every vector
    // element referenced by the shuffle.
    APInt DemandedLHS, DemandedRHS;
    Register Src1 = MI.getOperand(1).getReg();
    unsigned NumElts = MRI.getType(Src1).getNumElements();
    if (!getShuffleDemandedElts(NumElts, MI.getOperand(3).getShuffleMask(),
                                DemandedElts, DemandedLHS, DemandedRHS))
      return 1;

    if (!!DemandedLHS)
      FirstAnswer = computeNumSignBits(Src1, DemandedLHS, Depth + 1);
    // If we don't know anything, early out and try computeKnownBits fall-back.
    if (FirstAnswer == 1)
      break;
    if (!!DemandedRHS) {
      unsigned Tmp2 =
          computeNumSignBits(MI.getOperand(2).getReg(), DemandedRHS, Depth + 1);
      FirstAnswer = std::min(FirstAnswer, Tmp2);
    }
    break;
  }
  case TargetOpcode::G_SPLAT_VECTOR: {
    // Check if the sign bits of source go down as far as the truncated value.
    Register Src = MI.getOperand(1).getReg();
    unsigned NumSrcSignBits = computeNumSignBits(Src, APInt(1, 1), Depth + 1);
    unsigned NumSrcBits = MRI.getType(Src).getSizeInBits();
    if (NumSrcSignBits > (NumSrcBits - TyBits))
      return NumSrcSignBits - (NumSrcBits - TyBits);
    break;
  }
  case TargetOpcode::G_INTRINSIC:
  case TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS:
  case TargetOpcode::G_INTRINSIC_CONVERGENT:
  case TargetOpcode::G_INTRINSIC_CONVERGENT_W_SIDE_EFFECTS:
  default: {
    unsigned NumBits =
        TL.computeNumSignBitsForTargetInstr(*this, R, DemandedElts, MRI, Depth);
    if (NumBits > 1)
      FirstAnswer = std::max(FirstAnswer, NumBits);
    break;
  }
  }

  // Finally, if we can prove that the top bits of the result are 0's or 1's,
  // use this information.
  KnownBits Known = getKnownBits(R, DemandedElts, Depth);
  APInt Mask;
  if (Known.isNonNegative()) { // sign bit is 0
    Mask = Known.Zero;
  } else if (Known.isNegative()) { // sign bit is 1;
    Mask = Known.One;
  } else {
    // Nothing known.
    return FirstAnswer;
  }

  // Okay, we know that the sign bit in Mask is set.  Use CLO to determine
  // the number of identical bits in the top of the input value.
  Mask <<= Mask.getBitWidth() - TyBits;
  return std::max(FirstAnswer, Mask.countl_one());
}

unsigned GISelValueTracking::computeNumSignBits(Register R, unsigned Depth) {
  LLT Ty = MRI.getType(R);
  APInt DemandedElts =
      Ty.isFixedVector() ? APInt::getAllOnes(Ty.getNumElements()) : APInt(1, 1);
  return computeNumSignBits(R, DemandedElts, Depth);
}

std::optional<ConstantRange> GISelValueTracking::getValidShiftAmountRange(
    Register R, const APInt &DemandedElts, unsigned Depth) {
  // Shifting more than the bitwidth is not valid.
  MachineInstr &MI = *MRI.getVRegDef(R);
  unsigned Opcode = MI.getOpcode();

  LLT Ty = MRI.getType(R);
  unsigned BitWidth = Ty.getScalarSizeInBits();

  if (Opcode == TargetOpcode::G_CONSTANT) {
    const APInt &ShAmt = MI.getOperand(1).getCImm()->getValue();
    if (ShAmt.uge(BitWidth))
      return std::nullopt;
    return ConstantRange(ShAmt);
  }

  if (Opcode == TargetOpcode::G_BUILD_VECTOR) {
    const APInt *MinAmt = nullptr, *MaxAmt = nullptr;
    for (unsigned I = 0, E = MI.getNumOperands() - 1; I != E; ++I) {
      if (!DemandedElts[I])
        continue;
      MachineInstr *Op = MRI.getVRegDef(MI.getOperand(I + 1).getReg());
      if (Op->getOpcode() != TargetOpcode::G_CONSTANT) {
        MinAmt = MaxAmt = nullptr;
        break;
      }

      const APInt &ShAmt = Op->getOperand(1).getCImm()->getValue();
      if (ShAmt.uge(BitWidth))
        return std::nullopt;
      if (!MinAmt || MinAmt->ugt(ShAmt))
        MinAmt = &ShAmt;
      if (!MaxAmt || MaxAmt->ult(ShAmt))
        MaxAmt = &ShAmt;
    }
    assert(((!MinAmt && !MaxAmt) || (MinAmt && MaxAmt)) &&
           "Failed to find matching min/max shift amounts");
    if (MinAmt && MaxAmt)
      return ConstantRange(*MinAmt, *MaxAmt + 1);
  }

  // Use computeKnownBits to find a hidden constant/knownbits (usually type
  // legalized). e.g. Hidden behind multiple bitcasts/build_vector/casts etc.
  KnownBits KnownAmt = getKnownBits(R, DemandedElts, Depth);
  if (KnownAmt.getMaxValue().ult(BitWidth))
    return ConstantRange::fromKnownBits(KnownAmt, /*IsSigned=*/false);

  return std::nullopt;
}

std::optional<uint64_t> GISelValueTracking::getValidMinimumShiftAmount(
    Register R, const APInt &DemandedElts, unsigned Depth) {
  if (std::optional<ConstantRange> AmtRange =
          getValidShiftAmountRange(R, DemandedElts, Depth))
    return AmtRange->getUnsignedMin().getZExtValue();
  return std::nullopt;
}

void GISelValueTrackingAnalysisLegacy::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool GISelValueTrackingAnalysisLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  return false;
}

GISelValueTracking &GISelValueTrackingAnalysisLegacy::get(MachineFunction &MF) {
  if (!Info) {
    unsigned MaxDepth =
        MF.getTarget().getOptLevel() == CodeGenOptLevel::None ? 2 : 6;
    Info = std::make_unique<GISelValueTracking>(MF, MaxDepth);
  }
  return *Info;
}

AnalysisKey GISelValueTrackingAnalysis::Key;

GISelValueTracking
GISelValueTrackingAnalysis::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  return Result(MF);
}

PreservedAnalyses
GISelValueTrackingPrinterPass::run(MachineFunction &MF,
                                   MachineFunctionAnalysisManager &MFAM) {
  auto &VTA = MFAM.getResult<GISelValueTrackingAnalysis>(MF);
  const auto &MRI = MF.getRegInfo();
  OS << "name: ";
  MF.getFunction().printAsOperand(OS, /*PrintType=*/false);
  OS << '\n';

  for (MachineBasicBlock &BB : MF) {
    for (MachineInstr &MI : BB) {
      for (MachineOperand &MO : MI.defs()) {
        if (!MO.isReg() || MO.getReg().isPhysical())
          continue;
        Register Reg = MO.getReg();
        if (!MRI.getType(Reg).isValid())
          continue;
        KnownBits Known = VTA.getKnownBits(Reg);
        unsigned SignedBits = VTA.computeNumSignBits(Reg);
        OS << "  " << MO << " KnownBits:" << Known << " SignBits:" << SignedBits
           << '\n';
      };
    }
  }
  return PreservedAnalyses::all();
}
