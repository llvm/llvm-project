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
  // For now, we only maintain the cache during one request.
  assert(ComputeKnownBitsCache.empty() && "Cache should have been cleared");

  KnownBits Known;
  computeKnownBitsImpl(R, Known, DemandedElts, Depth);
  ComputeKnownBitsCache.clear();
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

LLVM_ATTRIBUTE_UNUSED static void
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
  auto CacheEntry = ComputeKnownBitsCache.find(R);
  if (CacheEntry != ComputeKnownBitsCache.end()) {
    Known = CacheEntry->second;
    LLVM_DEBUG(dbgs() << "Cache hit at ");
    LLVM_DEBUG(dumpResult(MI, Known, Depth));
    assert(Known.getBitWidth() == BitWidth && "Cache entry size doesn't match");
    return;
  }
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
    // Record in the cache that we know nothing for MI.
    // This will get updated later and in the meantime, if we reach that
    // phi again, because of a loop, we will cut the search thanks to this
    // cache entry.
    // We could actually build up more information on the phi by not cutting
    // the search, but that additional information is more a side effect
    // than an intended choice.
    // Therefore, for now, save on compile time until we derive a proper way
    // to derive known bits for PHIs within loops.
    ComputeKnownBitsCache[R] = KnownBits(BitWidth);
    // PHI's operand are a mix of registers and basic blocks interleaved.
    // We only care about the register ones.
    for (unsigned Idx = 1; Idx < MI.getNumOperands(); Idx += 2) {
      const MachineOperand &Src = MI.getOperand(Idx);
      Register SrcReg = Src.getReg();
      // Look through trivial copies and phis but don't look through trivial
      // copies or phis of the form `%1:(s32) = OP %0:gpr32`, known-bits
      // analysis is currently unable to determine the bit width of a
      // register class.
      //
      // We can't use NoSubRegister by name as it's defined by each target but
      // it's always defined to be 0 by tablegen.
      if (SrcReg.isVirtual() && Src.getSubReg() == 0 /*NoSubRegister*/ &&
          MRI.getType(SrcReg).isValid()) {
        // For COPYs we don't do anything, don't increase the depth.
        computeKnownBitsImpl(SrcReg, Known2, DemandedElts,
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
    Known = KnownBits::sub(Known, Known2);
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
  case TargetOpcode::G_SADDE:
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
  }

  LLVM_DEBUG(dumpResult(MI, Known, Depth));

  // Update the cache.
  ComputeKnownBitsCache[R] = Known;
}

static bool outputDenormalIsIEEEOrPosZero(const MachineFunction &MF, LLT Ty) {
  Ty = Ty.getScalarType();
  DenormalMode Mode = MF.getDenormalMode(getFltSemanticForLLT(Ty));
  return Mode.Output == DenormalMode::IEEE ||
         Mode.Output == DenormalMode::PositiveZero;
}

void GISelValueTracking::computeKnownFPClass(Register R, KnownFPClass &Known,
                                             FPClassTest InterestedClasses,
                                             unsigned Depth) {
  LLT Ty = MRI.getType(R);
  APInt DemandedElts =
      Ty.isFixedVector() ? APInt::getAllOnes(Ty.getNumElements()) : APInt(1, 1);
  computeKnownFPClass(R, DemandedElts, InterestedClasses, Known, Depth);
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

  // Sign should be preserved
  // TODO: Handle cannot be ordered greater than zero
  if (KnownSrc.cannotBeOrderedLessThanZero())
    Known.knownNot(KnownFPClass::OrderedLessThanZeroMask);

  Known.propagateNaN(KnownSrc, true);

  // Infinity needs a range check.
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

  auto ClearClassesFromFlags =
      make_scope_exit([=, &Known] { Known.knownNot(KnownNotFromFlags); });

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

    if (A != B)
      break;

    // The multiply cannot be -0 and therefore the add can't be -0
    Known.knownNot(fcNegZero);

    // x * x + y is non-negative if y is non-negative.
    KnownFPClass KnownAddend;
    computeKnownFPClass(C, DemandedElts, InterestedClasses, KnownAddend,
                        Depth + 1);

    if (KnownAddend.cannotBeOrderedLessThanZero())
      Known.knownNot(fcNegative);
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

    if (KnownSrc.isKnownNeverPosInfinity())
      Known.knownNot(fcPosInf);
    if (KnownSrc.isKnownNever(fcSNan))
      Known.knownNot(fcSNan);

    // Any negative value besides -0 returns a nan.
    if (KnownSrc.isKnownNeverNaN() && KnownSrc.cannotBeOrderedLessThanZero())
      Known.knownNot(fcNan);

    // The only negative value that can be returned is -0 for -0 inputs.
    Known.knownNot(fcNegInf | fcNegSubnormal | fcNegNormal);
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
  case TargetOpcode::G_FSIN:
  case TargetOpcode::G_FCOS:
  case TargetOpcode::G_FSINCOS: {
    // Return NaN on infinite inputs.
    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;

    computeKnownFPClass(Val, DemandedElts, InterestedClasses, KnownSrc,
                        Depth + 1);
    Known.knownNot(fcInf);

    if (KnownSrc.isKnownNeverNaN() && KnownSrc.isKnownNeverInfinity())
      Known.knownNot(fcNan);
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

    bool NeverNaN = KnownLHS.isKnownNeverNaN() || KnownRHS.isKnownNeverNaN();
    Known = KnownLHS | KnownRHS;

    // If either operand is not NaN, the result is not NaN.
    if (NeverNaN && (Opcode == TargetOpcode::G_FMINNUM ||
                     Opcode == TargetOpcode::G_FMAXNUM ||
                     Opcode == TargetOpcode::G_FMINIMUMNUM ||
                     Opcode == TargetOpcode::G_FMAXIMUMNUM))
      Known.knownNot(fcNan);

    if (Opcode == TargetOpcode::G_FMAXNUM ||
        Opcode == TargetOpcode::G_FMAXIMUMNUM ||
        Opcode == TargetOpcode::G_FMAXNUM_IEEE) {
      // If at least one operand is known to be positive, the result must be
      // positive.
      if ((KnownLHS.cannotBeOrderedLessThanZero() &&
           KnownLHS.isKnownNeverNaN()) ||
          (KnownRHS.cannotBeOrderedLessThanZero() &&
           KnownRHS.isKnownNeverNaN()))
        Known.knownNot(KnownFPClass::OrderedLessThanZeroMask);
    } else if (Opcode == TargetOpcode::G_FMAXIMUM) {
      // If at least one operand is known to be positive, the result must be
      // positive.
      if (KnownLHS.cannotBeOrderedLessThanZero() ||
          KnownRHS.cannotBeOrderedLessThanZero())
        Known.knownNot(KnownFPClass::OrderedLessThanZeroMask);
    } else if (Opcode == TargetOpcode::G_FMINNUM ||
               Opcode == TargetOpcode::G_FMINIMUMNUM ||
               Opcode == TargetOpcode::G_FMINNUM_IEEE) {
      // If at least one operand is known to be negative, the result must be
      // negative.
      if ((KnownLHS.cannotBeOrderedGreaterThanZero() &&
           KnownLHS.isKnownNeverNaN()) ||
          (KnownRHS.cannotBeOrderedGreaterThanZero() &&
           KnownRHS.isKnownNeverNaN()))
        Known.knownNot(KnownFPClass::OrderedGreaterThanZeroMask);
    } else if (Opcode == TargetOpcode::G_FMINIMUM) {
      // If at least one operand is known to be negative, the result must be
      // negative.
      if (KnownLHS.cannotBeOrderedGreaterThanZero() ||
          KnownRHS.cannotBeOrderedGreaterThanZero())
        Known.knownNot(KnownFPClass::OrderedGreaterThanZeroMask);
    } else {
      llvm_unreachable("unhandled intrinsic");
    }

    // Fixup zero handling if denormals could be returned as a zero.
    //
    // As there's no spec for denormal flushing, be conservative with the
    // treatment of denormals that could be flushed to zero. For older
    // subtargets on AMDGPU the min/max instructions would not flush the
    // output and return the original value.
    //
    if ((Known.KnownFPClasses & fcZero) != fcNone &&
        !Known.isKnownNeverSubnormal()) {
      DenormalMode Mode =
          MF->getDenormalMode(getFltSemanticForLLT(DstTy.getScalarType()));
      if (Mode != DenormalMode::getIEEE())
        Known.KnownFPClasses |= fcZero;
    }

    if (Known.isKnownNeverNaN()) {
      if (KnownLHS.SignBit && KnownRHS.SignBit &&
          *KnownLHS.SignBit == *KnownRHS.SignBit) {
        if (*KnownLHS.SignBit)
          Known.signBitMustBeOne();
        else
          Known.signBitMustBeZero();
      } else if ((Opcode == TargetOpcode::G_FMAXIMUM ||
                  Opcode == TargetOpcode::G_FMINIMUM) ||
                 Opcode == TargetOpcode::G_FMAXIMUMNUM ||
                 Opcode == TargetOpcode::G_FMINIMUMNUM ||
                 Opcode == TargetOpcode::G_FMAXNUM_IEEE ||
                 Opcode == TargetOpcode::G_FMINNUM_IEEE ||
                 // FIXME: Should be using logical zero versions
                 ((KnownLHS.isKnownNeverNegZero() ||
                   KnownRHS.isKnownNeverPosZero()) &&
                  (KnownLHS.isKnownNeverPosZero() ||
                   KnownRHS.isKnownNeverNegZero()))) {
        if ((Opcode == TargetOpcode::G_FMAXIMUM ||
             Opcode == TargetOpcode::G_FMAXNUM ||
             Opcode == TargetOpcode::G_FMAXIMUMNUM ||
             Opcode == TargetOpcode::G_FMAXNUM_IEEE) &&
            (KnownLHS.SignBit == false || KnownRHS.SignBit == false))
          Known.signBitMustBeZero();
        else if ((Opcode == TargetOpcode::G_FMINIMUM ||
                  Opcode == TargetOpcode::G_FMINNUM ||
                  Opcode == TargetOpcode::G_FMINIMUMNUM ||
                  Opcode == TargetOpcode::G_FMINNUM_IEEE) &&
                 (KnownLHS.SignBit == true || KnownRHS.SignBit == true))
          Known.signBitMustBeOne();
      }
    }
    break;
  }
  case TargetOpcode::G_FCANONICALIZE: {
    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Val, DemandedElts, InterestedClasses, KnownSrc,
                        Depth + 1);

    // This is essentially a stronger form of
    // propagateCanonicalizingSrc. Other "canonicalizing" operations don't
    // actually have an IR canonicalization guarantee.

    // Canonicalize may flush denormals to zero, so we have to consider the
    // denormal mode to preserve known-not-0 knowledge.
    Known.KnownFPClasses = KnownSrc.KnownFPClasses | fcZero | fcQNan;

    // Stronger version of propagateNaN
    // Canonicalize is guaranteed to quiet signaling nans.
    if (KnownSrc.isKnownNeverNaN())
      Known.knownNot(fcNan);
    else
      Known.knownNot(fcSNan);

    // If the parent function flushes denormals, the canonical output cannot
    // be a denormal.
    LLT Ty = MRI.getType(Val).getScalarType();
    const fltSemantics &FPType = getFltSemanticForLLT(Ty);
    DenormalMode DenormMode = MF->getDenormalMode(FPType);
    if (DenormMode == DenormalMode::getIEEE()) {
      if (KnownSrc.isKnownNever(fcPosZero))
        Known.knownNot(fcPosZero);
      if (KnownSrc.isKnownNever(fcNegZero))
        Known.knownNot(fcNegZero);
      break;
    }

    if (DenormMode.inputsAreZero() || DenormMode.outputsAreZero())
      Known.knownNot(fcSubnormal);

    if (DenormMode.Input == DenormalMode::PositiveZero ||
        (DenormMode.Output == DenormalMode::PositiveZero &&
         DenormMode.Input == DenormalMode::IEEE))
      Known.knownNot(fcNegZero);

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
  case TargetOpcode::G_TRUNC:
  case TargetOpcode::G_FFLOOR:
  case TargetOpcode::G_FCEIL:
  case TargetOpcode::G_FRINT:
  case TargetOpcode::G_FNEARBYINT:
  case TargetOpcode::G_INTRINSIC_FPTRUNC_ROUND:
  case TargetOpcode::G_INTRINSIC_ROUND: {
    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    FPClassTest InterestedSrcs = InterestedClasses;
    if (InterestedSrcs & fcPosFinite)
      InterestedSrcs |= fcPosFinite;
    if (InterestedSrcs & fcNegFinite)
      InterestedSrcs |= fcNegFinite;
    computeKnownFPClass(Val, DemandedElts, InterestedSrcs, KnownSrc, Depth + 1);

    // Integer results cannot be subnormal.
    Known.knownNot(fcSubnormal);

    Known.propagateNaN(KnownSrc, true);

    // TODO: handle multi unit FPTypes once LLT FPInfo lands

    // Negative round ups to 0 produce -0
    if (KnownSrc.isKnownNever(fcPosFinite))
      Known.knownNot(fcPosFinite);
    if (KnownSrc.isKnownNever(fcNegFinite))
      Known.knownNot(fcNegFinite);

    break;
  }
  case TargetOpcode::G_FEXP:
  case TargetOpcode::G_FEXP2:
  case TargetOpcode::G_FEXP10: {
    Known.knownNot(fcNegative);
    if ((InterestedClasses & fcNan) == fcNone)
      break;

    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Val, DemandedElts, InterestedClasses, KnownSrc,
                        Depth + 1);
    if (KnownSrc.isKnownNeverNaN()) {
      Known.knownNot(fcNan);
      Known.signBitMustBeZero();
    }

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
      InterestedSrcs |= fcNan | (fcNegative & ~fcNan);

    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Val, DemandedElts, InterestedSrcs, KnownSrc, Depth + 1);

    if (KnownSrc.isKnownNeverPosInfinity())
      Known.knownNot(fcPosInf);

    if (KnownSrc.isKnownNeverNaN() && KnownSrc.cannotBeOrderedLessThanZero())
      Known.knownNot(fcNan);

    LLT Ty = MRI.getType(Val).getScalarType();
    const fltSemantics &FltSem = getFltSemanticForLLT(Ty);
    DenormalMode Mode = MF->getDenormalMode(FltSem);

    if (KnownSrc.isKnownNeverLogicalZero(Mode))
      Known.knownNot(fcNegInf);

    break;
  }
  case TargetOpcode::G_FPOWI: {
    if ((InterestedClasses & fcNegative) == fcNone)
      break;

    Register Exp = MI.getOperand(2).getReg();
    LLT ExpTy = MRI.getType(Exp);
    KnownBits ExponentKnownBits = getKnownBits(
        Exp, ExpTy.isVector() ? DemandedElts : APInt(1, 1), Depth + 1);

    if (ExponentKnownBits.Zero[0]) { // Is even
      Known.knownNot(fcNegative);
      break;
    }

    // Given that exp is an integer, here are the
    // ways that pow can return a negative value:
    //
    //   pow(-x, exp)   --> negative if exp is odd and x is negative.
    //   pow(-0, exp)   --> -inf if exp is negative odd.
    //   pow(-0, exp)   --> -0 if exp is positive odd.
    //   pow(-inf, exp) --> -0 if exp is negative odd.
    //   pow(-inf, exp) --> -inf if exp is positive odd.
    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Val, DemandedElts, fcNegative, KnownSrc, Depth + 1);
    if (KnownSrc.isKnownNever(fcNegative))
      Known.knownNot(fcNegative);
    break;
  }
  case TargetOpcode::G_FLDEXP:
  case TargetOpcode::G_STRICT_FLDEXP: {
    Register Val = MI.getOperand(1).getReg();
    KnownFPClass KnownSrc;
    computeKnownFPClass(Val, DemandedElts, InterestedClasses, KnownSrc,
                        Depth + 1);
    Known.propagateNaN(KnownSrc, /*PropagateSign=*/true);

    // Sign is preserved, but underflows may produce zeroes.
    if (KnownSrc.isKnownNever(fcNegative))
      Known.knownNot(fcNegative);
    else if (KnownSrc.cannotBeOrderedLessThanZero())
      Known.knownNot(KnownFPClass::OrderedLessThanZeroMask);

    if (KnownSrc.isKnownNever(fcPositive))
      Known.knownNot(fcPositive);
    else if (KnownSrc.cannotBeOrderedGreaterThanZero())
      Known.knownNot(KnownFPClass::OrderedGreaterThanZeroMask);

    // Can refine inf/zero handling based on the exponent operand.
    const FPClassTest ExpInfoMask = fcZero | fcSubnormal | fcInf;
    if ((InterestedClasses & ExpInfoMask) == fcNone)
      break;
    if ((KnownSrc.KnownFPClasses & ExpInfoMask) == fcNone)
      break;

    // TODO: Handle constant range of Exp

    break;
  }
  case TargetOpcode::G_INTRINSIC_ROUNDEVEN: {
    computeKnownFPClassForFPTrunc(MI, DemandedElts, InterestedClasses, Known,
                                  Depth);
    break;
  }
  case TargetOpcode::G_FADD:
  case TargetOpcode::G_STRICT_FADD:
  case TargetOpcode::G_FSUB:
  case TargetOpcode::G_STRICT_FSUB: {
    Register LHS = MI.getOperand(1).getReg();
    Register RHS = MI.getOperand(2).getReg();
    KnownFPClass KnownLHS, KnownRHS;
    bool WantNegative =
        (Opcode == TargetOpcode::G_FADD ||
         Opcode == TargetOpcode::G_STRICT_FADD) &&
        (InterestedClasses & KnownFPClass::OrderedLessThanZeroMask) != fcNone;
    bool WantNaN = (InterestedClasses & fcNan) != fcNone;
    bool WantNegZero = (InterestedClasses & fcNegZero) != fcNone;

    if (!WantNaN && !WantNegative && !WantNegZero)
      break;

    FPClassTest InterestedSrcs = InterestedClasses;
    if (WantNegative)
      InterestedSrcs |= KnownFPClass::OrderedLessThanZeroMask;
    if (InterestedClasses & fcNan)
      InterestedSrcs |= fcInf;
    computeKnownFPClass(RHS, DemandedElts, InterestedSrcs, KnownRHS, Depth + 1);

    if ((WantNaN && KnownRHS.isKnownNeverNaN()) ||
        (WantNegative && KnownRHS.cannotBeOrderedLessThanZero()) ||
        WantNegZero ||
        (Opcode == TargetOpcode::G_FSUB ||
         Opcode == TargetOpcode::G_STRICT_FSUB)) {

      // RHS is canonically cheaper to compute. Skip inspecting the LHS if
      // there's no point.
      computeKnownFPClass(LHS, DemandedElts, InterestedSrcs, KnownLHS,
                          Depth + 1);
      // Adding positive and negative infinity produces NaN.
      // TODO: Check sign of infinities.
      if (KnownLHS.isKnownNeverNaN() && KnownRHS.isKnownNeverNaN() &&
          (KnownLHS.isKnownNeverInfinity() || KnownRHS.isKnownNeverInfinity()))
        Known.knownNot(fcNan);

      if (Opcode == Instruction::FAdd) {
        if (KnownLHS.cannotBeOrderedLessThanZero() &&
            KnownRHS.cannotBeOrderedLessThanZero())
          Known.knownNot(KnownFPClass::OrderedLessThanZeroMask);

        // (fadd x, 0.0) is guaranteed to return +0.0, not -0.0.
        if ((KnownLHS.isKnownNeverLogicalNegZero(MF->getDenormalMode(
                 getFltSemanticForLLT(DstTy.getScalarType()))) ||
             KnownRHS.isKnownNeverLogicalNegZero(MF->getDenormalMode(
                 getFltSemanticForLLT(DstTy.getScalarType())))) &&
            // Make sure output negative denormal can't flush to -0
            outputDenormalIsIEEEOrPosZero(*MF, DstTy))
          Known.knownNot(fcNegZero);
      } else {
        // Only fsub -0, +0 can return -0
        if ((KnownLHS.isKnownNeverLogicalNegZero(MF->getDenormalMode(
                 getFltSemanticForLLT(DstTy.getScalarType()))) ||
             KnownRHS.isKnownNeverLogicalPosZero(MF->getDenormalMode(
                 getFltSemanticForLLT(DstTy.getScalarType())))) &&
            // Make sure output negative denormal can't flush to -0
            outputDenormalIsIEEEOrPosZero(*MF, DstTy))
          Known.knownNot(fcNegZero);
      }
    }

    break;
  }
  case TargetOpcode::G_FMUL:
  case TargetOpcode::G_STRICT_FMUL: {
    Register LHS = MI.getOperand(1).getReg();
    Register RHS = MI.getOperand(2).getReg();
    // X * X is always non-negative or a NaN.
    if (LHS == RHS)
      Known.knownNot(fcNegative);

    if ((InterestedClasses & fcNan) != fcNan)
      break;

    // fcSubnormal is only needed in case of DAZ.
    const FPClassTest NeedForNan = fcNan | fcInf | fcZero | fcSubnormal;

    KnownFPClass KnownLHS, KnownRHS;
    computeKnownFPClass(RHS, DemandedElts, NeedForNan, KnownRHS, Depth + 1);
    if (!KnownRHS.isKnownNeverNaN())
      break;

    computeKnownFPClass(LHS, DemandedElts, NeedForNan, KnownLHS, Depth + 1);
    if (!KnownLHS.isKnownNeverNaN())
      break;

    if (KnownLHS.SignBit && KnownRHS.SignBit) {
      if (*KnownLHS.SignBit == *KnownRHS.SignBit)
        Known.signBitMustBeZero();
      else
        Known.signBitMustBeOne();
    }

    // If 0 * +/-inf produces NaN.
    if (KnownLHS.isKnownNeverInfinity() && KnownRHS.isKnownNeverInfinity()) {
      Known.knownNot(fcNan);
      break;
    }

    if ((KnownRHS.isKnownNeverInfinity() ||
         KnownLHS.isKnownNeverLogicalZero(MF->getDenormalMode(
             getFltSemanticForLLT(DstTy.getScalarType())))) &&
        (KnownLHS.isKnownNeverInfinity() ||
         KnownRHS.isKnownNeverLogicalZero(
             MF->getDenormalMode(getFltSemanticForLLT(DstTy.getScalarType())))))
      Known.knownNot(fcNan);

    break;
  }
  case TargetOpcode::G_FDIV:
  case TargetOpcode::G_FREM: {
    Register LHS = MI.getOperand(1).getReg();
    Register RHS = MI.getOperand(2).getReg();

    if (LHS == RHS) {
      // TODO: Could filter out snan if we inspect the operand
      if (Opcode == TargetOpcode::G_FDIV) {
        // X / X is always exactly 1.0 or a NaN.
        Known.KnownFPClasses = fcNan | fcPosNormal;
      } else {
        // X % X is always exactly [+-]0.0 or a NaN.
        Known.KnownFPClasses = fcNan | fcZero;
      }

      break;
    }

    const bool WantNan = (InterestedClasses & fcNan) != fcNone;
    const bool WantNegative = (InterestedClasses & fcNegative) != fcNone;
    const bool WantPositive = Opcode == TargetOpcode::G_FREM &&
                              (InterestedClasses & fcPositive) != fcNone;
    if (!WantNan && !WantNegative && !WantPositive)
      break;

    KnownFPClass KnownLHS, KnownRHS;

    computeKnownFPClass(RHS, DemandedElts, fcNan | fcInf | fcZero | fcNegative,
                        KnownRHS, Depth + 1);

    bool KnowSomethingUseful =
        KnownRHS.isKnownNeverNaN() || KnownRHS.isKnownNever(fcNegative);

    if (KnowSomethingUseful || WantPositive) {
      const FPClassTest InterestedLHS =
          WantPositive ? fcAllFlags
                       : fcNan | fcInf | fcZero | fcSubnormal | fcNegative;

      computeKnownFPClass(LHS, DemandedElts, InterestedClasses & InterestedLHS,
                          KnownLHS, Depth + 1);
    }

    if (Opcode == Instruction::FDiv) {
      // Only 0/0, Inf/Inf produce NaN.
      if (KnownLHS.isKnownNeverNaN() && KnownRHS.isKnownNeverNaN() &&
          (KnownLHS.isKnownNeverInfinity() ||
           KnownRHS.isKnownNeverInfinity()) &&
          ((KnownLHS.isKnownNeverLogicalZero(MF->getDenormalMode(
               getFltSemanticForLLT(DstTy.getScalarType())))) ||
           (KnownRHS.isKnownNeverLogicalZero(MF->getDenormalMode(
               getFltSemanticForLLT(DstTy.getScalarType())))))) {
        Known.knownNot(fcNan);
      }

      // X / -0.0 is -Inf (or NaN).
      // +X / +X is +X
      if (KnownLHS.isKnownNever(fcNegative) &&
          KnownRHS.isKnownNever(fcNegative))
        Known.knownNot(fcNegative);
    } else {
      // Inf REM x and x REM 0 produce NaN.
      if (KnownLHS.isKnownNeverNaN() && KnownRHS.isKnownNeverNaN() &&
          KnownLHS.isKnownNeverInfinity() &&
          KnownRHS.isKnownNeverLogicalZero(MF->getDenormalMode(
              getFltSemanticForLLT(DstTy.getScalarType())))) {
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
  case TargetOpcode::G_FPEXT: {
    Register Dst = MI.getOperand(0).getReg();
    Register Src = MI.getOperand(1).getReg();
    // Infinity, nan and zero propagate from source.
    computeKnownFPClass(R, DemandedElts, InterestedClasses, Known, Depth + 1);

    LLT DstTy = MRI.getType(Dst).getScalarType();
    const fltSemantics &DstSem = getFltSemanticForLLT(DstTy);
    LLT SrcTy = MRI.getType(Src).getScalarType();
    const fltSemantics &SrcSem = getFltSemanticForLLT(SrcTy);

    // All subnormal inputs should be in the normal range in the result type.
    if (APFloat::isRepresentableAsNormalIn(SrcSem, DstSem)) {
      if (Known.KnownFPClasses & fcPosSubnormal)
        Known.KnownFPClasses |= fcPosNormal;
      if (Known.KnownFPClasses & fcNegSubnormal)
        Known.KnownFPClasses |= fcNegNormal;
      Known.knownNot(fcSubnormal);
    }

    // Sign bit of a nan isn't guaranteed.
    if (!Known.isKnownNeverNaN())
      Known.SignBit = std::nullopt;
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
    if (Opcode == TargetOpcode::G_UITOFP)
      Known.signBitMustBeZero();

    Register Val = MI.getOperand(1).getReg();
    LLT Ty = MRI.getType(Val);

    if (InterestedClasses & fcInf) {
      // Get width of largest magnitude integer (remove a bit if signed).
      // This still works for a signed minimum value because the largest FP
      // value is scaled by some fraction close to 2.0 (1.0 + 0.xxxx).;
      int IntSize = Ty.getScalarSizeInBits();
      if (Opcode == TargetOpcode::G_SITOFP)
        --IntSize;

      // If the exponent of the largest finite FP value can hold the largest
      // integer, the result of the cast must be finite.
      LLT FPTy = DstTy.getScalarType();
      const fltSemantics &FltSem = getFltSemanticForLLT(FPTy);
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
      if (!llvm::getShuffleDemandedElts(DstTy.getNumElements(), Shuf.getMask(),
                                        DemandedElts, DemandedLHS,
                                        DemandedRHS)) {
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
