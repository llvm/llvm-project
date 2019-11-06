//===- lib/CodeGen/GlobalISel/GISelKnownBits.cpp --------------*- C++ *-===//
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
//===------------------
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetOpcodes.h"

#define DEBUG_TYPE "gisel-known-bits"

using namespace llvm;

char llvm::GISelKnownBitsAnalysis::ID = 0;

INITIALIZE_PASS_BEGIN(GISelKnownBitsAnalysis, DEBUG_TYPE,
                      "Analysis for ComputingKnownBits", false, true)
INITIALIZE_PASS_END(GISelKnownBitsAnalysis, DEBUG_TYPE,
                    "Analysis for ComputingKnownBits", false, true)

GISelKnownBits::GISelKnownBits(MachineFunction &MF)
    : MF(MF), MRI(MF.getRegInfo()), TL(*MF.getSubtarget().getTargetLowering()),
      DL(MF.getFunction().getParent()->getDataLayout()) {}

Align GISelKnownBits::inferAlignmentForFrameIdx(int FrameIdx, int Offset,
                                                const MachineFunction &MF) {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return commonAlignment(Align(MFI.getObjectAlignment(FrameIdx)), Offset);
  // TODO: How to handle cases with Base + Offset?
}

MaybeAlign GISelKnownBits::inferPtrAlignment(const MachineInstr &MI) {
  if (MI.getOpcode() == TargetOpcode::G_FRAME_INDEX) {
    int FrameIdx = MI.getOperand(1).getIndex();
    return inferAlignmentForFrameIdx(FrameIdx, 0, *MI.getMF());
  }
  return None;
}

void GISelKnownBits::computeKnownBitsForFrameIndex(Register R, KnownBits &Known,
                                                   const APInt &DemandedElts,
                                                   unsigned Depth) {
  const MachineInstr &MI = *MRI.getVRegDef(R);
  computeKnownBitsForAlignment(Known, inferPtrAlignment(MI));
}

void GISelKnownBits::computeKnownBitsForAlignment(KnownBits &Known,
                                                  MaybeAlign Alignment) {
  if (Alignment)
    // The low bits are known zero if the pointer is aligned.
    Known.Zero.setLowBits(Log2(Alignment));
}

KnownBits GISelKnownBits::getKnownBits(MachineInstr &MI) {
  return getKnownBits(MI.getOperand(0).getReg());
}

KnownBits GISelKnownBits::getKnownBits(Register R) {
  KnownBits Known;
  LLT Ty = MRI.getType(R);
  APInt DemandedElts =
      Ty.isVector() ? APInt::getAllOnesValue(Ty.getNumElements()) : APInt(1, 1);
  computeKnownBitsImpl(R, Known, DemandedElts);
  return Known;
}

bool GISelKnownBits::signBitIsZero(Register R) {
  LLT Ty = MRI.getType(R);
  unsigned BitWidth = Ty.getScalarSizeInBits();
  return maskedValueIsZero(R, APInt::getSignMask(BitWidth));
}

APInt GISelKnownBits::getKnownZeroes(Register R) {
  return getKnownBits(R).Zero;
}

APInt GISelKnownBits::getKnownOnes(Register R) { return getKnownBits(R).One; }

void GISelKnownBits::computeKnownBitsImpl(Register R, KnownBits &Known,
                                          const APInt &DemandedElts,
                                          unsigned Depth) {
  MachineInstr &MI = *MRI.getVRegDef(R);
  unsigned Opcode = MI.getOpcode();
  LLT DstTy = MRI.getType(R);

  // Handle the case where this is called on a register that does not have a
  // type constraint (i.e. it has a register class constraint instead). This is
  // unlikely to occur except by looking through copies but it is possible for
  // the initial register being queried to be in this state.
  if (!DstTy.isValid()) {
    Known = KnownBits();
    return;
  }

  unsigned BitWidth = DstTy.getSizeInBits();
  Known = KnownBits(BitWidth); // Don't know anything

  if (DstTy.isVector())
    return; // TODO: Handle vectors.

  if (Depth == getMaxDepth())
    return;

  if (!DemandedElts)
    return; // No demanded elts, better to assume we don't know anything.

  KnownBits Known2;

  switch (Opcode) {
  default:
    TL.computeKnownBitsForTargetInstr(*this, R, Known, DemandedElts, MRI,
                                      Depth);
    break;
  case TargetOpcode::COPY: {
    MachineOperand Dst = MI.getOperand(0);
    MachineOperand Src = MI.getOperand(1);
    // Look through trivial copies but don't look through trivial copies of the
    // form `%1:(s32) = OP %0:gpr32` known-bits analysis is currently unable to
    // determine the bit width of a register class.
    //
    // We can't use NoSubRegister by name as it's defined by each target but
    // it's always defined to be 0 by tablegen.
    if (Dst.getSubReg() == 0 /*NoSubRegister*/ && Src.getReg().isVirtual() &&
        Src.getSubReg() == 0 /*NoSubRegister*/ &&
        MRI.getType(Src.getReg()).isValid()) {
      // Don't increment Depth for this one since we didn't do any work.
      computeKnownBitsImpl(Src.getReg(), Known, DemandedElts, Depth);
    }
    break;
  }
  case TargetOpcode::G_CONSTANT: {
    auto CstVal = getConstantVRegVal(R, MRI);
    if (!CstVal)
      break;
    Known.One = *CstVal;
    Known.Zero = ~Known.One;
    break;
  }
  case TargetOpcode::G_FRAME_INDEX: {
    computeKnownBitsForFrameIndex(R, Known, DemandedElts);
    break;
  }
  case TargetOpcode::G_SUB: {
    // If low bits are known to be zero in both operands, then we know they are
    // going to be 0 in the result. Both addition and complement operations
    // preserve the low zero bits.
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known2, DemandedElts,
                         Depth + 1);
    unsigned KnownZeroLow = Known2.countMinTrailingZeros();
    if (KnownZeroLow == 0)
      break;
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known2, DemandedElts,
                         Depth + 1);
    KnownZeroLow = std::min(KnownZeroLow, Known2.countMinTrailingZeros());
    Known.Zero.setLowBits(KnownZeroLow);
    break;
  }
  case TargetOpcode::G_XOR: {
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known2, DemandedElts,
                         Depth + 1);

    // Output known-0 bits are known if clear or set in both the LHS & RHS.
    APInt KnownZeroOut = (Known.Zero & Known2.Zero) | (Known.One & Known2.One);
    // Output known-1 are known to be set if set in only one of the LHS, RHS.
    Known.One = (Known.Zero & Known2.One) | (Known.One & Known2.Zero);
    Known.Zero = KnownZeroOut;
    break;
  }
  case TargetOpcode::G_PTR_ADD: {
    // G_PTR_ADD is like G_ADD. FIXME: Is this true for all targets?
    LLT Ty = MRI.getType(MI.getOperand(1).getReg());
    if (DL.isNonIntegralAddressSpace(Ty.getAddressSpace()))
      break;
    LLVM_FALLTHROUGH;
  }
  case TargetOpcode::G_ADD: {
    // Output known-0 bits are known if clear or set in both the low clear bits
    // common to both LHS & RHS.  For example, 8+(X<<3) is known to have the
    // low 3 bits clear.
    // Output known-0 bits are also known if the top bits of each input are
    // known to be clear. For example, if one input has the top 10 bits clear
    // and the other has the top 8 bits clear, we know the top 7 bits of the
    // output must be clear.
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known2, DemandedElts,
                         Depth + 1);
    unsigned KnownZeroHigh = Known2.countMinLeadingZeros();
    unsigned KnownZeroLow = Known2.countMinTrailingZeros();
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known2, DemandedElts,
                         Depth + 1);
    KnownZeroHigh = std::min(KnownZeroHigh, Known2.countMinLeadingZeros());
    KnownZeroLow = std::min(KnownZeroLow, Known2.countMinTrailingZeros());
    Known.Zero.setLowBits(KnownZeroLow);
    if (KnownZeroHigh > 1)
      Known.Zero.setHighBits(KnownZeroHigh - 1);
    break;
  }
  case TargetOpcode::G_AND: {
    // If either the LHS or the RHS are Zero, the result is zero.
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known2, DemandedElts,
                         Depth + 1);

    // Output known-1 bits are only known if set in both the LHS & RHS.
    Known.One &= Known2.One;
    // Output known-0 are known to be clear if zero in either the LHS | RHS.
    Known.Zero |= Known2.Zero;
    break;
  }
  case TargetOpcode::G_OR: {
    // If either the LHS or the RHS are Zero, the result is zero.
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known2, DemandedElts,
                         Depth + 1);

    // Output known-0 bits are only known if clear in both the LHS & RHS.
    Known.Zero &= Known2.Zero;
    // Output known-1 are known to be set if set in either the LHS | RHS.
    Known.One |= Known2.One;
    break;
  }
  case TargetOpcode::G_MUL: {
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known2, DemandedElts,
                         Depth + 1);
    // If low bits are zero in either operand, output low known-0 bits.
    // Also compute a conservative estimate for high known-0 bits.
    // More trickiness is possible, but this is sufficient for the
    // interesting case of alignment computation.
    unsigned TrailZ =
        Known.countMinTrailingZeros() + Known2.countMinTrailingZeros();
    unsigned LeadZ =
        std::max(Known.countMinLeadingZeros() + Known2.countMinLeadingZeros(),
                 BitWidth) -
        BitWidth;

    Known.resetAll();
    Known.Zero.setLowBits(std::min(TrailZ, BitWidth));
    Known.Zero.setHighBits(std::min(LeadZ, BitWidth));
    break;
  }
  case TargetOpcode::G_SELECT: {
    computeKnownBitsImpl(MI.getOperand(3).getReg(), Known, DemandedElts,
                         Depth + 1);
    // If we don't know any bits, early out.
    if (Known.isUnknown())
      break;
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known2, DemandedElts,
                         Depth + 1);
    // Only known if known in both the LHS and RHS.
    Known.One &= Known2.One;
    Known.Zero &= Known2.Zero;
    break;
  }
  case TargetOpcode::G_FCMP:
  case TargetOpcode::G_ICMP: {
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
  case TargetOpcode::G_ANYEXT: {
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known, DemandedElts,
                         Depth + 1);
    Known = Known.zext(BitWidth, true /* ExtendedBitsAreKnownZero */);
    break;
  }
  case TargetOpcode::G_LOAD: {
    if (MI.hasOneMemOperand()) {
      const MachineMemOperand *MMO = *MI.memoperands_begin();
      if (const MDNode *Ranges = MMO->getRanges()) {
        computeKnownBitsFromRangeMetadata(*Ranges, Known);
      }
    }
    break;
  }
  case TargetOpcode::G_ZEXTLOAD: {
    // Everything above the retrieved bits is zero
    if (MI.hasOneMemOperand())
      Known.Zero.setBitsFrom((*MI.memoperands_begin())->getSizeInBits());
    break;
  }
  case TargetOpcode::G_ASHR:
  case TargetOpcode::G_LSHR:
  case TargetOpcode::G_SHL: {
    KnownBits RHSKnown;
    computeKnownBitsImpl(MI.getOperand(2).getReg(), RHSKnown, DemandedElts,
                         Depth + 1);
    if (!RHSKnown.isConstant()) {
      LLVM_DEBUG(
          MachineInstr *RHSMI = MRI.getVRegDef(MI.getOperand(2).getReg());
          dbgs() << '[' << Depth << "] Shift not known constant: " << *RHSMI);
      break;
    }
    uint64_t Shift = RHSKnown.getConstant().getZExtValue();
    LLVM_DEBUG(dbgs() << '[' << Depth << "] Shift is " << Shift << '\n');

    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known, DemandedElts,
                         Depth + 1);

    switch (Opcode) {
    case TargetOpcode::G_ASHR:
      Known.Zero = Known.Zero.ashr(Shift);
      Known.One = Known.One.ashr(Shift);
      break;
    case TargetOpcode::G_LSHR:
      Known.Zero = Known.Zero.lshr(Shift);
      Known.One = Known.One.lshr(Shift);
      Known.Zero.setBitsFrom(Known.Zero.getBitWidth() - Shift);
      break;
    case TargetOpcode::G_SHL:
      Known.Zero = Known.Zero.shl(Shift);
      Known.One = Known.One.shl(Shift);
      Known.Zero.setBits(0, Shift);
      break;
    }
    break;
  }
  case TargetOpcode::G_INTTOPTR:
  case TargetOpcode::G_PTRTOINT:
    // Fall through and handle them the same as zext/trunc.
    LLVM_FALLTHROUGH;
  case TargetOpcode::G_ZEXT:
  case TargetOpcode::G_TRUNC: {
    Register SrcReg = MI.getOperand(1).getReg();
    LLT SrcTy = MRI.getType(SrcReg);
    unsigned SrcBitWidth = SrcTy.isPointer()
                               ? DL.getIndexSizeInBits(SrcTy.getAddressSpace())
                               : SrcTy.getSizeInBits();
    assert(SrcBitWidth && "SrcBitWidth can't be zero");
    Known = Known.zextOrTrunc(SrcBitWidth, true);
    computeKnownBitsImpl(SrcReg, Known, DemandedElts, Depth + 1);
    Known = Known.zextOrTrunc(BitWidth, true);
    if (BitWidth > SrcBitWidth)
      Known.Zero.setBitsFrom(SrcBitWidth);
    break;
  }
  }

  assert(!Known.hasConflict() && "Bits known to be one AND zero?");
  LLVM_DEBUG(dbgs() << "[" << Depth << "] Compute known bits: " << MI << "["
                    << Depth << "] Computed for: " << MI << "[" << Depth
                    << "] Known: 0x"
                    << (Known.Zero | Known.One).toString(16, false) << "\n"
                    << "[" << Depth << "] Zero: 0x"
                    << Known.Zero.toString(16, false) << "\n"
                    << "[" << Depth << "] One:  0x"
                    << Known.One.toString(16, false) << "\n");
}

void GISelKnownBitsAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool GISelKnownBitsAnalysis::runOnMachineFunction(MachineFunction &MF) {
  return false;
}
