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

INITIALIZE_PASS(GISelKnownBitsAnalysis, DEBUG_TYPE,
                "Analysis for ComputingKnownBits", false, true)

GISelKnownBits::GISelKnownBits(MachineFunction &MF, unsigned MaxDepth)
    : MF(MF), MRI(MF.getRegInfo()), TL(*MF.getSubtarget().getTargetLowering()),
      DL(MF.getFunction().getParent()->getDataLayout()), MaxDepth(MaxDepth) {}

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
  // For now, we only maintain the cache during one request.
  assert(ComputeKnownBitsCache.empty() && "Cache should have been cleared");
  computeKnownBitsImpl(R, Known, DemandedElts);
  ComputeKnownBitsCache.clear();
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

LLVM_ATTRIBUTE_UNUSED static void
dumpResult(const MachineInstr &MI, const KnownBits &Known, unsigned Depth) {
  dbgs() << "[" << Depth << "] Compute known bits: " << MI << "[" << Depth
         << "] Computed for: " << MI << "[" << Depth << "] Known: 0x"
         << (Known.Zero | Known.One).toString(16, false) << "\n"
         << "[" << Depth << "] Zero: 0x" << Known.Zero.toString(16, false)
         << "\n"
         << "[" << Depth << "] One:  0x" << Known.One.toString(16, false)
         << "\n";
}

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
  auto CacheEntry = ComputeKnownBitsCache.find(R);
  if (CacheEntry != ComputeKnownBitsCache.end()) {
    Known = CacheEntry->second;
    LLVM_DEBUG(dbgs() << "Cache hit at ");
    LLVM_DEBUG(dumpResult(MI, Known, Depth));
    assert(Known.getBitWidth() == BitWidth && "Cache entry size doesn't match");
    return;
  }
  Known = KnownBits(BitWidth); // Don't know anything

  if (DstTy.isVector())
    return; // TODO: Handle vectors.

  // Depth may get bigger than max depth if it gets passed to a different
  // GISelKnownBits object.
  // This may happen when say a generic part uses a GISelKnownBits object
  // with some max depth, but then we hit TL.computeKnownBitsForTargetInstr
  // which creates a new GISelKnownBits object with a different and smaller
  // depth. If we just check for equality, we would never exit if the depth
  // that is passed down to the target specific GISelKnownBits object is
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
  case TargetOpcode::COPY:
  case TargetOpcode::G_PHI:
  case TargetOpcode::PHI: {
    Known.One = APInt::getAllOnesValue(BitWidth);
    Known.Zero = APInt::getAllOnesValue(BitWidth);
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
        Known.One &= Known2.One;
        Known.Zero &= Known2.Zero;
        // If we reach a point where we don't know anything
        // just stop looking through the operands.
        if (Known.One == 0 && Known.Zero == 0)
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
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known2, DemandedElts,
                         Depth + 1);
    Known = KnownBits::computeForAddSub(/*Add*/ false, /*NSW*/ false, Known,
                                        Known2);
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
    computeKnownBitsImpl(MI.getOperand(1).getReg(), Known, DemandedElts,
                         Depth + 1);
    computeKnownBitsImpl(MI.getOperand(2).getReg(), Known2, DemandedElts,
                         Depth + 1);
    Known =
        KnownBits::computeForAddSub(/*Add*/ true, /*NSW*/ false, Known, Known2);
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
    Known = Known.zext(BitWidth);
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
    Known = Known.zextOrTrunc(SrcBitWidth);
    computeKnownBitsImpl(SrcReg, Known, DemandedElts, Depth + 1);
    Known = Known.zextOrTrunc(BitWidth);
    if (BitWidth > SrcBitWidth)
      Known.Zero.setBitsFrom(SrcBitWidth);
    break;
  }
  }

  assert(!Known.hasConflict() && "Bits known to be one AND zero?");
  LLVM_DEBUG(dumpResult(MI, Known, Depth));

  // Update the cache.
  ComputeKnownBitsCache[R] = Known;
}

unsigned GISelKnownBits::computeNumSignBits(Register R,
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

  // Handle the case where this is called on a register that does not have a
  // type constraint. This is unlikely to occur except by looking through copies
  // but it is possible for the initial register being queried to be in this
  // state.
  if (!DstTy.isValid())
    return 1;

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
  default:
    break;
  }

  // TODO: Handle target instructions
  // TODO: Fall back to known bits
  return 1;
}

unsigned GISelKnownBits::computeNumSignBits(Register R, unsigned Depth) {
  LLT Ty = MRI.getType(R);
  APInt DemandedElts = Ty.isVector()
                           ? APInt::getAllOnesValue(Ty.getNumElements())
                           : APInt(1, 1);
  return computeNumSignBits(R, DemandedElts, Depth);
}

void GISelKnownBitsAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool GISelKnownBitsAnalysis::runOnMachineFunction(MachineFunction &MF) {
  return false;
}
