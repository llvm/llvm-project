//===- ARMTargetTransformInfo.cpp - ARM specific TTI ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ARMTargetTransformInfo.h"
#include "ARMSubtarget.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/CostTable.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsARM.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MachineValueType.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "armtti"

static cl::opt<bool> EnableMaskedLoadStores(
  "enable-arm-maskedldst", cl::Hidden, cl::init(true),
  cl::desc("Enable the generation of masked loads and stores"));

static cl::opt<bool> DisableLowOverheadLoops(
  "disable-arm-loloops", cl::Hidden, cl::init(false),
  cl::desc("Disable the generation of low-overhead loops"));

extern cl::opt<bool> DisableTailPredication;

extern cl::opt<bool> EnableMaskedGatherScatters;

bool ARMTTIImpl::areInlineCompatible(const Function *Caller,
                                     const Function *Callee) const {
  const TargetMachine &TM = getTLI()->getTargetMachine();
  const FeatureBitset &CallerBits =
      TM.getSubtargetImpl(*Caller)->getFeatureBits();
  const FeatureBitset &CalleeBits =
      TM.getSubtargetImpl(*Callee)->getFeatureBits();

  // To inline a callee, all features not in the whitelist must match exactly.
  bool MatchExact = (CallerBits & ~InlineFeatureWhitelist) ==
                    (CalleeBits & ~InlineFeatureWhitelist);
  // For features in the whitelist, the callee's features must be a subset of
  // the callers'.
  bool MatchSubset = ((CallerBits & CalleeBits) & InlineFeatureWhitelist) ==
                     (CalleeBits & InlineFeatureWhitelist);
  return MatchExact && MatchSubset;
}

bool ARMTTIImpl::shouldFavorBackedgeIndex(const Loop *L) const {
  if (L->getHeader()->getParent()->hasOptSize())
    return false;
  if (ST->hasMVEIntegerOps())
    return false;
  return ST->isMClass() && ST->isThumb2() && L->getNumBlocks() == 1;
}

bool ARMTTIImpl::shouldFavorPostInc() const {
  if (ST->hasMVEIntegerOps())
    return true;
  return false;
}

int ARMTTIImpl::getIntImmCost(const APInt &Imm, Type *Ty,
                              TTI::TargetCostKind CostKind) {
  assert(Ty->isIntegerTy());

 unsigned Bits = Ty->getPrimitiveSizeInBits();
 if (Bits == 0 || Imm.getActiveBits() >= 64)
   return 4;

  int64_t SImmVal = Imm.getSExtValue();
  uint64_t ZImmVal = Imm.getZExtValue();
  if (!ST->isThumb()) {
    if ((SImmVal >= 0 && SImmVal < 65536) ||
        (ARM_AM::getSOImmVal(ZImmVal) != -1) ||
        (ARM_AM::getSOImmVal(~ZImmVal) != -1))
      return 1;
    return ST->hasV6T2Ops() ? 2 : 3;
  }
  if (ST->isThumb2()) {
    if ((SImmVal >= 0 && SImmVal < 65536) ||
        (ARM_AM::getT2SOImmVal(ZImmVal) != -1) ||
        (ARM_AM::getT2SOImmVal(~ZImmVal) != -1))
      return 1;
    return ST->hasV6T2Ops() ? 2 : 3;
  }
  // Thumb1, any i8 imm cost 1.
  if (Bits == 8 || (SImmVal >= 0 && SImmVal < 256))
    return 1;
  if ((~SImmVal < 256) || ARM_AM::isThumbImmShiftedVal(ZImmVal))
    return 2;
  // Load from constantpool.
  return 3;
}

// Constants smaller than 256 fit in the immediate field of
// Thumb1 instructions so we return a zero cost and 1 otherwise.
int ARMTTIImpl::getIntImmCodeSizeCost(unsigned Opcode, unsigned Idx,
                                      const APInt &Imm, Type *Ty) {
  if (Imm.isNonNegative() && Imm.getLimitedValue() < 256)
    return 0;

  return 1;
}

int ARMTTIImpl::getIntImmCostInst(unsigned Opcode, unsigned Idx, const APInt &Imm,
                                  Type *Ty, TTI::TargetCostKind CostKind) {
  // Division by a constant can be turned into multiplication, but only if we
  // know it's constant. So it's not so much that the immediate is cheap (it's
  // not), but that the alternative is worse.
  // FIXME: this is probably unneeded with GlobalISel.
  if ((Opcode == Instruction::SDiv || Opcode == Instruction::UDiv ||
       Opcode == Instruction::SRem || Opcode == Instruction::URem) &&
      Idx == 1)
    return 0;

  if (Opcode == Instruction::And) {
    // UXTB/UXTH
    if (Imm == 255 || Imm == 65535)
      return 0;
    // Conversion to BIC is free, and means we can use ~Imm instead.
    return std::min(getIntImmCost(Imm, Ty, CostKind),
                    getIntImmCost(~Imm, Ty, CostKind));
  }

  if (Opcode == Instruction::Add)
    // Conversion to SUB is free, and means we can use -Imm instead.
    return std::min(getIntImmCost(Imm, Ty, CostKind),
                    getIntImmCost(-Imm, Ty, CostKind));

  if (Opcode == Instruction::ICmp && Imm.isNegative() &&
      Ty->getIntegerBitWidth() == 32) {
    int64_t NegImm = -Imm.getSExtValue();
    if (ST->isThumb2() && NegImm < 1<<12)
      // icmp X, #-C -> cmn X, #C
      return 0;
    if (ST->isThumb() && NegImm < 1<<8)
      // icmp X, #-C -> adds X, #C
      return 0;
  }

  // xor a, -1 can always be folded to MVN
  if (Opcode == Instruction::Xor && Imm.isAllOnesValue())
    return 0;

  return getIntImmCost(Imm, Ty, CostKind);
}

int ARMTTIImpl::getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                                 TTI::TargetCostKind CostKind,
                                 const Instruction *I) {
  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  // Single to/from double precision conversions.
  static const CostTblEntry NEONFltDblTbl[] = {
    // Vector fptrunc/fpext conversions.
    { ISD::FP_ROUND,   MVT::v2f64, 2 },
    { ISD::FP_EXTEND,  MVT::v2f32, 2 },
    { ISD::FP_EXTEND,  MVT::v4f32, 4 }
  };

  if (Src->isVectorTy() && ST->hasNEON() && (ISD == ISD::FP_ROUND ||
                                          ISD == ISD::FP_EXTEND)) {
    std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, Src);
    if (const auto *Entry = CostTableLookup(NEONFltDblTbl, ISD, LT.second))
      return LT.first * Entry->Cost;
  }

  EVT SrcTy = TLI->getValueType(DL, Src);
  EVT DstTy = TLI->getValueType(DL, Dst);

  if (!SrcTy.isSimple() || !DstTy.isSimple())
    return BaseT::getCastInstrCost(Opcode, Dst, Src, CostKind, I);

  // The extend of a load is free
  if (I && isa<LoadInst>(I->getOperand(0))) {
    static const TypeConversionCostTblEntry LoadConversionTbl[] = {
        {ISD::SIGN_EXTEND, MVT::i32, MVT::i16, 0},
        {ISD::ZERO_EXTEND, MVT::i32, MVT::i16, 0},
        {ISD::SIGN_EXTEND, MVT::i32, MVT::i8, 0},
        {ISD::ZERO_EXTEND, MVT::i32, MVT::i8, 0},
        {ISD::SIGN_EXTEND, MVT::i16, MVT::i8, 0},
        {ISD::ZERO_EXTEND, MVT::i16, MVT::i8, 0},
        {ISD::SIGN_EXTEND, MVT::i64, MVT::i32, 1},
        {ISD::ZERO_EXTEND, MVT::i64, MVT::i32, 1},
        {ISD::SIGN_EXTEND, MVT::i64, MVT::i16, 1},
        {ISD::ZERO_EXTEND, MVT::i64, MVT::i16, 1},
        {ISD::SIGN_EXTEND, MVT::i64, MVT::i8, 1},
        {ISD::ZERO_EXTEND, MVT::i64, MVT::i8, 1},
    };
    if (const auto *Entry = ConvertCostTableLookup(
            LoadConversionTbl, ISD, DstTy.getSimpleVT(), SrcTy.getSimpleVT()))
      return Entry->Cost;

    static const TypeConversionCostTblEntry MVELoadConversionTbl[] = {
        {ISD::SIGN_EXTEND, MVT::v4i32, MVT::v4i16, 0},
        {ISD::ZERO_EXTEND, MVT::v4i32, MVT::v4i16, 0},
        {ISD::SIGN_EXTEND, MVT::v4i32, MVT::v4i8, 0},
        {ISD::ZERO_EXTEND, MVT::v4i32, MVT::v4i8, 0},
        {ISD::SIGN_EXTEND, MVT::v8i16, MVT::v8i8, 0},
        {ISD::ZERO_EXTEND, MVT::v8i16, MVT::v8i8, 0},
    };
    if (SrcTy.isVector() && ST->hasMVEIntegerOps()) {
      if (const auto *Entry =
              ConvertCostTableLookup(MVELoadConversionTbl, ISD,
                                     DstTy.getSimpleVT(), SrcTy.getSimpleVT()))
        return Entry->Cost;
    }
  }

  // NEON vector operations that can extend their inputs.
  if ((ISD == ISD::SIGN_EXTEND || ISD == ISD::ZERO_EXTEND) &&
      I && I->hasOneUse() && ST->hasNEON() && SrcTy.isVector()) {
    static const TypeConversionCostTblEntry NEONDoubleWidthTbl[] = {
      // vaddl
      { ISD::ADD, MVT::v4i32, MVT::v4i16, 0 },
      { ISD::ADD, MVT::v8i16, MVT::v8i8,  0 },
      // vsubl
      { ISD::SUB, MVT::v4i32, MVT::v4i16, 0 },
      { ISD::SUB, MVT::v8i16, MVT::v8i8,  0 },
      // vmull
      { ISD::MUL, MVT::v4i32, MVT::v4i16, 0 },
      { ISD::MUL, MVT::v8i16, MVT::v8i8,  0 },
      // vshll
      { ISD::SHL, MVT::v4i32, MVT::v4i16, 0 },
      { ISD::SHL, MVT::v8i16, MVT::v8i8,  0 },
    };

    auto *User = cast<Instruction>(*I->user_begin());
    int UserISD = TLI->InstructionOpcodeToISD(User->getOpcode());
    if (auto *Entry = ConvertCostTableLookup(NEONDoubleWidthTbl, UserISD,
                                             DstTy.getSimpleVT(),
                                             SrcTy.getSimpleVT())) {
      return Entry->Cost;
    }
  }

  // Some arithmetic, load and store operations have specific instructions
  // to cast up/down their types automatically at no extra cost.
  // TODO: Get these tables to know at least what the related operations are.
  static const TypeConversionCostTblEntry NEONVectorConversionTbl[] = {
    { ISD::SIGN_EXTEND, MVT::v4i32, MVT::v4i16, 1 },
    { ISD::ZERO_EXTEND, MVT::v4i32, MVT::v4i16, 1 },
    { ISD::SIGN_EXTEND, MVT::v2i64, MVT::v2i32, 1 },
    { ISD::ZERO_EXTEND, MVT::v2i64, MVT::v2i32, 1 },
    { ISD::TRUNCATE,    MVT::v4i32, MVT::v4i64, 0 },
    { ISD::TRUNCATE,    MVT::v4i16, MVT::v4i32, 1 },

    // The number of vmovl instructions for the extension.
    { ISD::SIGN_EXTEND, MVT::v8i16, MVT::v8i8,  1 },
    { ISD::ZERO_EXTEND, MVT::v8i16, MVT::v8i8,  1 },
    { ISD::SIGN_EXTEND, MVT::v4i32, MVT::v4i8,  2 },
    { ISD::ZERO_EXTEND, MVT::v4i32, MVT::v4i8,  2 },
    { ISD::SIGN_EXTEND, MVT::v2i64, MVT::v2i8,  3 },
    { ISD::ZERO_EXTEND, MVT::v2i64, MVT::v2i8,  3 },
    { ISD::SIGN_EXTEND, MVT::v2i64, MVT::v2i16, 2 },
    { ISD::ZERO_EXTEND, MVT::v2i64, MVT::v2i16, 2 },
    { ISD::SIGN_EXTEND, MVT::v4i64, MVT::v4i16, 3 },
    { ISD::ZERO_EXTEND, MVT::v4i64, MVT::v4i16, 3 },
    { ISD::SIGN_EXTEND, MVT::v8i32, MVT::v8i8, 3 },
    { ISD::ZERO_EXTEND, MVT::v8i32, MVT::v8i8, 3 },
    { ISD::SIGN_EXTEND, MVT::v8i64, MVT::v8i8, 7 },
    { ISD::ZERO_EXTEND, MVT::v8i64, MVT::v8i8, 7 },
    { ISD::SIGN_EXTEND, MVT::v8i64, MVT::v8i16, 6 },
    { ISD::ZERO_EXTEND, MVT::v8i64, MVT::v8i16, 6 },
    { ISD::SIGN_EXTEND, MVT::v16i32, MVT::v16i8, 6 },
    { ISD::ZERO_EXTEND, MVT::v16i32, MVT::v16i8, 6 },

    // Operations that we legalize using splitting.
    { ISD::TRUNCATE,    MVT::v16i8, MVT::v16i32, 6 },
    { ISD::TRUNCATE,    MVT::v8i8, MVT::v8i32, 3 },

    // Vector float <-> i32 conversions.
    { ISD::SINT_TO_FP,  MVT::v4f32, MVT::v4i32, 1 },
    { ISD::UINT_TO_FP,  MVT::v4f32, MVT::v4i32, 1 },

    { ISD::SINT_TO_FP,  MVT::v2f32, MVT::v2i8, 3 },
    { ISD::UINT_TO_FP,  MVT::v2f32, MVT::v2i8, 3 },
    { ISD::SINT_TO_FP,  MVT::v2f32, MVT::v2i16, 2 },
    { ISD::UINT_TO_FP,  MVT::v2f32, MVT::v2i16, 2 },
    { ISD::SINT_TO_FP,  MVT::v2f32, MVT::v2i32, 1 },
    { ISD::UINT_TO_FP,  MVT::v2f32, MVT::v2i32, 1 },
    { ISD::SINT_TO_FP,  MVT::v4f32, MVT::v4i1, 3 },
    { ISD::UINT_TO_FP,  MVT::v4f32, MVT::v4i1, 3 },
    { ISD::SINT_TO_FP,  MVT::v4f32, MVT::v4i8, 3 },
    { ISD::UINT_TO_FP,  MVT::v4f32, MVT::v4i8, 3 },
    { ISD::SINT_TO_FP,  MVT::v4f32, MVT::v4i16, 2 },
    { ISD::UINT_TO_FP,  MVT::v4f32, MVT::v4i16, 2 },
    { ISD::SINT_TO_FP,  MVT::v8f32, MVT::v8i16, 4 },
    { ISD::UINT_TO_FP,  MVT::v8f32, MVT::v8i16, 4 },
    { ISD::SINT_TO_FP,  MVT::v8f32, MVT::v8i32, 2 },
    { ISD::UINT_TO_FP,  MVT::v8f32, MVT::v8i32, 2 },
    { ISD::SINT_TO_FP,  MVT::v16f32, MVT::v16i16, 8 },
    { ISD::UINT_TO_FP,  MVT::v16f32, MVT::v16i16, 8 },
    { ISD::SINT_TO_FP,  MVT::v16f32, MVT::v16i32, 4 },
    { ISD::UINT_TO_FP,  MVT::v16f32, MVT::v16i32, 4 },

    { ISD::FP_TO_SINT,  MVT::v4i32, MVT::v4f32, 1 },
    { ISD::FP_TO_UINT,  MVT::v4i32, MVT::v4f32, 1 },
    { ISD::FP_TO_SINT,  MVT::v4i8, MVT::v4f32, 3 },
    { ISD::FP_TO_UINT,  MVT::v4i8, MVT::v4f32, 3 },
    { ISD::FP_TO_SINT,  MVT::v4i16, MVT::v4f32, 2 },
    { ISD::FP_TO_UINT,  MVT::v4i16, MVT::v4f32, 2 },

    // Vector double <-> i32 conversions.
    { ISD::SINT_TO_FP,  MVT::v2f64, MVT::v2i32, 2 },
    { ISD::UINT_TO_FP,  MVT::v2f64, MVT::v2i32, 2 },

    { ISD::SINT_TO_FP,  MVT::v2f64, MVT::v2i8, 4 },
    { ISD::UINT_TO_FP,  MVT::v2f64, MVT::v2i8, 4 },
    { ISD::SINT_TO_FP,  MVT::v2f64, MVT::v2i16, 3 },
    { ISD::UINT_TO_FP,  MVT::v2f64, MVT::v2i16, 3 },
    { ISD::SINT_TO_FP,  MVT::v2f64, MVT::v2i32, 2 },
    { ISD::UINT_TO_FP,  MVT::v2f64, MVT::v2i32, 2 },

    { ISD::FP_TO_SINT,  MVT::v2i32, MVT::v2f64, 2 },
    { ISD::FP_TO_UINT,  MVT::v2i32, MVT::v2f64, 2 },
    { ISD::FP_TO_SINT,  MVT::v8i16, MVT::v8f32, 4 },
    { ISD::FP_TO_UINT,  MVT::v8i16, MVT::v8f32, 4 },
    { ISD::FP_TO_SINT,  MVT::v16i16, MVT::v16f32, 8 },
    { ISD::FP_TO_UINT,  MVT::v16i16, MVT::v16f32, 8 }
  };

  if (SrcTy.isVector() && ST->hasNEON()) {
    if (const auto *Entry = ConvertCostTableLookup(NEONVectorConversionTbl, ISD,
                                                   DstTy.getSimpleVT(),
                                                   SrcTy.getSimpleVT()))
      return Entry->Cost;
  }

  // Scalar float to integer conversions.
  static const TypeConversionCostTblEntry NEONFloatConversionTbl[] = {
    { ISD::FP_TO_SINT,  MVT::i1, MVT::f32, 2 },
    { ISD::FP_TO_UINT,  MVT::i1, MVT::f32, 2 },
    { ISD::FP_TO_SINT,  MVT::i1, MVT::f64, 2 },
    { ISD::FP_TO_UINT,  MVT::i1, MVT::f64, 2 },
    { ISD::FP_TO_SINT,  MVT::i8, MVT::f32, 2 },
    { ISD::FP_TO_UINT,  MVT::i8, MVT::f32, 2 },
    { ISD::FP_TO_SINT,  MVT::i8, MVT::f64, 2 },
    { ISD::FP_TO_UINT,  MVT::i8, MVT::f64, 2 },
    { ISD::FP_TO_SINT,  MVT::i16, MVT::f32, 2 },
    { ISD::FP_TO_UINT,  MVT::i16, MVT::f32, 2 },
    { ISD::FP_TO_SINT,  MVT::i16, MVT::f64, 2 },
    { ISD::FP_TO_UINT,  MVT::i16, MVT::f64, 2 },
    { ISD::FP_TO_SINT,  MVT::i32, MVT::f32, 2 },
    { ISD::FP_TO_UINT,  MVT::i32, MVT::f32, 2 },
    { ISD::FP_TO_SINT,  MVT::i32, MVT::f64, 2 },
    { ISD::FP_TO_UINT,  MVT::i32, MVT::f64, 2 },
    { ISD::FP_TO_SINT,  MVT::i64, MVT::f32, 10 },
    { ISD::FP_TO_UINT,  MVT::i64, MVT::f32, 10 },
    { ISD::FP_TO_SINT,  MVT::i64, MVT::f64, 10 },
    { ISD::FP_TO_UINT,  MVT::i64, MVT::f64, 10 }
  };
  if (SrcTy.isFloatingPoint() && ST->hasNEON()) {
    if (const auto *Entry = ConvertCostTableLookup(NEONFloatConversionTbl, ISD,
                                                   DstTy.getSimpleVT(),
                                                   SrcTy.getSimpleVT()))
      return Entry->Cost;
  }

  // Scalar integer to float conversions.
  static const TypeConversionCostTblEntry NEONIntegerConversionTbl[] = {
    { ISD::SINT_TO_FP,  MVT::f32, MVT::i1, 2 },
    { ISD::UINT_TO_FP,  MVT::f32, MVT::i1, 2 },
    { ISD::SINT_TO_FP,  MVT::f64, MVT::i1, 2 },
    { ISD::UINT_TO_FP,  MVT::f64, MVT::i1, 2 },
    { ISD::SINT_TO_FP,  MVT::f32, MVT::i8, 2 },
    { ISD::UINT_TO_FP,  MVT::f32, MVT::i8, 2 },
    { ISD::SINT_TO_FP,  MVT::f64, MVT::i8, 2 },
    { ISD::UINT_TO_FP,  MVT::f64, MVT::i8, 2 },
    { ISD::SINT_TO_FP,  MVT::f32, MVT::i16, 2 },
    { ISD::UINT_TO_FP,  MVT::f32, MVT::i16, 2 },
    { ISD::SINT_TO_FP,  MVT::f64, MVT::i16, 2 },
    { ISD::UINT_TO_FP,  MVT::f64, MVT::i16, 2 },
    { ISD::SINT_TO_FP,  MVT::f32, MVT::i32, 2 },
    { ISD::UINT_TO_FP,  MVT::f32, MVT::i32, 2 },
    { ISD::SINT_TO_FP,  MVT::f64, MVT::i32, 2 },
    { ISD::UINT_TO_FP,  MVT::f64, MVT::i32, 2 },
    { ISD::SINT_TO_FP,  MVT::f32, MVT::i64, 10 },
    { ISD::UINT_TO_FP,  MVT::f32, MVT::i64, 10 },
    { ISD::SINT_TO_FP,  MVT::f64, MVT::i64, 10 },
    { ISD::UINT_TO_FP,  MVT::f64, MVT::i64, 10 }
  };

  if (SrcTy.isInteger() && ST->hasNEON()) {
    if (const auto *Entry = ConvertCostTableLookup(NEONIntegerConversionTbl,
                                                   ISD, DstTy.getSimpleVT(),
                                                   SrcTy.getSimpleVT()))
      return Entry->Cost;
  }

  // MVE extend costs, taken from codegen tests. i8->i16 or i16->i32 is one
  // instruction, i8->i32 is two. i64 zexts are an VAND with a constant, sext
  // are linearised so take more.
  static const TypeConversionCostTblEntry MVEVectorConversionTbl[] = {
    { ISD::SIGN_EXTEND, MVT::v8i16, MVT::v8i8, 1 },
    { ISD::ZERO_EXTEND, MVT::v8i16, MVT::v8i8, 1 },
    { ISD::SIGN_EXTEND, MVT::v4i32, MVT::v4i8, 2 },
    { ISD::ZERO_EXTEND, MVT::v4i32, MVT::v4i8, 2 },
    { ISD::SIGN_EXTEND, MVT::v2i64, MVT::v2i8, 10 },
    { ISD::ZERO_EXTEND, MVT::v2i64, MVT::v2i8, 2 },
    { ISD::SIGN_EXTEND, MVT::v4i32, MVT::v4i16, 1 },
    { ISD::ZERO_EXTEND, MVT::v4i32, MVT::v4i16, 1 },
    { ISD::SIGN_EXTEND, MVT::v2i64, MVT::v2i16, 10 },
    { ISD::ZERO_EXTEND, MVT::v2i64, MVT::v2i16, 2 },
    { ISD::SIGN_EXTEND, MVT::v2i64, MVT::v2i32, 8 },
    { ISD::ZERO_EXTEND, MVT::v2i64, MVT::v2i32, 2 },
  };

  if (SrcTy.isVector() && ST->hasMVEIntegerOps()) {
    if (const auto *Entry = ConvertCostTableLookup(MVEVectorConversionTbl,
                                                   ISD, DstTy.getSimpleVT(),
                                                   SrcTy.getSimpleVT()))
      return Entry->Cost * ST->getMVEVectorCostFactor();
  }

  // Scalar integer conversion costs.
  static const TypeConversionCostTblEntry ARMIntegerConversionTbl[] = {
    // i16 -> i64 requires two dependent operations.
    { ISD::SIGN_EXTEND, MVT::i64, MVT::i16, 2 },

    // Truncates on i64 are assumed to be free.
    { ISD::TRUNCATE,    MVT::i32, MVT::i64, 0 },
    { ISD::TRUNCATE,    MVT::i16, MVT::i64, 0 },
    { ISD::TRUNCATE,    MVT::i8,  MVT::i64, 0 },
    { ISD::TRUNCATE,    MVT::i1,  MVT::i64, 0 }
  };

  if (SrcTy.isInteger()) {
    if (const auto *Entry = ConvertCostTableLookup(ARMIntegerConversionTbl, ISD,
                                                   DstTy.getSimpleVT(),
                                                   SrcTy.getSimpleVT()))
      return Entry->Cost;
  }

  int BaseCost = ST->hasMVEIntegerOps() && Src->isVectorTy()
                     ? ST->getMVEVectorCostFactor()
                     : 1;
  return BaseCost * BaseT::getCastInstrCost(Opcode, Dst, Src, CostKind, I);
}

int ARMTTIImpl::getVectorInstrCost(unsigned Opcode, Type *ValTy,
                                   unsigned Index) {
  // Penalize inserting into an D-subregister. We end up with a three times
  // lower estimated throughput on swift.
  if (ST->hasSlowLoadDSubregister() && Opcode == Instruction::InsertElement &&
      ValTy->isVectorTy() && ValTy->getScalarSizeInBits() <= 32)
    return 3;

  if (ST->hasNEON() && (Opcode == Instruction::InsertElement ||
                        Opcode == Instruction::ExtractElement)) {
    // Cross-class copies are expensive on many microarchitectures,
    // so assume they are expensive by default.
    if (cast<VectorType>(ValTy)->getElementType()->isIntegerTy())
      return 3;

    // Even if it's not a cross class copy, this likely leads to mixing
    // of NEON and VFP code and should be therefore penalized.
    if (ValTy->isVectorTy() &&
        ValTy->getScalarSizeInBits() <= 32)
      return std::max(BaseT::getVectorInstrCost(Opcode, ValTy, Index), 2U);
  }

  if (ST->hasMVEIntegerOps() && (Opcode == Instruction::InsertElement ||
                                 Opcode == Instruction::ExtractElement)) {
    // We say MVE moves costs at least the MVEVectorCostFactor, even though
    // they are scalar instructions. This helps prevent mixing scalar and
    // vector, to prevent vectorising where we end up just scalarising the
    // result anyway.
    return std::max(BaseT::getVectorInstrCost(Opcode, ValTy, Index),
                    ST->getMVEVectorCostFactor()) *
           cast<FixedVectorType>(ValTy)->getNumElements() / 2;
  }

  return BaseT::getVectorInstrCost(Opcode, ValTy, Index);
}

int ARMTTIImpl::getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy,
                                   TTI::TargetCostKind CostKind,
                                   const Instruction *I) {
  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  // On NEON a vector select gets lowered to vbsl.
  if (ST->hasNEON() && ValTy->isVectorTy() && ISD == ISD::SELECT) {
    // Lowering of some vector selects is currently far from perfect.
    static const TypeConversionCostTblEntry NEONVectorSelectTbl[] = {
      { ISD::SELECT, MVT::v4i1, MVT::v4i64, 4*4 + 1*2 + 1 },
      { ISD::SELECT, MVT::v8i1, MVT::v8i64, 50 },
      { ISD::SELECT, MVT::v16i1, MVT::v16i64, 100 }
    };

    EVT SelCondTy = TLI->getValueType(DL, CondTy);
    EVT SelValTy = TLI->getValueType(DL, ValTy);
    if (SelCondTy.isSimple() && SelValTy.isSimple()) {
      if (const auto *Entry = ConvertCostTableLookup(NEONVectorSelectTbl, ISD,
                                                     SelCondTy.getSimpleVT(),
                                                     SelValTy.getSimpleVT()))
        return Entry->Cost;
    }

    std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, ValTy);
    return LT.first;
  }

  int BaseCost = ST->hasMVEIntegerOps() && ValTy->isVectorTy()
                     ? ST->getMVEVectorCostFactor()
                     : 1;
  return BaseCost * BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, CostKind,
                                              I);
}

int ARMTTIImpl::getAddressComputationCost(Type *Ty, ScalarEvolution *SE,
                                          const SCEV *Ptr) {
  // Address computations in vectorized code with non-consecutive addresses will
  // likely result in more instructions compared to scalar code where the
  // computation can more often be merged into the index mode. The resulting
  // extra micro-ops can significantly decrease throughput.
  unsigned NumVectorInstToHideOverhead = 10;
  int MaxMergeDistance = 64;

  if (ST->hasNEON()) {
    if (Ty->isVectorTy() && SE &&
        !BaseT::isConstantStridedAccessLessThan(SE, Ptr, MaxMergeDistance + 1))
      return NumVectorInstToHideOverhead;

    // In many cases the address computation is not merged into the instruction
    // addressing mode.
    return 1;
  }
  return BaseT::getAddressComputationCost(Ty, SE, Ptr);
}

bool ARMTTIImpl::isProfitableLSRChainElement(Instruction *I) {
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    // If a VCTP is part of a chain, it's already profitable and shouldn't be
    // optimized, else LSR may block tail-predication.
    switch (II->getIntrinsicID()) {
    case Intrinsic::arm_mve_vctp8:
    case Intrinsic::arm_mve_vctp16:
    case Intrinsic::arm_mve_vctp32:
    case Intrinsic::arm_mve_vctp64:
      return true;
    default:
      break;
    }
  }
  return false;
}

bool ARMTTIImpl::isLegalMaskedLoad(Type *DataTy, MaybeAlign Alignment) {
  if (!EnableMaskedLoadStores || !ST->hasMVEIntegerOps())
    return false;

  if (auto *VecTy = dyn_cast<FixedVectorType>(DataTy)) {
    // Don't support v2i1 yet.
    if (VecTy->getNumElements() == 2)
      return false;

    // We don't support extending fp types.
     unsigned VecWidth = DataTy->getPrimitiveSizeInBits();
    if (VecWidth != 128 && VecTy->getElementType()->isFloatingPointTy())
      return false;
  }

  unsigned EltWidth = DataTy->getScalarSizeInBits();
  return (EltWidth == 32 && (!Alignment || Alignment >= 4)) ||
         (EltWidth == 16 && (!Alignment || Alignment >= 2)) ||
         (EltWidth == 8);
}

bool ARMTTIImpl::isLegalMaskedGather(Type *Ty, MaybeAlign Alignment) {
  if (!EnableMaskedGatherScatters || !ST->hasMVEIntegerOps())
    return false;

  // This method is called in 2 places:
  //  - from the vectorizer with a scalar type, in which case we need to get
  //  this as good as we can with the limited info we have (and rely on the cost
  //  model for the rest).
  //  - from the masked intrinsic lowering pass with the actual vector type.
  // For MVE, we have a custom lowering pass that will already have custom
  // legalised any gathers that we can to MVE intrinsics, and want to expand all
  // the rest. The pass runs before the masked intrinsic lowering pass, so if we
  // are here, we know we want to expand.
  if (isa<VectorType>(Ty))
    return false;

  unsigned EltWidth = Ty->getScalarSizeInBits();
  return ((EltWidth == 32 && (!Alignment || Alignment >= 4)) ||
          (EltWidth == 16 && (!Alignment || Alignment >= 2)) || EltWidth == 8);
}

int ARMTTIImpl::getMemcpyCost(const Instruction *I) {
  const MemCpyInst *MI = dyn_cast<MemCpyInst>(I);
  assert(MI && "MemcpyInst expected");
  ConstantInt *C = dyn_cast<ConstantInt>(MI->getLength());

  // To model the cost of a library call, we assume 1 for the call, and
  // 3 for the argument setup.
  const unsigned LibCallCost = 4;

  // If 'size' is not a constant, a library call will be generated.
  if (!C)
    return LibCallCost;

  const unsigned Size = C->getValue().getZExtValue();
  const Align DstAlign = *MI->getDestAlign();
  const Align SrcAlign = *MI->getSourceAlign();
  const Function *F = I->getParent()->getParent();
  const unsigned Limit = TLI->getMaxStoresPerMemmove(F->hasMinSize());
  std::vector<EVT> MemOps;

  // MemOps will be poplulated with a list of data types that needs to be
  // loaded and stored. That's why we multiply the number of elements by 2 to
  // get the cost for this memcpy.
  if (getTLI()->findOptimalMemOpLowering(
          MemOps, Limit,
          MemOp::Copy(Size, /*DstAlignCanChange*/ false, DstAlign, SrcAlign,
                      /*IsVolatile*/ true),
          MI->getDestAddressSpace(), MI->getSourceAddressSpace(),
          F->getAttributes()))
    return MemOps.size() * 2;

  // If we can't find an optimal memop lowering, return the default cost
  return LibCallCost;
}

int ARMTTIImpl::getShuffleCost(TTI::ShuffleKind Kind, VectorType *Tp,
                               int Index, VectorType *SubTp) {
  if (ST->hasNEON()) {
    if (Kind == TTI::SK_Broadcast) {
      static const CostTblEntry NEONDupTbl[] = {
          // VDUP handles these cases.
          {ISD::VECTOR_SHUFFLE, MVT::v2i32, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v2f32, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v2i64, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v2f64, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v4i16, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v8i8, 1},

          {ISD::VECTOR_SHUFFLE, MVT::v4i32, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v4f32, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v8i16, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v16i8, 1}};

      std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, Tp);

      if (const auto *Entry =
              CostTableLookup(NEONDupTbl, ISD::VECTOR_SHUFFLE, LT.second))
        return LT.first * Entry->Cost;
    }
    if (Kind == TTI::SK_Reverse) {
      static const CostTblEntry NEONShuffleTbl[] = {
          // Reverse shuffle cost one instruction if we are shuffling within a
          // double word (vrev) or two if we shuffle a quad word (vrev, vext).
          {ISD::VECTOR_SHUFFLE, MVT::v2i32, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v2f32, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v2i64, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v2f64, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v4i16, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v8i8, 1},

          {ISD::VECTOR_SHUFFLE, MVT::v4i32, 2},
          {ISD::VECTOR_SHUFFLE, MVT::v4f32, 2},
          {ISD::VECTOR_SHUFFLE, MVT::v8i16, 2},
          {ISD::VECTOR_SHUFFLE, MVT::v16i8, 2}};

      std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, Tp);

      if (const auto *Entry =
              CostTableLookup(NEONShuffleTbl, ISD::VECTOR_SHUFFLE, LT.second))
        return LT.first * Entry->Cost;
    }
    if (Kind == TTI::SK_Select) {
      static const CostTblEntry NEONSelShuffleTbl[] = {
          // Select shuffle cost table for ARM. Cost is the number of
          // instructions
          // required to create the shuffled vector.

          {ISD::VECTOR_SHUFFLE, MVT::v2f32, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v2i64, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v2f64, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v2i32, 1},

          {ISD::VECTOR_SHUFFLE, MVT::v4i32, 2},
          {ISD::VECTOR_SHUFFLE, MVT::v4f32, 2},
          {ISD::VECTOR_SHUFFLE, MVT::v4i16, 2},

          {ISD::VECTOR_SHUFFLE, MVT::v8i16, 16},

          {ISD::VECTOR_SHUFFLE, MVT::v16i8, 32}};

      std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, Tp);
      if (const auto *Entry = CostTableLookup(NEONSelShuffleTbl,
                                              ISD::VECTOR_SHUFFLE, LT.second))
        return LT.first * Entry->Cost;
    }
  }
  if (ST->hasMVEIntegerOps()) {
    if (Kind == TTI::SK_Broadcast) {
      static const CostTblEntry MVEDupTbl[] = {
          // VDUP handles these cases.
          {ISD::VECTOR_SHUFFLE, MVT::v4i32, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v8i16, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v16i8, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v4f32, 1},
          {ISD::VECTOR_SHUFFLE, MVT::v8f16, 1}};

      std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, Tp);

      if (const auto *Entry = CostTableLookup(MVEDupTbl, ISD::VECTOR_SHUFFLE,
                                              LT.second))
        return LT.first * Entry->Cost * ST->getMVEVectorCostFactor();
    }
  }
  int BaseCost = ST->hasMVEIntegerOps() && Tp->isVectorTy()
                     ? ST->getMVEVectorCostFactor()
                     : 1;
  return BaseCost * BaseT::getShuffleCost(Kind, Tp, Index, SubTp);
}

int ARMTTIImpl::getArithmeticInstrCost(unsigned Opcode, Type *Ty,
                                       TTI::TargetCostKind CostKind,
                                       TTI::OperandValueKind Op1Info,
                                       TTI::OperandValueKind Op2Info,
                                       TTI::OperandValueProperties Opd1PropInfo,
                                       TTI::OperandValueProperties Opd2PropInfo,
                                       ArrayRef<const Value *> Args,
                                       const Instruction *CxtI) {
  int ISDOpcode = TLI->InstructionOpcodeToISD(Opcode);
  std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, Ty);

  if (ST->hasNEON()) {
    const unsigned FunctionCallDivCost = 20;
    const unsigned ReciprocalDivCost = 10;
    static const CostTblEntry CostTbl[] = {
      // Division.
      // These costs are somewhat random. Choose a cost of 20 to indicate that
      // vectorizing devision (added function call) is going to be very expensive.
      // Double registers types.
      { ISD::SDIV, MVT::v1i64, 1 * FunctionCallDivCost},
      { ISD::UDIV, MVT::v1i64, 1 * FunctionCallDivCost},
      { ISD::SREM, MVT::v1i64, 1 * FunctionCallDivCost},
      { ISD::UREM, MVT::v1i64, 1 * FunctionCallDivCost},
      { ISD::SDIV, MVT::v2i32, 2 * FunctionCallDivCost},
      { ISD::UDIV, MVT::v2i32, 2 * FunctionCallDivCost},
      { ISD::SREM, MVT::v2i32, 2 * FunctionCallDivCost},
      { ISD::UREM, MVT::v2i32, 2 * FunctionCallDivCost},
      { ISD::SDIV, MVT::v4i16,     ReciprocalDivCost},
      { ISD::UDIV, MVT::v4i16,     ReciprocalDivCost},
      { ISD::SREM, MVT::v4i16, 4 * FunctionCallDivCost},
      { ISD::UREM, MVT::v4i16, 4 * FunctionCallDivCost},
      { ISD::SDIV, MVT::v8i8,      ReciprocalDivCost},
      { ISD::UDIV, MVT::v8i8,      ReciprocalDivCost},
      { ISD::SREM, MVT::v8i8,  8 * FunctionCallDivCost},
      { ISD::UREM, MVT::v8i8,  8 * FunctionCallDivCost},
      // Quad register types.
      { ISD::SDIV, MVT::v2i64, 2 * FunctionCallDivCost},
      { ISD::UDIV, MVT::v2i64, 2 * FunctionCallDivCost},
      { ISD::SREM, MVT::v2i64, 2 * FunctionCallDivCost},
      { ISD::UREM, MVT::v2i64, 2 * FunctionCallDivCost},
      { ISD::SDIV, MVT::v4i32, 4 * FunctionCallDivCost},
      { ISD::UDIV, MVT::v4i32, 4 * FunctionCallDivCost},
      { ISD::SREM, MVT::v4i32, 4 * FunctionCallDivCost},
      { ISD::UREM, MVT::v4i32, 4 * FunctionCallDivCost},
      { ISD::SDIV, MVT::v8i16, 8 * FunctionCallDivCost},
      { ISD::UDIV, MVT::v8i16, 8 * FunctionCallDivCost},
      { ISD::SREM, MVT::v8i16, 8 * FunctionCallDivCost},
      { ISD::UREM, MVT::v8i16, 8 * FunctionCallDivCost},
      { ISD::SDIV, MVT::v16i8, 16 * FunctionCallDivCost},
      { ISD::UDIV, MVT::v16i8, 16 * FunctionCallDivCost},
      { ISD::SREM, MVT::v16i8, 16 * FunctionCallDivCost},
      { ISD::UREM, MVT::v16i8, 16 * FunctionCallDivCost},
      // Multiplication.
    };

    if (const auto *Entry = CostTableLookup(CostTbl, ISDOpcode, LT.second))
      return LT.first * Entry->Cost;

    int Cost = BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Op1Info,
                                             Op2Info,
                                             Opd1PropInfo, Opd2PropInfo);

    // This is somewhat of a hack. The problem that we are facing is that SROA
    // creates a sequence of shift, and, or instructions to construct values.
    // These sequences are recognized by the ISel and have zero-cost. Not so for
    // the vectorized code. Because we have support for v2i64 but not i64 those
    // sequences look particularly beneficial to vectorize.
    // To work around this we increase the cost of v2i64 operations to make them
    // seem less beneficial.
    if (LT.second == MVT::v2i64 &&
        Op2Info == TargetTransformInfo::OK_UniformConstantValue)
      Cost += 4;

    return Cost;
  }

  // If this operation is a shift on arm/thumb2, it might well be folded into
  // the following instruction, hence having a cost of 0.
  auto LooksLikeAFreeShift = [&]() {
    if (ST->isThumb1Only() || Ty->isVectorTy())
      return false;

    if (!CxtI || !CxtI->hasOneUse() || !CxtI->isShift())
      return false;
    if (Op2Info != TargetTransformInfo::OK_UniformConstantValue)
      return false;

    // Folded into a ADC/ADD/AND/BIC/CMP/EOR/MVN/ORR/ORN/RSB/SBC/SUB
    switch (cast<Instruction>(CxtI->user_back())->getOpcode()) {
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::And:
    case Instruction::Xor:
    case Instruction::Or:
    case Instruction::ICmp:
      return true;
    default:
      return false;
    }
  };
  if (LooksLikeAFreeShift())
    return 0;

  int BaseCost = ST->hasMVEIntegerOps() && Ty->isVectorTy()
                     ? ST->getMVEVectorCostFactor()
                     : 1;

  // The rest of this mostly follows what is done in BaseT::getArithmeticInstrCost,
  // without treating floats as more expensive that scalars or increasing the
  // costs for custom operations. The results is also multiplied by the
  // MVEVectorCostFactor where appropriate.
  if (TLI->isOperationLegalOrCustomOrPromote(ISDOpcode, LT.second))
    return LT.first * BaseCost;

  // Else this is expand, assume that we need to scalarize this op.
  if (auto *VTy = dyn_cast<FixedVectorType>(Ty)) {
    unsigned Num = VTy->getNumElements();
    unsigned Cost = getArithmeticInstrCost(Opcode, Ty->getScalarType(),
                                           CostKind);
    // Return the cost of multiple scalar invocation plus the cost of
    // inserting and extracting the values.
    return BaseT::getScalarizationOverhead(VTy, Args) + Num * Cost;
  }

  return BaseCost;
}

int ARMTTIImpl::getMemoryOpCost(unsigned Opcode, Type *Src,
                                MaybeAlign Alignment, unsigned AddressSpace,
                                TTI::TargetCostKind CostKind,
                                const Instruction *I) {
  std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, Src);

  if (ST->hasNEON() && Src->isVectorTy() &&
      (Alignment && *Alignment != Align(16)) &&
      cast<VectorType>(Src)->getElementType()->isDoubleTy()) {
    // Unaligned loads/stores are extremely inefficient.
    // We need 4 uops for vst.1/vld.1 vs 1uop for vldr/vstr.
    return LT.first * 4;
  }
  int BaseCost = ST->hasMVEIntegerOps() && Src->isVectorTy()
                     ? ST->getMVEVectorCostFactor()
                     : 1;
  return BaseCost * LT.first;
}

int ARMTTIImpl::getInterleavedMemoryOpCost(
    unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
    unsigned Alignment, unsigned AddressSpace,
    TTI::TargetCostKind CostKind,
    bool UseMaskForCond, bool UseMaskForGaps) {
  assert(Factor >= 2 && "Invalid interleave factor");
  assert(isa<VectorType>(VecTy) && "Expect a vector type");

  // vldN/vstN doesn't support vector types of i64/f64 element.
  bool EltIs64Bits = DL.getTypeSizeInBits(VecTy->getScalarType()) == 64;

  if (Factor <= TLI->getMaxSupportedInterleaveFactor() && !EltIs64Bits &&
      !UseMaskForCond && !UseMaskForGaps) {
    unsigned NumElts = cast<FixedVectorType>(VecTy)->getNumElements();
    auto *SubVecTy =
        FixedVectorType::get(VecTy->getScalarType(), NumElts / Factor);

    // vldN/vstN only support legal vector types of size 64 or 128 in bits.
    // Accesses having vector types that are a multiple of 128 bits can be
    // matched to more than one vldN/vstN instruction.
    int BaseCost = ST->hasMVEIntegerOps() ? ST->getMVEVectorCostFactor() : 1;
    if (NumElts % Factor == 0 &&
        TLI->isLegalInterleavedAccessType(Factor, SubVecTy, DL))
      return Factor * BaseCost * TLI->getNumInterleavedAccesses(SubVecTy, DL);

    // Some smaller than legal interleaved patterns are cheap as we can make
    // use of the vmovn or vrev patterns to interleave a standard load. This is
    // true for v4i8, v8i8 and v4i16 at least (but not for v4f16 as it is
    // promoted differently). The cost of 2 here is then a load and vrev or
    // vmovn.
    if (ST->hasMVEIntegerOps() && Factor == 2 && NumElts / Factor > 2 &&
        VecTy->isIntOrIntVectorTy() && DL.getTypeSizeInBits(SubVecTy) <= 64)
      return 2 * BaseCost;
  }

  return BaseT::getInterleavedMemoryOpCost(Opcode, VecTy, Factor, Indices,
                                           Alignment, AddressSpace, CostKind,
                                           UseMaskForCond, UseMaskForGaps);
}

unsigned ARMTTIImpl::getGatherScatterOpCost(unsigned Opcode, Type *DataTy,
                                            Value *Ptr, bool VariableMask,
                                            unsigned Alignment,
                                            TTI::TargetCostKind CostKind,
                                            const Instruction *I) {
  using namespace PatternMatch;
  if (!ST->hasMVEIntegerOps() || !EnableMaskedGatherScatters)
    return BaseT::getGatherScatterOpCost(Opcode, DataTy, Ptr, VariableMask,
                                         Alignment, CostKind, I);

  assert(DataTy->isVectorTy() && "Can't do gather/scatters on scalar!");
  auto *VTy = cast<FixedVectorType>(DataTy);

  // TODO: Splitting, once we do that.

  unsigned NumElems = VTy->getNumElements();
  unsigned EltSize = VTy->getScalarSizeInBits();
  std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, DataTy);

  // For now, it is assumed that for the MVE gather instructions the loads are
  // all effectively serialised. This means the cost is the scalar cost
  // multiplied by the number of elements being loaded. This is possibly very
  // conservative, but even so we still end up vectorising loops because the
  // cost per iteration for many loops is lower than for scalar loops.
  unsigned VectorCost = NumElems * LT.first;
  // The scalarization cost should be a lot higher. We use the number of vector
  // elements plus the scalarization overhead.
  unsigned ScalarCost =
      NumElems * LT.first + BaseT::getScalarizationOverhead(VTy, {});

  if (Alignment < EltSize / 8)
    return ScalarCost;

  unsigned ExtSize = EltSize;
  // Check whether there's a single user that asks for an extended type
  if (I != nullptr) {
    // Dependent of the caller of this function, a gather instruction will
    // either have opcode Instruction::Load or be a call to the masked_gather
    // intrinsic
    if ((I->getOpcode() == Instruction::Load ||
         match(I, m_Intrinsic<Intrinsic::masked_gather>())) &&
        I->hasOneUse()) {
      const User *Us = *I->users().begin();
      if (isa<ZExtInst>(Us) || isa<SExtInst>(Us)) {
        // only allow valid type combinations
        unsigned TypeSize =
            cast<Instruction>(Us)->getType()->getScalarSizeInBits();
        if (((TypeSize == 32 && (EltSize == 8 || EltSize == 16)) ||
             (TypeSize == 16 && EltSize == 8)) &&
            TypeSize * NumElems == 128) {
          ExtSize = TypeSize;
        }
      }
    }
    // Check whether the input data needs to be truncated
    TruncInst *T;
    if ((I->getOpcode() == Instruction::Store ||
         match(I, m_Intrinsic<Intrinsic::masked_scatter>())) &&
        (T = dyn_cast<TruncInst>(I->getOperand(0)))) {
      // Only allow valid type combinations
      unsigned TypeSize = T->getOperand(0)->getType()->getScalarSizeInBits();
      if (((EltSize == 16 && TypeSize == 32) ||
           (EltSize == 8 && (TypeSize == 32 || TypeSize == 16))) &&
          TypeSize * NumElems == 128)
        ExtSize = TypeSize;
    }
  }

  if (ExtSize * NumElems != 128 || NumElems < 4)
    return ScalarCost;

  // Any (aligned) i32 gather will not need to be scalarised.
  if (ExtSize == 32)
    return VectorCost;
  // For smaller types, we need to ensure that the gep's inputs are correctly
  // extended from a small enough value. Other sizes (including i64) are
  // scalarized for now.
  if (ExtSize != 8 && ExtSize != 16)
    return ScalarCost;

  if (auto BC = dyn_cast<BitCastInst>(Ptr))
    Ptr = BC->getOperand(0);
  if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
    if (GEP->getNumOperands() != 2)
      return ScalarCost;
    unsigned Scale = DL.getTypeAllocSize(GEP->getResultElementType());
    // Scale needs to be correct (which is only relevant for i16s).
    if (Scale != 1 && Scale * 8 != ExtSize)
      return ScalarCost;
    // And we need to zext (not sext) the indexes from a small enough type.
    if (auto ZExt = dyn_cast<ZExtInst>(GEP->getOperand(1))) {
      if (ZExt->getOperand(0)->getType()->getScalarSizeInBits() <= ExtSize)
        return VectorCost;
    }
    return ScalarCost;
  }
  return ScalarCost;
}

bool ARMTTIImpl::isLoweredToCall(const Function *F) {
  if (!F->isIntrinsic())
    BaseT::isLoweredToCall(F);

  // Assume all Arm-specific intrinsics map to an instruction.
  if (F->getName().startswith("llvm.arm"))
    return false;

  switch (F->getIntrinsicID()) {
  default: break;
  case Intrinsic::powi:
  case Intrinsic::sin:
  case Intrinsic::cos:
  case Intrinsic::pow:
  case Intrinsic::log:
  case Intrinsic::log10:
  case Intrinsic::log2:
  case Intrinsic::exp:
  case Intrinsic::exp2:
    return true;
  case Intrinsic::sqrt:
  case Intrinsic::fabs:
  case Intrinsic::copysign:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::round:
  case Intrinsic::canonicalize:
  case Intrinsic::lround:
  case Intrinsic::llround:
  case Intrinsic::lrint:
  case Intrinsic::llrint:
    if (F->getReturnType()->isDoubleTy() && !ST->hasFP64())
      return true;
    if (F->getReturnType()->isHalfTy() && !ST->hasFullFP16())
      return true;
    // Some operations can be handled by vector instructions and assume
    // unsupported vectors will be expanded into supported scalar ones.
    // TODO Handle scalar operations properly.
    return !ST->hasFPARMv8Base() && !ST->hasVFP2Base();
  case Intrinsic::masked_store:
  case Intrinsic::masked_load:
  case Intrinsic::masked_gather:
  case Intrinsic::masked_scatter:
    return !ST->hasMVEIntegerOps();
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::uadd_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::usub_with_overflow:
  case Intrinsic::sadd_sat:
  case Intrinsic::uadd_sat:
  case Intrinsic::ssub_sat:
  case Intrinsic::usub_sat:
    return false;
  }

  return BaseT::isLoweredToCall(F);
}

bool ARMTTIImpl::isHardwareLoopProfitable(Loop *L, ScalarEvolution &SE,
                                          AssumptionCache &AC,
                                          TargetLibraryInfo *LibInfo,
                                          HardwareLoopInfo &HWLoopInfo) {
  // Low-overhead branches are only supported in the 'low-overhead branch'
  // extension of v8.1-m.
  if (!ST->hasLOB() || DisableLowOverheadLoops) {
    LLVM_DEBUG(dbgs() << "ARMHWLoops: Disabled\n");
    return false;
  }

  if (!SE.hasLoopInvariantBackedgeTakenCount(L)) {
    LLVM_DEBUG(dbgs() << "ARMHWLoops: No BETC\n");
    return false;
  }

  const SCEV *BackedgeTakenCount = SE.getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BackedgeTakenCount)) {
    LLVM_DEBUG(dbgs() << "ARMHWLoops: Uncomputable BETC\n");
    return false;
  }

  const SCEV *TripCountSCEV =
    SE.getAddExpr(BackedgeTakenCount,
                  SE.getOne(BackedgeTakenCount->getType()));

  // We need to store the trip count in LR, a 32-bit register.
  if (SE.getUnsignedRangeMax(TripCountSCEV).getBitWidth() > 32) {
    LLVM_DEBUG(dbgs() << "ARMHWLoops: Trip count does not fit into 32bits\n");
    return false;
  }

  // Making a call will trash LR and clear LO_BRANCH_INFO, so there's little
  // point in generating a hardware loop if that's going to happen.
  auto MaybeCall = [this](Instruction &I) {
    const ARMTargetLowering *TLI = getTLI();
    unsigned ISD = TLI->InstructionOpcodeToISD(I.getOpcode());
    EVT VT = TLI->getValueType(DL, I.getType(), true);
    if (TLI->getOperationAction(ISD, VT) == TargetLowering::LibCall)
      return true;

    // Check if an intrinsic will be lowered to a call and assume that any
    // other CallInst will generate a bl.
    if (auto *Call = dyn_cast<CallInst>(&I)) {
      if (isa<IntrinsicInst>(Call)) {
        if (const Function *F = Call->getCalledFunction())
          return isLoweredToCall(F);
      }
      return true;
    }

    // FPv5 provides conversions between integer, double-precision,
    // single-precision, and half-precision formats.
    switch (I.getOpcode()) {
    default:
      break;
    case Instruction::FPToSI:
    case Instruction::FPToUI:
    case Instruction::SIToFP:
    case Instruction::UIToFP:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
      return !ST->hasFPARMv8Base();
    }

    // FIXME: Unfortunately the approach of checking the Operation Action does
    // not catch all cases of Legalization that use library calls. Our
    // Legalization step categorizes some transformations into library calls as
    // Custom, Expand or even Legal when doing type legalization. So for now
    // we have to special case for instance the SDIV of 64bit integers and the
    // use of floating point emulation.
    if (VT.isInteger() && VT.getSizeInBits() >= 64) {
      switch (ISD) {
      default:
        break;
      case ISD::SDIV:
      case ISD::UDIV:
      case ISD::SREM:
      case ISD::UREM:
      case ISD::SDIVREM:
      case ISD::UDIVREM:
        return true;
      }
    }

    // Assume all other non-float operations are supported.
    if (!VT.isFloatingPoint())
      return false;

    // We'll need a library call to handle most floats when using soft.
    if (TLI->useSoftFloat()) {
      switch (I.getOpcode()) {
      default:
        return true;
      case Instruction::Alloca:
      case Instruction::Load:
      case Instruction::Store:
      case Instruction::Select:
      case Instruction::PHI:
        return false;
      }
    }

    // We'll need a libcall to perform double precision operations on a single
    // precision only FPU.
    if (I.getType()->isDoubleTy() && !ST->hasFP64())
      return true;

    // Likewise for half precision arithmetic.
    if (I.getType()->isHalfTy() && !ST->hasFullFP16())
      return true;

    return false;
  };

  auto IsHardwareLoopIntrinsic = [](Instruction &I) {
    if (auto *Call = dyn_cast<IntrinsicInst>(&I)) {
      switch (Call->getIntrinsicID()) {
      default:
        break;
      case Intrinsic::set_loop_iterations:
      case Intrinsic::test_set_loop_iterations:
      case Intrinsic::loop_decrement:
      case Intrinsic::loop_decrement_reg:
        return true;
      }
    }
    return false;
  };

  // Scan the instructions to see if there's any that we know will turn into a
  // call or if this loop is already a low-overhead loop.
  auto ScanLoop = [&](Loop *L) {
    for (auto *BB : L->getBlocks()) {
      for (auto &I : *BB) {
        if (MaybeCall(I) || IsHardwareLoopIntrinsic(I)) {
          LLVM_DEBUG(dbgs() << "ARMHWLoops: Bad instruction: " << I << "\n");
          return false;
        }
      }
    }
    return true;
  };

  // Visit inner loops.
  for (auto Inner : *L)
    if (!ScanLoop(Inner))
      return false;

  if (!ScanLoop(L))
    return false;

  // TODO: Check whether the trip count calculation is expensive. If L is the
  // inner loop but we know it has a low trip count, calculating that trip
  // count (in the parent loop) may be detrimental.

  LLVMContext &C = L->getHeader()->getContext();
  HWLoopInfo.CounterInReg = true;
  HWLoopInfo.IsNestingLegal = false;
  HWLoopInfo.PerformEntryTest = true;
  HWLoopInfo.CountType = Type::getInt32Ty(C);
  HWLoopInfo.LoopDecrement = ConstantInt::get(HWLoopInfo.CountType, 1);
  return true;
}

static bool canTailPredicateInstruction(Instruction &I, int &ICmpCount) {
  // We don't allow icmp's, and because we only look at single block loops,
  // we simply count the icmps, i.e. there should only be 1 for the backedge.
  if (isa<ICmpInst>(&I) && ++ICmpCount > 1)
    return false;

  if (isa<FCmpInst>(&I))
    return false;

  // We could allow extending/narrowing FP loads/stores, but codegen is
  // too inefficient so reject this for now.
  if (isa<FPExtInst>(&I) || isa<FPTruncInst>(&I))
    return false;

  // Extends have to be extending-loads
  if (isa<SExtInst>(&I) || isa<ZExtInst>(&I) )
    if (!I.getOperand(0)->hasOneUse() || !isa<LoadInst>(I.getOperand(0)))
      return false;

  // Truncs have to be narrowing-stores
  if (isa<TruncInst>(&I) )
    if (!I.hasOneUse() || !isa<StoreInst>(*I.user_begin()))
      return false;

  return true;
}

// To set up a tail-predicated loop, we need to know the total number of
// elements processed by that loop. Thus, we need to determine the element
// size and:
// 1) it should be uniform for all operations in the vector loop, so we
//    e.g. don't want any widening/narrowing operations.
// 2) it should be smaller than i64s because we don't have vector operations
//    that work on i64s.
// 3) we don't want elements to be reversed or shuffled, to make sure the
//    tail-predication masks/predicates the right lanes.
//
static bool canTailPredicateLoop(Loop *L, LoopInfo *LI, ScalarEvolution &SE,
                                 const DataLayout &DL,
                                 const LoopAccessInfo *LAI) {
  PredicatedScalarEvolution PSE = LAI->getPSE();
  int ICmpCount = 0;
  int Stride = 0;

  LLVM_DEBUG(dbgs() << "tail-predication: checking allowed instructions\n");
  SmallVector<Instruction *, 16> LoadStores;
  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : BB->instructionsWithoutDebug()) {
      if (isa<PHINode>(&I))
        continue;
      if (!canTailPredicateInstruction(I, ICmpCount)) {
        LLVM_DEBUG(dbgs() << "Instruction not allowed: "; I.dump());
        return false;
      }

      Type *T  = I.getType();
      if (T->isPointerTy())
        T = T->getPointerElementType();

      if (T->getScalarSizeInBits() > 32) {
        LLVM_DEBUG(dbgs() << "Unsupported Type: "; T->dump());
        return false;
      }

      if (isa<StoreInst>(I) || isa<LoadInst>(I)) {
        Value *Ptr = isa<LoadInst>(I) ? I.getOperand(0) : I.getOperand(1);
        int64_t NextStride = getPtrStride(PSE, Ptr, L);
        // TODO: for now only allow consecutive strides of 1. We could support
        // other strides as long as it is uniform, but let's keep it simple for
        // now.
        if (Stride == 0 && NextStride == 1) {
          Stride = NextStride;
          continue;
        }
        if (Stride != NextStride) {
          LLVM_DEBUG(dbgs() << "Different strides found, can't "
                               "tail-predicate\n.");
          return false;
        }
      }
    }
  }

  LLVM_DEBUG(dbgs() << "tail-predication: all instructions allowed!\n");
  return true;
}

bool ARMTTIImpl::preferPredicateOverEpilogue(Loop *L, LoopInfo *LI,
                                             ScalarEvolution &SE,
                                             AssumptionCache &AC,
                                             TargetLibraryInfo *TLI,
                                             DominatorTree *DT,
                                             const LoopAccessInfo *LAI) {
  if (DisableTailPredication)
    return false;

  // Creating a predicated vector loop is the first step for generating a
  // tail-predicated hardware loop, for which we need the MVE masked
  // load/stores instructions:
  if (!ST->hasMVEIntegerOps())
    return false;

  // For now, restrict this to single block loops.
  if (L->getNumBlocks() > 1) {
    LLVM_DEBUG(dbgs() << "preferPredicateOverEpilogue: not a single block "
                         "loop.\n");
    return false;
  }

  assert(L->empty() && "preferPredicateOverEpilogue: inner-loop expected");

  HardwareLoopInfo HWLoopInfo(L);
  if (!HWLoopInfo.canAnalyze(*LI)) {
    LLVM_DEBUG(dbgs() << "preferPredicateOverEpilogue: hardware-loop is not "
                         "analyzable.\n");
    return false;
  }

  // This checks if we have the low-overhead branch architecture
  // extension, and if we will create a hardware-loop:
  if (!isHardwareLoopProfitable(L, SE, AC, TLI, HWLoopInfo)) {
    LLVM_DEBUG(dbgs() << "preferPredicateOverEpilogue: hardware-loop is not "
                         "profitable.\n");
    return false;
  }

  if (!HWLoopInfo.isHardwareLoopCandidate(SE, *LI, *DT)) {
    LLVM_DEBUG(dbgs() << "preferPredicateOverEpilogue: hardware-loop is not "
                         "a candidate.\n");
    return false;
  }

  return canTailPredicateLoop(L, LI, SE, DL, LAI);
}


void ARMTTIImpl::getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                                         TTI::UnrollingPreferences &UP) {
  // Only currently enable these preferences for M-Class cores.
  if (!ST->isMClass())
    return BasicTTIImplBase::getUnrollingPreferences(L, SE, UP);

  // Disable loop unrolling for Oz and Os.
  UP.OptSizeThreshold = 0;
  UP.PartialOptSizeThreshold = 0;
  if (L->getHeader()->getParent()->hasOptSize())
    return;

  // Only enable on Thumb-2 targets.
  if (!ST->isThumb2())
    return;

  SmallVector<BasicBlock*, 4> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);
  LLVM_DEBUG(dbgs() << "Loop has:\n"
                    << "Blocks: " << L->getNumBlocks() << "\n"
                    << "Exit blocks: " << ExitingBlocks.size() << "\n");

  // Only allow another exit other than the latch. This acts as an early exit
  // as it mirrors the profitability calculation of the runtime unroller.
  if (ExitingBlocks.size() > 2)
    return;

  // Limit the CFG of the loop body for targets with a branch predictor.
  // Allowing 4 blocks permits if-then-else diamonds in the body.
  if (ST->hasBranchPredictor() && L->getNumBlocks() > 4)
    return;

  // Scan the loop: don't unroll loops with calls as this could prevent
  // inlining.
  unsigned Cost = 0;
  for (auto *BB : L->getBlocks()) {
    for (auto &I : *BB) {
      // Don't unroll vectorised loop. MVE does not benefit from it as much as
      // scalar code.
      if (I.getType()->isVectorTy())
        return;

      if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
        if (const Function *F = cast<CallBase>(I).getCalledFunction()) {
          if (!isLoweredToCall(F))
            continue;
        }
        return;
      }

      SmallVector<const Value*, 4> Operands(I.value_op_begin(),
                                            I.value_op_end());
      Cost += getUserCost(&I, Operands, TargetTransformInfo::TCK_CodeSize);
    }
  }

  LLVM_DEBUG(dbgs() << "Cost of loop: " << Cost << "\n");

  UP.Partial = true;
  UP.Runtime = true;
  UP.UpperBound = true;
  UP.UnrollRemainder = true;
  UP.DefaultUnrollRuntimeCount = 4;
  UP.UnrollAndJam = true;
  UP.UnrollAndJamInnerLoopThreshold = 60;

  // Force unrolling small loops can be very useful because of the branch
  // taken cost of the backedge.
  if (Cost < 12)
    UP.Force = true;
}

bool ARMTTIImpl::useReductionIntrinsic(unsigned Opcode, Type *Ty,
                                       TTI::ReductionFlags Flags) const {
  assert(isa<VectorType>(Ty) && "Expected Ty to be a vector type");
  unsigned ScalarBits = Ty->getScalarSizeInBits();
  if (!ST->hasMVEIntegerOps())
    return false;

  switch (Opcode) {
  case Instruction::FAdd:
  case Instruction::FMul:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Mul:
  case Instruction::FCmp:
    return false;
  case Instruction::ICmp:
  case Instruction::Add:
    return ScalarBits < 64 &&
           (ScalarBits * cast<FixedVectorType>(Ty)->getNumElements()) % 128 ==
               0;
  default:
    llvm_unreachable("Unhandled reduction opcode");
  }
  return false;
}
