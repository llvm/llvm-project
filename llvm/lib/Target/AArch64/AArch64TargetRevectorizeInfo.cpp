//===- AArch64TargetRevectorizeInfo.cpp - AArch64 TRVI --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64TargetRevectorizeInfo.h"
#include "AArch64Subtarget.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/VectorTypeUtils.h"
#include <optional>

using namespace llvm;

// Define the struct for the generated intrinsic mapping table
namespace {
struct NEONToSVEIntrinsicMapping {
  StringRef NEONIntrinsic;
  StringRef SVEIntrinsic;
  StringRef ArgMappings; // Comma-separated string: "T,Z,A:0,A:1"
  bool isCustom() const { return SVEIntrinsic.empty() || ArgMappings.empty(); }
};

/// Specifies how to build an argument for an SVE intrinsic, possibly from
/// another argument of an equivalent NEON intrinsic.
class ArgMapping {
public:
  enum MappingKind { True, Zero, Poison, Arg, Imm };

  ArgMapping(StringRef Spec) {
    if (Spec == "T") {
      Kind = True;
    } else if (Spec == "Z") {
      Kind = Zero;
    } else if (Spec == "P") {
      Kind = Poison;
    } else if (Spec.starts_with("A:")) {
      Kind = Arg;
      unsigned Idx;
      if (Spec.substr(2).getAsInteger(10, Idx))
        llvm_unreachable("Invalid argument index in mapping!");
      Index = Idx;
    } else if (Spec.starts_with("I:")) {
      Kind = Imm;
      unsigned Idx;
      if (Spec.substr(2).getAsInteger(10, Idx))
        llvm_unreachable("Invalid argument index in mapping!");
      Index = Idx;
    } else {
      llvm_unreachable("Unknown argument mapping kind!");
    }
  }

  MappingKind getKind() const { return Kind; }

  unsigned getArgIndex() const {
    assert((Kind == Arg || Kind == Imm) &&
           "getArgIndex called on non-Arg mapping");
    return *Index;
  }

private:
  MappingKind Kind;
  std::optional<unsigned> Index;
};

// Parse comma-separated argument mapping string into ArgMapping objects
static SmallVector<ArgMapping, 4> parseArgMappings(StringRef ArgMappingsStr) {
  SmallVector<ArgMapping, 4> Result;
  SmallVector<StringRef, 4> Specs;
  ArgMappingsStr.split(Specs, ',');
  for (StringRef Spec : Specs)
    Result.emplace_back(Spec);
  return Result;
}

/// Generate a segmented UZP shuffle that can later be selected to UZPQ.
/// \pre Src0 and Src1 are scalable vectors
Instruction *getSegmentedUZP(Value *Src0, Value *Src1, bool EvenElts,
                             IRBuilderBase &Builder) {
  assert(isa<ScalableVectorType>(Src0->getType()));
  auto *VTy = cast<ScalableVectorType>(Src0->getType());
  const unsigned SegmentSize = VTy->getMinNumElements();
  SmallVector<int, 8> Mask(SegmentSize);
  transform(seq<int>(SegmentSize), Mask.begin(),
            [EvenElts](int Idx) { return EvenElts ? Idx * 2 : Idx * 2 + 1; });
  return Builder.CreateSegmentedShuffleVector(Src0, Src1, Mask);
}

/// Concatenate the even words of \p Src with its odd words within each quad.
/// I.e. generate UZPQ1 Src.w,, (EXT Src, 4).w
Instruction *concatEvenThenOddWordsWithinQuads(Value *Src,
                                               IRBuilderBase &Builder) {
  assert(Src->getType()->getPrimitiveSizeInBits().getKnownMinValue() ==
             AArch64::SVEBitsPerBlock &&
         "Expected legal SVE vector.");
  assert(!Src->getType()->getScalarType()->isIntegerTy(32) &&
         "Unexpected nxv4i32 input");
  Type *OrigTy = Src->getType();
  Src = Builder.CreateBitCast(Src,
                              ScalableVectorType::get(Builder.getInt32Ty(), 4));
  Value *ShiftOddToEven =
      Builder.CreateVectorSpliceLeft(Src, PoisonValue::get(Src->getType()), 1);
  Instruction *Res =
      getSegmentedUZP(Src, ShiftOddToEven, /*EvenElts=*/true, Builder);
  return cast<Instruction>(Builder.CreateBitCast(Res, OrigTy));
}

SmallVector<Type *, 2> computeTysForDecl(Intrinsic::ID ID, Type *RetTy,
                                         ArrayRef<Value *> Args,
                                         const TargetTransformInfo &TTI) {
  SmallVector<Type *, 2> Tys;
  if (TTI.isTargetIntrinsicWithOverloadTypeAtArg(ID, -1))
    Tys.push_back(RetTy);
  for (auto [ArgIdx, Arg] : enumerate(Args))
    if (TTI.isTargetIntrinsicWithOverloadTypeAtArg(ID, ArgIdx))
      Tys.push_back(Arg->getType());
  return Tys;
}

/// A helper class to rewrite a widened NEON intrinsic to SVE.
class IntrinsicRewriter {
public:
  IntrinsicRewriter(ArrayRef<Type *> TysForDecl, ArrayRef<Value *> WideArgs,
                    IRBuilderBase &Builder)
      : TysForDecl(TysForDecl), WideArgs(WideArgs), Builder(Builder) {}

  /// Automatically rewrite a widened intrinsic using a \p
  /// NEONToSVEIntrinsicMapping.
  Instruction *rewriteWithMapping(const NEONToSVEIntrinsicMapping &Mapping,
                                  const TargetTransformInfo &TTI) {
    assert(!Mapping.isCustom() &&
           "NEON intrinsic was expected to be custom-lowered to SVE");

    // DataTy governs the creation of zero/poison values and predicates. It is
    // conveniently always the first "overload type" in TysForDecl.
    Type *DataTy = TysForDecl[0];

    // Parse and apply argument mappings from the table
    SmallVector<ArgMapping, 4> ArgMappings =
        parseArgMappings(Mapping.ArgMappings);
    SmallVector<Value *, 4> SVEArgs;
    for (const ArgMapping &AM : ArgMappings) {
      switch (AM.getKind()) {
      case ArgMapping::True:
        SVEArgs.push_back(createTrue(DataTy));
        break;
      case ArgMapping::Zero:
        SVEArgs.push_back(createZero(DataTy));
        break;
      case ArgMapping::Poison:
        SVEArgs.push_back(createPoison(DataTy));
        break;
      case ArgMapping::Arg:
        SVEArgs.push_back(createArg(AM.getArgIndex()));
        break;
      case ArgMapping::Imm:
        SVEArgs.push_back(createImm(AM.getArgIndex()));
        break;
      }
    }

    Intrinsic::ID SVEIntrinsicID =
        Intrinsic::lookupIntrinsicID(Mapping.SVEIntrinsic);
    auto TysForSVEDecl = computeTysForDecl(
        SVEIntrinsicID, getLegalizedSVETy(DataTy), SVEArgs, TTI);
    assert(SVEIntrinsicID != Intrinsic::not_intrinsic);
    return fromSVEInst(
        Builder.CreateIntrinsic(SVEIntrinsicID, TysForSVEDecl, SVEArgs),
        DataTy);
  }

  /// Custom rewrite a widened intrinsic without using a \p
  /// NEONToSVEIntrinsicMapping.
  Instruction *rewriteCustom(Intrinsic::ID IID, ElementCount VF) {
    // Custom lower some intrinsics.
    switch (IID) {
    case Intrinsic::aarch64_neon_uaddlp:
    case Intrinsic::aarch64_neon_saddlp: {
      // ADDLP works with adjacent elements: scaled 64-bit NEON inputs cannot be
      // legalised to SVE by interleaving. Instead, extend and use a normal
      // ADDP.
      Type *SrcTy = WideArgs[0]->getType();
      if (SrcTy->getPrimitiveSizeInBits().getKnownMinValue() != 64U)
        break;

      Type *DstTy = TysForDecl[0];
      Type *WideSrcTy = getLegalizedSVETy(DstTy);
      const bool IsSigned = IID == Intrinsic::aarch64_neon_saddlp;
      Value *Src = IsSigned ? Builder.CreateSExt(WideArgs[0], WideSrcTy)
                            : Builder.CreateZExt(WideArgs[0], WideSrcTy);
      return fromSVEInst(
          Builder.CreateIntrinsic(Intrinsic::aarch64_sve_addp, {WideSrcTy},
                                  {createTrue(WideSrcTy), Src, Src}),
          DstTy);
    }
    case Intrinsic::aarch64_neon_addp:
    case Intrinsic::aarch64_neon_faddp: {
      Type *Ty = TysForDecl[0];
      const bool IsInt = IID == Intrinsic::aarch64_neon_addp;
      // ADDP works with adjacent elements: scaled 64-bit NEON inputs cannot be
      // legalised to SVE by interleaving with undef elements.
      // Instead, deinterleave and use a normal ADD.

      // But if the initial NEON vector contained a single pair, we can
      // directly use SVE's ADDP without prior UZPQ de-interleaving.
      ElementCount NumPairsPerVec = getElementCount(Ty).divideCoefficientBy(2);
      unsigned NumPairsPerSrcVec =
          ElementCount::get(NumPairsPerVec.getKnownMinValue() /
                                VF.getKnownMinValue(),
                            NumPairsPerVec.isScalable() != VF.isScalable())
              .getFixedValue();
      if (Ty == getLegalizedSVETy(Ty) && NumPairsPerSrcVec == 1) {
        Intrinsic::ID IID =
            IsInt ? Intrinsic::aarch64_sve_addp : Intrinsic::aarch64_sve_faddp;
        return Builder.CreateIntrinsic(
            IID, {Ty}, {createTrue(Ty), WideArgs[0], WideArgs[1]});
      }

      // Within each segment, ensure we have all the pairwise adds from Args0
      // followed by those of Add1.
      Value *EvenElts =
          getSegmentedUZP(WideArgs[0], WideArgs[1], /*EvenElts=*/true, Builder);
      Value *OddElts = getSegmentedUZP(WideArgs[0], WideArgs[1],
                                       /*EvenElts=*/false, Builder);
      auto *Res =
          cast<Instruction>(IsInt ? Builder.CreateAdd(EvenElts, OddElts)
                                  : Builder.CreateFAdd(EvenElts, OddElts));

      // The segment type might not match the original type (VF = vscale x 2),
      // i.e. the original type is 64-bit but our segments are 128-bit.
      // Then, ensure that within each 64-bit segment, the lo 32-bit are
      // pairwise adds from Arg0 and the hi 32-bit are pairwise adds from Arg1.
      if (VF != ElementCount::getScalable(1)) {
        assert(VF == ElementCount::getScalable(2) &&
               Ty == getLegalizedSVETy(Ty) &&
               "Expected REVEC from NEON 64-bit to legal SVE");
        Res = concatEvenThenOddWordsWithinQuads(Res, Builder);
      }
      return Res;
    }
    case Intrinsic::aarch64_neon_sshl:
    case Intrinsic::aarch64_neon_ushl: {
      // SVE does not have a "plain" non-saturating non-rounding shl.
      // This means we need to use [su]rshl for positive shift amounts
      // and [su]qshl for negative shift amounts (effectively a shift right).
      const bool IsSignedShift = IID == Intrinsic::aarch64_neon_sshl;
      Intrinsic::ID SHLIID = IsSignedShift ? Intrinsic::aarch64_sve_srshl
                                           : Intrinsic::aarch64_sve_urshl;
      Intrinsic::ID SHRIID = IsSignedShift ? Intrinsic::aarch64_sve_sqshl
                                           : Intrinsic::aarch64_sve_uqshl;
      Type *ArgTy = TysForDecl[0];
      Value *Arg0 = createArg(0);
      Value *Arg1 = createArg(1);
      Value *SHLMask = Builder.CreateICmpSLE(
          Arg1, ConstantInt::get(getLegalizedSVETy(ArgTy), 0), "shl.mask");
      Value *SHL = Builder.CreateIntrinsic(SHLIID, {getLegalizedSVETy(ArgTy)},
                                           {SHLMask, Arg0, Arg1});
      Value *SHRMask =
          Builder.CreateIntrinsic(Intrinsic::ctlz, {SHLMask->getType()},
                                  {SHLMask, Builder.getTrue()}, {}, "shr.mask");
      return fromSVEInst(
          Builder.CreateIntrinsic(SHRIID, {getLegalizedSVETy(ArgTy)},
                                  {SHRMask, SHL, Arg1}, {}, "wide.bidir.shl"),
          ArgTy);
    }
    case Intrinsic::aarch64_neon_tbl1:
    case Intrinsic::aarch64_neon_tbl2:
    case Intrinsic::aarch64_neon_tbl3:
    case Intrinsic::aarch64_neon_tbl4: {
      Type *Ty = TysForDecl[0];
      assert(VF == ElementCount::getScalable(1) &&
             "Unexpected vscale x 2 REVEC");
      Value *Src0 = createArg(0);
      Value *Mask = createArg(WideArgs.size() - 1);
      Instruction *Res = Builder.CreateIntrinsic(
          Intrinsic::aarch64_sve_tblq, {Src0->getType()}, {Src0, Mask});
      for (unsigned SrcIdx = 1; SrcIdx < WideArgs.size() - 1; ++SrcIdx) {
        Value *Src = createArg(SrcIdx);
        Mask = Builder.CreateSub(Mask, ConstantInt::get(Mask->getType(), 16));
        Res = Builder.CreateIntrinsic(Intrinsic::aarch64_sve_tbxq,
                                      {Src->getType()}, {Res, Src, Mask});
      }
      return fromSVEInst(Res, Ty);
    }
    }
    return nullptr;
  }

private:
  ArrayRef<Type *> TysForDecl;
  ArrayRef<Value *> WideArgs;
  IRBuilderBase &Builder;

  /// Get the type that \p Ty can be legalized to for SVE instructions.
  Type *getLegalizedSVETy(Type *Ty) const {
    if (!Ty->isVectorTy())
      return Ty;
    auto *SVTy = cast<ScalableVectorType>(Ty);
    unsigned FixedBits = SVTy->getMinNumElements() *
                         SVTy->getElementType()->getScalarSizeInBits();
    return toVectorTy(
        Ty, ElementCount::getFixed(AArch64::SVEBitsPerBlock / FixedBits));
  };

  /// Turn a widened value into something that an SVE instruction can consume.
  Value *toSVEVal(Value *V) {
    Type *LegalSVETy = getLegalizedSVETy(V->getType());
    if (LegalSVETy == V->getType())
      return V;

    // For scaled 64-bit vectors, interleave/deinterleave even elements to makes
    // them valid SVE types. This plays nicely for both int and fp types later
    // during ISel.
    return Builder.CreateVectorInterleave({V, PoisonValue::get(V->getType())});
  };

  /// Turn a legalised widened value back to its expected widened type.
  Instruction *fromSVEInst(Instruction *I, Type *ExpectedTy) {
    if (ExpectedTy == I->getType())
      return I;
    auto *RevecTy = cast<VectorType>(I->getType());
    auto *ExpectedEltTy = cast<VectorType>(ExpectedTy)->getElementType();
    if (RevecTy->getElementType() != ExpectedEltTy) {
      assert(RevecTy->getElementType()->isIntegerTy(1) &&
             "Only expected SVE predicate to be turned into NEON vector.");
      Type *NeonQuadPredTy =
          VectorType::get(ExpectedEltTy, RevecTy->getElementCount());
      I = cast<Instruction>(Builder.CreateZExt(I, NeonQuadPredTy));
    }
    if (ExpectedTy == I->getType())
      return I;
    auto *Deinterleave = Builder.CreateIntrinsic(
        Intrinsic::vector_deinterleave2, I->getType(), I);
    return cast<Instruction>(Builder.CreateExtractValue(Deinterleave, {0U}));
  };

  /// Create a zero-initialised value for the \p Ty widened type.
  Value *createZero(Type *Ty) {
    return Constant::getNullValue(getLegalizedSVETy(Ty));
  }

  /// Create a poison value for the \p Ty widened type.
  Value *createPoison(Type *Ty) {
    return PoisonValue::get(getLegalizedSVETy(Ty));
  }

  /// Create an all-true predicate for the \p Ty widened data type.
  Value *createTrue(Type *Ty) {
    auto *VectorTy = cast<VectorType>(getLegalizedSVETy(Ty));
    // SVE intrinsics that narrow the element type have a predicate type with
    // half as many elements as their return type because every odd lane is
    // zeroed/undefined.
    bool IsNarrowing = any_of(WideArgs, [Ty](const Value *V) {
      return V->getType()->getScalarSizeInBits() > Ty->getScalarSizeInBits();
    });
    if (IsNarrowing)
      VectorTy = VectorType::getOneNthElementsVectorType(VectorTy, 2);
    auto *PredTy = VectorType::get(IntegerType::get(Ty->getContext(), 1),
                                   VectorTy->getElementCount());
    return Constant::getAllOnesValue(PredTy);
  }

  /// Create a legal SVE value for the argument at \p ArgIdx.
  Value *createArg(unsigned ArgIdx) {
    assert(ArgIdx < WideArgs.size() && "Argument index out of range!");
    Value *V = WideArgs[ArgIdx];
    return toSVEVal(V);
  };

  /// Derive a scalar predicate from the argument at \p ArgIdx.
  Value *createImm(unsigned ArgIdx) {
    assert(ArgIdx < WideArgs.size() && "Argument index out of range!");
    Value *V = WideArgs[ArgIdx];
    assert(isa<Constant>(V) && cast<Constant>(V)->getSplatValue());
    return cast<Constant>(V)->getSplatValue();
  };
};
} // end anonymous namespace

// Include generated vector intrinsic mapping table
#define GET_IntrinsicMappingTable_IMPL
#include "AArch64GenVectorIntrinsicMappings.inc"

bool AArch64RevectorizeInfoImpl::isTargetIntrinsicVectorizable(
    Intrinsic::ID ID) const {
  return ST.hasSVE2p1() &&
         lookupNEONToSVEMappingByNEONIntrinsic(Intrinsic::getBaseName(ID));
}

Instruction *AArch64RevectorizeInfoImpl::vectorizeTargetIntrinsic(
    Intrinsic::ID FromID, ArrayRef<Type *> TysForDecl,
    ArrayRef<Value *> WideArgs, ElementCount VF, IRBuilderBase &Builder) const {

  // Look up the mapping in the generated table
  StringRef NEONIntrinsicName = Intrinsic::getBaseName(FromID);
  const NEONToSVEIntrinsicMapping *Mapping =
      lookupNEONToSVEMappingByNEONIntrinsic(NEONIntrinsicName);
  if (!Mapping)
    llvm_unreachable("Unimplemented intrinsic vectorisation!");

  auto IsScaledNEONOrScalar = [](const Value *V) -> bool {
    if (V->getType()->isIntegerTy())
      return true;
    auto *SVTy = dyn_cast<ScalableVectorType>(V->getType());
    if (!SVTy)
      return false;
    unsigned FixedBits = SVTy->getMinNumElements() *
                         SVTy->getElementType()->getScalarSizeInBits();
    return FixedBits == AArch64::SVEBitsPerBlock ||
           FixedBits == (AArch64::SVEBitsPerBlock / 2);
  };
  assert(all_of(WideArgs, IsScaledNEONOrScalar) &&
         "Expected SVE-compatible values.");

  IntrinsicRewriter Rewriter(TysForDecl, WideArgs, Builder);
  if (Instruction *I = Rewriter.rewriteCustom(FromID, VF))
    return I;
  return Rewriter.rewriteWithMapping(*Mapping, TTI);
}

InstructionCost AArch64RevectorizeInfoImpl::getTargetIntrinsicVectorizationCost(
    Intrinsic::ID FromID, Type *WideRetTy, ArrayRef<Type *> WideArgTys,
    ElementCount VF) const {
  auto IsAcceptableTyForREVEC = [](const Type *Ty) {
    return Ty->getPrimitiveSizeInBits().getKnownMinValue() <=
           AArch64::SVEBitsPerBlock;
  };

  switch (FromID) {
  case Intrinsic::aarch64_neon_sshl:
  case Intrinsic::aarch64_neon_ushl: {
    // Re-vectorising NEON's non-saturating non-rounding bi-directional shl
    // requires the use of two SVE bi-directional shl: one for the non-rounding
    // part and one for the non-saturating part. (+masking)
    if (!IsAcceptableTyForREVEC(WideRetTy))
      return InstructionCost::getInvalid();
    return InstructionCost(2);
  }
  case Intrinsic::aarch64_neon_addp:
  case Intrinsic::aarch64_neon_faddp: {
    // Re-vectorising NEON's ADDP requires prior deinterleaving because SVE does
    // not have have a ADDPQ variant working like NEON within quads.
    // Re-vectorising 64-bit NEON requires even more instructions.
    if (!IsAcceptableTyForREVEC(WideRetTy))
      return InstructionCost::getInvalid();
    return InstructionCost(2);
  }
  case Intrinsic::aarch64_neon_tbl1:
  case Intrinsic::aarch64_neon_tbl2:
  case Intrinsic::aarch64_neon_tbl3:
  case Intrinsic::aarch64_neon_tbl4: {
    assert(WideArgTys.size() >= 2 &&
           "Tbl needs at least one source and one mask");
    if (!IsAcceptableTyForREVEC(WideRetTy) ||
        !all_of(WideArgTys, IsAcceptableTyForREVEC))
      return InstructionCost::getInvalid();
    const unsigned NumSrcs = WideArgTys.size() - 1;
    return InstructionCost(NumSrcs);
  }
  default:
    // Only allow REVEC of NEON intrinsics when the types are at most scalable
    // 128-bit vectors. This avoids crashes when forcing e.g. VF = vscale x 2.
    return (IsAcceptableTyForREVEC(WideRetTy) &&
            all_of(WideArgTys, IsAcceptableTyForREVEC))
               ? InstructionCost(1)
               : InstructionCost::getInvalid();
  }
}
