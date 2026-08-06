//===-- PISALegalizerInfo.cpp --- PISA Legalization Rules -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISALegalizerInfo.h"
#include "PISA.h"
#include "PISASubtarget.h"
#include "PISATargetMachine.h"
#include "llvm/ADT/bit.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/IR/IntrinsicsPISA.h"
#include "llvm/IR/PISAIntrinsicUtils.h"
#include "llvm/Support/PISAAddrSpace.h"

using namespace llvm;
using namespace llvm::LegalizeActions;
using namespace llvm::LegalizeMutations;
using namespace llvm::LegalityPredicates;

namespace {
constexpr ElementCount EC0 = ElementCount::getFixed(0);
constexpr ElementCount EC2 = ElementCount::getFixed(2);
constexpr ElementCount EC4 = ElementCount::getFixed(4);

// integer types
constexpr LLT I1 = LLT(LLT::Kind::INTEGER, EC0, 1);
constexpr LLT I8 = LLT(LLT::Kind::INTEGER, EC0, 8);
constexpr LLT I16 = LLT(LLT::Kind::INTEGER, EC0, 16);
constexpr LLT I32 = LLT(LLT::Kind::INTEGER, EC0, 32);
constexpr LLT I64 = LLT(LLT::Kind::INTEGER, EC0, 64);
constexpr LLT I128 = LLT(LLT::Kind::INTEGER, EC0, 128);

constexpr LLT V2I8 = LLT::fixed_vector(2, I8);
constexpr LLT V2I16 = LLT::fixed_vector(2, I16);
constexpr LLT V2I32 = LLT::fixed_vector(2, I32);

constexpr LLT V4I8 = LLT::fixed_vector(4, I8);

// floating-point types
constexpr LLT BF16 = LLT::bfloat16();
constexpr LLT F16 = LLT::float16();
constexpr LLT F32 = LLT::float32();
constexpr LLT F64 = LLT::float64();

// return true if natively supported type
static bool isLegalType(LLT Ty, bool Vector = true) {
  auto EltSize = Ty.getScalarSizeInBits();
  if (Ty.isVector() && !Vector)
    return false;
  if (Ty.isVector()) {
    auto NumElts = Ty.getNumElements();
    if (EltSize == 32)
      return NumElts <= 8 || NumElts == 16 || NumElts == 32 || NumElts == 64;
    if (!llvm::isPowerOf2_32(EltSize) || EltSize < 8 || EltSize > 64)
      return false;
    return NumElts <= 4;
  }
  if (!llvm::isPowerOf2_32(EltSize) || EltSize < 8 || EltSize > 128)
    return false;
  return true;
}

LegalityPredicate is3x8BitVector(unsigned TypeIdx) {
  return [=](const LegalityQuery &Query) {
    const LLT Ty = Query.Types[TypeIdx];
    return Ty.isVector() && Ty.getScalarSizeInBits() == 8 &&
           Ty.getNumElements() == 3;
  };
}

LegalityPredicate isWiderThan2x16BitVector(unsigned TypeIdx) {
  return [=](const LegalityQuery &Query) {
    const LLT Ty = Query.Types[TypeIdx];
    return Ty.isVector() && Ty.getScalarSizeInBits() == 16 &&
           Ty.getNumElements() > 2;
  };
}

LegalityPredicate isFloatingPointType(unsigned TypeIdx) {
  return [=](const LegalityQuery &Query) {
    const LLT Ty = Query.Types[TypeIdx];
    return Ty.getScalarType().isFloat();
  };
}
LegalizeMutation changeElementTypeToInteger(unsigned TypeIdx) {
  return [=](const LegalityQuery &Query) {
    const LLT Ty = Query.Types[TypeIdx];
    auto NewEltTy = LLT::integer(Ty.getScalarSizeInBits());
    auto NewTy = Ty.isVector()
                     ? LLT::fixed_vector(Ty.getNumElements(), NewEltTy)
                     : NewEltTy;
    return std::pair(TypeIdx, NewTy);
  };
}
} // namespace

/// Returns true if the given G_LOAD instruction operates on a vector of 5-7
/// elements each of 32 bits and should be widened for better hardware
/// utilization. Potentially can be used for other memory types.
static bool shouldWidenLoad(unsigned int Opcode, const LLT Ty,
                            unsigned AddressSpace, uint64_t Alignbits) {
  if (AddressSpace == static_cast<unsigned>(PISAAS::AddressSpace::SHARED)) {
    return (Opcode == TargetOpcode::G_LOAD && Ty.isVector() &&
            Ty.getScalarSizeInBits() == 32 && Ty.getNumElements() >= 5 &&
            Ty.getNumElements() <= 7);
  }
  if (AddressSpace == static_cast<unsigned>(PISAAS::AddressSpace::GLOBAL) ||
      AddressSpace == static_cast<unsigned>(PISAAS::AddressSpace::CONSTANT)) {
    return (Opcode == TargetOpcode::G_LOAD && Ty.isVector() &&
            Ty.getScalarSizeInBits() == 32 && Ty.getNumElements() >= 5 &&
            Ty.getNumElements() <= 7 && Alignbits >= 64);
  }
  return false;
}

PISALegalizerInfo::PISALegalizerInfo(const PISASubtarget &ST) {
  using namespace TargetOpcode;

  auto &TM = ST.getTargetLowering()->getTargetMachine();
  auto GetPointerLlt = [&](PISAAS::AddressSpace Addrspace) {
    uint32_t NumBits =
        TM.getPointerSizeInBits(static_cast<unsigned>(Addrspace));
    return LLT::pointer(static_cast<unsigned>(Addrspace), NumBits);
  };

  const LLT PrivatePtr = GetPointerLlt(PISAAS::AddressSpace::PRIVATE);
  const LLT GlobalPtr = GetPointerLlt(PISAAS::AddressSpace::GLOBAL);
  const LLT ConstantPtr = GetPointerLlt(PISAAS::AddressSpace::CONSTANT);
  const LLT SharedPtr = GetPointerLlt(PISAAS::AddressSpace::SHARED);
  const LLT GenericPtr = GetPointerLlt(PISAAS::AddressSpace::GENERIC);

  const std::initializer_list<LLT> AddrSpaces64 = {GlobalPtr, ConstantPtr,
                                                   GenericPtr};
  const std::initializer_list<LLT> AddrSpaces32 = {PrivatePtr, SharedPtr};

  auto AllIntegers = {I8, I16, I32, I64};
  auto AllFloats = {BF16, F16, F32, F64};
  auto AllPtrs = {PrivatePtr, GlobalPtr, ConstantPtr, SharedPtr, GenericPtr};

  getActionDefinitionsBuilder(
      {G_FADD, G_FCONSTANT, G_FSUB, G_FMUL, G_FMINNUM, G_FMAXNUM, G_FMINIMUM,
       G_FMAXIMUM, G_FNEG, G_FMA, G_FCEIL, G_FFLOOR, G_FRINT, G_FNEARBYINT,
       G_INTRINSIC_ROUND, G_INTRINSIC_ROUNDEVEN, G_FSQRT, G_INTRINSIC_TRUNC})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .legalFor(AllFloats)
      .scalarize(0);

  // G_FABS is lowered to bitwise AND to clear the sign bit (strict IEEE
  // semantics). For nnan cases, llvm.pisa.fabs is used instead which maps
  // directly to the PISA fabs instruction.
  getActionDefinitionsBuilder(G_FABS)
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, ElementCount::getFixed(4)))
      .fewerElementsIf(isWiderThan2x16BitVector(0),
                       changeElementCountTo(0, EC2))
      .customFor(AllFloats)
      .customIf([](const LegalityQuery &Q) {
        LLT Ty = Q.Types[0];
        return Ty.isVector() && Ty.getNumElements() == 2 &&
               Ty.getScalarSizeInBits() == 16;
      })
      .scalarize(0);

  getActionDefinitionsBuilder(
      {G_ADD, G_SUB, G_MUL, G_SDIV, G_UDIV, G_SREM, G_UREM})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .legalFor({I16, I32, I64})
      .clampScalar(0, I16, I64)
      .widenScalarToNextPow2(0)
      .scalarize(0);

  getActionDefinitionsBuilder({G_UMULO, G_SMULO})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .scalarize(0)
      .minScalar(0, I16)
      .lower();

  // leave these as scalar type for now, as they are used for legalization
  // of e.g. shufflevector, which operates on both floating and integer types
  getActionDefinitionsBuilder({G_AND, G_OR, G_XOR})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .fewerElementsIf(is3x8BitVector(0), changeTo(0, V2I8))
      .fewerElementsIf(isWiderThan2x16BitVector(0), changeTo(0, V2I16))
      .bitcastIf(LegalityPredicate(([=](const LegalityQuery &Query) {
                   const LLT Ty = Query.Types[0];
                   if (!Ty.isVector())
                     return false;
                   auto VecBitSize = Ty.getSizeInBits();
                   return VecBitSize == 32 || VecBitSize == 16;
                 })),
                 LegalizeMutation(([=](const LegalityQuery &Query) {
                   return std::pair(
                       0, LLT::integer(Query.Types[0].getSizeInBits()));
                 })))
      .legalFor({I16, I32, I64})
      .widenScalarIf(
          [=](const LegalityQuery &Query) {
            const LLT Ty = Query.Types[0];
            return Ty.getSizeInBits() == 1;
          },
          [=](const LegalityQuery &Query) { return std::pair(0, I32); })
      .widenScalarToNextPow2(0, 16)
      .clampScalar(0, I16, I64)
      .scalarize(0);

  // prelegalizer rules (div_rem_to_divrem) that generate these are disabled
  getActionDefinitionsBuilder({G_UDIVREM, G_SDIVREM}).unsupported();

  getActionDefinitionsBuilder({G_SHL, G_LSHR, G_ASHR})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .legalFor({{I16, I32}, {I32, I32}, {I64, I32}})
      .clampScalar(1, I32, I32)
      .widenScalarToNextPow2(0, 16)
      .clampScalar(0, I16, I64)
      .scalarize(0)
      .lower();

  getActionDefinitionsBuilder(G_TRUNC)
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .scalarize(0)
      .legalFor(
          {{I8, I16}, {I16, I32}, {I8, I32}, {I32, I64}, {I16, I64}, {I8, I64}})
      .customIf([=](const LegalityQuery &Query) {
        return Query.Types[0].getScalarSizeInBits() == 1 ||
               (Query.Types[1].getScalarSizeInBits() == 128 &&
                !(Query.Types[0].getScalarSizeInBits() > 64));
      })
      .alwaysLegal();

  getActionDefinitionsBuilder({G_SEXT, G_ZEXT, G_ANYEXT})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .scalarize(0)
      .legalFor(
          {{I16, I8}, {I32, I8}, {I64, I8}, {I32, I16}, {I64, I16}, {I64, I32}})
      .customIf([=](const LegalityQuery &Query) {
        auto DstSize = Query.Types[0].getScalarSizeInBits();
        auto SrcSize = Query.Types[1].getScalarSizeInBits();
        auto UseSelect = SrcSize == 1;
        auto UseShuffle = (SrcSize % 8 == 0 && !isPowerOf2_32(SrcSize)) ||
                          (DstSize % 8 == 0 && !isPowerOf2_32(DstSize));
        return UseSelect || UseShuffle;
      })
      .clampScalar(0, I16, I64)
      .clampScalar(1, I16, I32);

  getActionDefinitionsBuilder(G_SEXT_INREG).lower();

  getActionDefinitionsBuilder({G_FPTRUNC, G_INTRINSIC_FPTRUNC_ROUND})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .legalFor({{BF16, F32}, {F16, F32}, {BF16, F64}, {F16, F64}, {F32, F64}})
      .scalarize(0);

  getActionDefinitionsBuilder(G_FPEXT)
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .legalFor({{F32, BF16}, {F32, F16}, {F64, BF16}, {F64, F16}, {F64, F32}})
      .scalarize(0);

  getActionDefinitionsBuilder(G_CTPOP)
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .legalFor({{I16, I16}, {I32, I32}})
      .clampScalar(0, I16, I32)
      .clampScalar(1, I16, I32)
      .scalarize(0);

  getActionDefinitionsBuilder({G_CTTZ, G_CTLZ})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .legalFor({{I16, I16}, {I32, I32}})
      .clampScalar(0, I16, I32)
      .clampScalar(1, I16, I32)
      .scalarize(0);

  getActionDefinitionsBuilder({G_CTTZ_ZERO_POISON, G_CTLZ_ZERO_POISON})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .legalFor({{I16, I16}, {I32, I32}})
      .clampScalar(0, I16, I32)
      .clampScalar(1, I16, I64)
      .maxScalar(1, I32)
      .scalarize(0);

  getActionDefinitionsBuilder(G_BITREVERSE)
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .legalFor({I32})
      .clampScalar(0, I32, I32)
      .scalarize(0);

  getActionDefinitionsBuilder(G_FDIV)
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .customFor({BF16, F16})
      .legalFor({F32, F64})
      .scalarize(0);

  getActionDefinitionsBuilder(G_FREM)
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .customFor({BF16, F16, F32, F64})
      .scalarize(0);

  getActionDefinitionsBuilder(G_CONSTANT)
      .legalFor({I1, I8, I16, I32, I64})
      .legalIf(isPointer(0))
      .widenScalarToNextPow2(0)
      .clampScalar(0, I16, I64)
      .scalarize(0);

  getActionDefinitionsBuilder(G_PTR_ADD)
      .legalIf(all(isPointer(0), sameSize(0, 1)))
      .scalarize(0)
      .scalarSameSizeAs(1, 0);

  getActionDefinitionsBuilder({G_FLDEXP, G_STRICT_FLDEXP})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .customFor({{BF16, I32}, {F16, I32}, {F32, I32}, {F64, I32}})
      .scalarize(0);

  getActionDefinitionsBuilder({G_FSHR, G_FSHL})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .legalFor({{I32, I32}})
      .scalarize(0)
      .lower();

  getActionDefinitionsBuilder({G_ROTL, G_ROTR})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      // Scalarize vectors with i32 elements to enable lowering to scalar
      // fshl/fshr.
      .scalarizeIf(
          [](const LegalityQuery &Query) {
            return Query.Types[0].isVector() &&
                   Query.Types[0].getScalarSizeInBits() == 32;
          },
          0)
      .lower();

  getActionDefinitionsBuilder(G_IS_FPCLASS).scalarize(0).custom();

  /////////////////////////////////////////////////////////////////////////

  getActionDefinitionsBuilder(G_GLOBAL_VALUE).alwaysLegal();

  getActionDefinitionsBuilder({G_INTRINSIC, G_INTRINSIC_W_SIDE_EFFECTS,
                               G_INTRINSIC_CONVERGENT,
                               G_INTRINSIC_CONVERGENT_W_SIDE_EFFECTS})
      .alwaysLegal();

  getActionDefinitionsBuilder(G_SHUFFLE_VECTOR)
      .customIf([](const LegalityQuery &Query) {
        auto Ty = Query.Types[0];
        return Ty.isVector() && (Ty.getScalarSizeInBits() == 32) &&
               isPowerOf2_32(Ty.getNumElements());
      })
      .lower();

  getActionDefinitionsBuilder({G_MEMCPY, G_MEMCPY_INLINE, G_MEMMOVE, G_MEMSET})
      .lower();

  getActionDefinitionsBuilder(G_ADDRSPACE_CAST)
      .scalarize(0)
      .customIf([=](const LegalityQuery &Query) -> bool {
        unsigned DstAS = Query.Types[0].getAddressSpace();
        unsigned SrcAS = Query.Types[1].getAddressSpace();
        return (DstAS != (unsigned)PISAAS::AddressSpace::GENERIC) &&
               (SrcAS != (unsigned)PISAAS::AddressSpace::GENERIC);
      })
      .legalForCartesianProduct(AllPtrs, AllPtrs);

  getActionDefinitionsBuilder({G_LOAD, G_STORE})
      .bitcastIf(isFloatingPointType(0), changeElementTypeToInteger(0))
      // Handle sub-byte types: vectors with sub-byte elements (>1 bit) are
      // bitcast to scalar, then widened to multiple of 8 bits. Sub-byte
      // scalars are widened directly.
      //.widenScalar does not update MI.memoperands()[0].getType(), hence
      .customIf([=](const LegalityQuery &Query) -> bool {
        auto Ty = Query.Types[0];
        if (Ty.isVector() && (Ty.getScalarSizeInBits() > 1) &&
            (Ty.getScalarSizeInBits() < 8))
          return true;
        if (!Ty.isVector() && ((Ty.getSizeInBits() % 8) != 0))
          return true;
        return false;
      })
      .fewerElementsIf(
          [=](const LegalityQuery &Query) -> bool {
            auto EltTy = Query.Types[0];
            auto BitSize = EltTy.getScalarSizeInBits();
            auto NumElts = EltTy.isVector() ? EltTy.getNumElements() : 1;
            auto AlignInBits = Query.MMODescrs[0].AlignInBits;
            // small (bitsize<32) vectors with non-power-of-2 elements
            // can be broken into power-of-2 vectors that can be later
            // upconverted to vectors of i32 for better codegen
            return EltTy.isVector() && !isPowerOf2_32(NumElts) &&
                   BitSize != 1 && (BitSize < 32) && (BitSize < AlignInBits);
          },
          [=](const LegalityQuery &Query) -> std::pair<unsigned, LLT> {
            auto EltTy = Query.Types[0];
            auto NumElts = EltTy.getNumElements();
            auto NewNumElts = PowerOf2Ceil(NumElts) / 2;
            return std::make_pair(
                0, LLT::fixed_vector(NewNumElts, EltTy.getScalarType()));
          })
      // split up vectors of non-standard size elements
      .fewerElementsIf(
          [=](const LegalityQuery &Query) -> bool {
            auto EltTy = Query.Types[0];
            return EltTy.isVector() &&
                   !isPowerOf2_32(EltTy.getScalarSizeInBits());
          },
          [=](const LegalityQuery &Query) -> std::pair<unsigned, LLT> {
            auto EltTy = Query.Types[0];
            return std::make_pair(0, EltTy.getScalarType());
          })
      // cast non-^2 scalars to vectors of i8
      .bitcastIf(
          [=](const LegalityQuery &Query) -> bool {
            const LLT EltTy = Query.Types[0];
            auto NumBits = EltTy.getSizeInBits();
            return !EltTy.isVector() && !isPowerOf2_32(NumBits);
          },
          [=](const LegalityQuery &Query) -> std::pair<unsigned, LLT> {
            auto Size = Query.Types[0].getSizeInBits();
            return std::pair(0, LLT::fixed_vector(Size / 8, I8));
          })
      // cast scalar/vector with large bitsize into <? x i32>
      .bitcastIf(
          [=](const LegalityQuery &Query) -> bool {
            const LLT EltTy = Query.Types[0];
            auto NumBits = EltTy.getScalarSizeInBits();
            auto IsAtomic128 =
                EltTy.isScalar() && (NumBits == 128) &&
                isStrongerThanMonotonic(Query.MMODescrs[0].Ordering);
            return !IsAtomic128 && (NumBits % 32 == 0) && (NumBits > 64);
          },
          [=](const LegalityQuery &Query) -> std::pair<unsigned, LLT> {
            const LLT EltTy = Query.Types[0];
            auto NumBits = EltTy.getSizeInBits();
            return std::pair(0, LLT::fixed_vector(NumBits / 32, I32));
          })
      .bitcastIf(([=](const LegalityQuery &Query) -> bool {
                   auto EltTy = Query.Types[0];
                   auto BitSize = EltTy.getScalarSizeInBits();
                   auto AccSize = EltTy.getSizeInBits();
                   auto AlignInBits = Query.MMODescrs[0].AlignInBits;
                   auto SmallVectorWithManyElements =
                       (BitSize < 32) && EltTy.isVector() &&
                       (EltTy.getNumElements() > 4);

                   if ((AlignInBits >= AccSize) && !SmallVectorWithManyElements)
                     return false; // all good already
                   if (AlignInBits == BitSize)
                     return false; // handled by scalarizeIf code below
                   if ((AlignInBits < AccSize) && (AccSize % AlignInBits))
                     return false; // weird size/alignment
                   if ((BitSize < 32) && (AccSize % 32 == 0) &&
                       (AlignInBits % 32 == 0))
                     return true; // will bitcast to <? x i32>

                   return (AlignInBits < BitSize) ||
                          ((BitSize < 32) && (AlignInBits < AccSize));
                 }),
                 [=](const LegalityQuery &Query) -> std::pair<unsigned, LLT> {
                   auto EltTy = Query.Types[0];
                   auto BitSize = EltTy.getScalarSizeInBits();
                   auto AccSize = EltTy.getSizeInBits();
                   auto AlignInBits = Query.MMODescrs[0].AlignInBits;

                   if ((BitSize < 32) && (AccSize % 32 == 0) &&
                       (AlignInBits % 32 == 0))
                     AlignInBits = 32;

                   auto NewEltTy = LLT::integer(AlignInBits);
                   auto NewNumElts = AccSize / AlignInBits;
                   auto NewTy = NewNumElts == 1
                                    ? NewEltTy
                                    : LLT::fixed_vector(NewNumElts, NewEltTy);
                   return std::pair(0, NewTy);
                 })
      // bitcast <6 x i32> to <3 x i64> if alignment is sufficient
      .bitcastIf(([=](const LegalityQuery &Query) -> bool {
                   auto EltTy = Query.Types[0];
                   return EltTy.isVector() &&
                          (EltTy.getScalarSizeInBits() == 32) &&
                          (EltTy.getNumElements() == 6) &&
                          (Query.MMODescrs[0].AlignInBits >= 64);
                 }),
                 [=](const LegalityQuery &Query) -> std::pair<unsigned, LLT> {
                   auto NewTy = LLT::fixed_vector(3, LLT::integer(64));
                   return std::pair(0, NewTy);
                 })
      // Increase the number of elements to corresponding vector of i8
      .customIf([=](const LegalityQuery &Query) {
        auto EltTy = Query.Types[0];
        return EltTy.getScalarSizeInBits() == 1;
      })
      // expand s32 vectors with 4 < elts < 8 to have 8 elements
      // Enabled only for shared memory where loads of OOB accesses are
      // guaranteed to return 0
      .customIf([=](const LegalityQuery &Query) {
        auto EltTy = Query.Types[0];
        auto AlignInBits = Query.MMODescrs[0].AlignInBits;
        return shouldWidenLoad(Query.Opcode, EltTy,
                               Query.Types[1].getAddressSpace(), AlignInBits);
      })
      // <4 x i8> align 1 .. needs to be broken down into 4 loads
      .scalarizeIf(([=](const LegalityQuery &Query) -> bool {
                     auto EltTy = Query.Types[0];
                     auto BitSize = EltTy.getScalarSizeInBits();
                     auto AccSize = EltTy.getSizeInBits();
                     auto AlignInBits = Query.MMODescrs[0].AlignInBits;
                     return ((BitSize < 32) && (AlignInBits < AccSize));
                   }),
                   0)
      // maximum number of each type that we can load/store
      .clampMaxNumElements(0, PrivatePtr, 8)
      .clampMaxNumElements(0, GlobalPtr, 4)
      .clampMaxNumElements(0, ConstantPtr, 4)
      .clampMaxNumElements(0, SharedPtr, 8)
      .clampMaxNumElements(0, GenericPtr, 4)
      .clampMaxNumElements(0, I8, 4)
      .clampMaxNumElements(0, I16, 4)
      .clampMaxNumElements(0, I32, 8)
      .clampMaxNumElements(0, I64, 4)
      .clampMaxNumElements(0, I1, 64)
      // support odd-element vectors, e.g. <7 x i32>
      // others, e.g. <5 x i16> have been clamped above
      .fewerElementsIf(
          [=](const LegalityQuery &Query) -> bool {
            const LLT EltTy = Query.Types[0];
            if (!EltTy.isVector())
              return false;
            auto NumElements = EltTy.getNumElements();
            if (!isPowerOf2_32(NumElements))
              return !((NumElements == 3) &&
                       ((EltTy.getScalarSizeInBits() == 32) ||
                        (EltTy.getScalarSizeInBits() == 64)));
            return false;
          },
          [=](const LegalityQuery &Query) -> std::pair<unsigned, LLT> {
            const LLT EltTy = Query.Types[0];
            auto NewNumElts = PowerOf2Ceil(EltTy.getNumElements()) / 2;
            return std::pair(
                0, LLT::fixed_vector(NewNumElts, EltTy.getScalarType()));
          })
      // load/store of ptr requires inttoptr/ptrtoint
      // - has to come after clamping of max elements
      .customIf([=](const LegalityQuery &Query) {
        return Query.Types[0].getScalarType().isPointer();
      })
      // default
      .legalIf(typeInSet(1, AllPtrs));

  // lower to a narrow G_LOAD + // G_SEXT/G_ZEXT.
  getActionDefinitionsBuilder({G_SEXTLOAD, G_ZEXTLOAD}).custom();

  getActionDefinitionsBuilder(G_FCANONICALIZE).legalFor(AllFloats);

  getActionDefinitionsBuilder({G_FPTOSI, G_FPTOUI, G_FPTOSI_SAT, G_FPTOUI_SAT})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .legalForCartesianProduct(AllIntegers, AllFloats)
      .scalarize(0)
      .minScalar(0, I8);

  getActionDefinitionsBuilder(G_LROUND)
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .legalFor({{I32, F32}, {I64, F32}, {I32, F64}, {I64, F64}})
      .clampScalar(0, I32, I64)
      .scalarize(0);

  getActionDefinitionsBuilder(G_LLROUND)
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .legalFor({{I64, F32}, {I64, F64}})
      .clampScalar(0, I64, I64)
      .scalarize(0);

  getActionDefinitionsBuilder({G_SITOFP, G_UITOFP})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      // GlobalIsel built-in lowering doesn't fully support fp16 yet,
      // so we have to custom lower it for I1 source type
      .customIf(all(typeIs(1, I1), typeIs(0, BF16)))
      .customIf(all(typeIs(1, I1), typeIs(0, F16)))
      .legalForCartesianProduct(AllFloats, AllIntegers)
      // other types should prefer built-in lowering
      .lowerIf(typeIs(1, I1))
      .widenScalarToNextPow2(1)
      .scalarize(0);

  getActionDefinitionsBuilder({G_SMIN, G_SMAX, G_UMIN, G_UMAX, G_ABS})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .legalFor({I16, I32, I64})
      .minScalar(0, I16)
      .scalarize(0)
      .lower();

  // G_PHI is legal for vector types. However, since most of
  // the PISA operations are scalar, there will be a need for
  // (vector) extract op. The assumption here is that extraction
  // in the loop header will allow for better loop body codegen.
  getActionDefinitionsBuilder(G_PHI)
      .legalFor(AllPtrs)
      .legalFor(AllIntegers)
      .legalFor(AllFloats)
      .legalFor({I1})
      .widenScalarToNextPow2(0, 16)
      .clampScalar(0, I16, I64)
      .scalarize(0);

  getActionDefinitionsBuilder(G_BITCAST)
      // allow bitcasts between pointers and non-pointers (ptr2int/int2ptr)
      .customIf([=](const LegalityQuery &Query) {
        auto DstTy = Query.Types[0];
        auto SrcTy = Query.Types[1];
        return (DstTy.isPointer() != SrcTy.isPointer());
      })
      // In cases where both source and destination operands are vectors,
      // the standard bitcast lowering expects the number of elements to be
      // divisible by each other, e.g. <4 x i32> to <8 x i16>; use custom
      // legalization to handle other cases, e.g. <5 x i32> to <2 x i80>
      .customIf([=](const LegalityQuery &Query) {
        auto DstTy = Query.Types[0];
        auto SrcTy = Query.Types[1];
        if (!SrcTy.isVector() || !DstTy.isVector())
          return false;
        unsigned SrcNumElts = SrcTy.getNumElements();
        unsigned DstNumElts = DstTy.getNumElements();
        return (SrcNumElts % DstNumElts != 0) && (DstNumElts % SrcNumElts != 0);
      })
      // Handle bitcasts between vectors with the same element count and scalar
      // size but more than 4 elements whose total bit width is not a power of
      // 2 (e.g. <5 x f16> to <5 x i16>, 80 bits). Decompose element-wise to
      // avoid creating illegal G_UNMERGE_VALUES on odd-sized vectors
      // downstream.
      .customIf([=](const LegalityQuery &Query) {
        auto DstTy = Query.Types[0];
        auto SrcTy = Query.Types[1];
        if (!SrcTy.isVector() || !DstTy.isVector())
          return false;
        auto DstNumElts = DstTy.isVector() ? DstTy.getNumElements() : 1;
        auto SrcNumElts = SrcTy.isVector() ? SrcTy.getNumElements() : 1;
        auto DstEltSize = DstTy.getScalarSizeInBits();
        auto SrcEltSize = SrcTy.getScalarSizeInBits();
        if (DstNumElts != SrcNumElts)
          return false;
        if (DstEltSize != SrcEltSize)
          return false;
        return DstNumElts > 4 && DstEltSize != 32 &&
               !isPowerOf2_32(DstTy.getSizeInBits());
      })
      .legalIf([=](const LegalityQuery &Query) {
        auto DstTy = Query.Types[0];
        auto SrcTy = Query.Types[1];
        auto DstNumElts = DstTy.isVector() ? DstTy.getNumElements() : 1;
        auto SrcNumElts = SrcTy.isVector() ? SrcTy.getNumElements() : 1;
        auto DstEltSize = DstTy.getScalarSizeInBits();
        auto SrcEltSize = SrcTy.getScalarSizeInBits();
        auto CastSize = DstTy.getSizeInBits();
        if (DstEltSize == 32 && SrcEltSize == 32)
          // vectors of 32bit integer <=> floats
          return DstNumElts <= 8 || DstNumElts == 16 || DstNumElts == 32 ||
                 DstNumElts == 64;
        if (DstNumElts > 4 || SrcNumElts > 4)
          // can not use swizzle for copy
          return false;
        if (DstEltSize < 8 || SrcEltSize < 8)
          // sub-byte types are not natively supported
          return false;
        if ((DstTy.isVector() && DstEltSize > 64) ||
            (SrcTy.isVector() && SrcEltSize > 64))
          // No vector register class exists for elements wider than 64 bits
          // (e.g. <2 x i128>); such casts must be lowered element-wise rather
          // than marked legal, otherwise instruction selection has no vector
          // register class to constrain to and asserts.
          return false;
        if (CastSize > 128)
          // no registers of such size
          return DstNumElts != 1 && SrcNumElts != 1;
        if (!llvm::isPowerOf2_32(CastSize))
          // non-power-of-2 bitcasts can only be between 3-element vectors
          return DstNumElts == 3 && SrcNumElts == 3;
        return true;
      })
      .lower();

  for (unsigned Op : {G_EXTRACT_VECTOR_ELT, G_INSERT_VECTOR_ELT}) {
    unsigned SrcTyIdx = Op == G_EXTRACT_VECTOR_ELT ? 1 : 0;

    getActionDefinitionsBuilder(Op)
        // extend vectors of i1 to have power of two elements
        .moreElementsIf(
            ([=](const LegalityQuery &Query) {
              auto DstTy = Query.Types[SrcTyIdx];
              auto BitSize = DstTy.getSizeInBits();
              return DstTy.isVector() && (DstTy.getScalarSizeInBits() == 1) &&
                     ((BitSize < 8) || !isPowerOf2_32(BitSize));
            }),
            [=](const LegalityQuery &Query) {
              auto DstTy = Query.Types[SrcTyIdx];
              unsigned NumElts = PowerOf2Ceil(DstTy.getNumElements());
              NumElts = std::max(8u, NumElts);
              return std::pair(
                  SrcTyIdx, LLT::fixed_vector(NumElts, DstTy.getScalarType()));
            })
        // <? x i1>
        .customIf([=](const LegalityQuery &Query) {
          auto EltSize = Query.Types[SrcTyIdx].getScalarSizeInBits();
          return (EltSize == 1);
        })
        // increase to a multiple of elements, e.g. <5 x i16> => <8 x i16>
        .moreElementsIf(
            [=](const LegalityQuery &Query) {
              auto SrcTy = Query.Types[SrcTyIdx];
              unsigned NumElts = SrcTy.getNumElements();
              if (SrcTy.getScalarSizeInBits() != 32)
                return (NumElts > 4) && (NumElts % 4);
              return (NumElts > 8) && (NumElts != 16) && (NumElts % 32);
            },
            [=](const LegalityQuery &Query) {
              auto SrcTy = Query.Types[SrcTyIdx];
              auto ScalarTy = SrcTy.getScalarType();
              auto NumElts = PowerOf2Ceil(SrcTy.getNumElements());
              return std::pair(SrcTyIdx, LLT::fixed_vector(NumElts, ScalarTy));
            })
        // cast non-s32 elements to s32 vector, e.g.
        // <N x s8> => <N/4 x s32>, iff the index is non-constant
        .customIf([=](const LegalityQuery &Query) {
          const LLT Ty = Query.Types[SrcTyIdx];
          const auto EltSize = Ty.getScalarSizeInBits();
          const auto NumElts = Ty.getNumElements();
          // Only needed for non-s32 elements
          if (EltSize != 8 && EltSize != 16 && EltSize != 64)
            return false;
          // The vector must fit into <64 x s32>, otherwise cannot bitcast
          if (NumElts * EltSize / 32 > 64)
            return false;
          return true;
        })
        // reduce to a multiple of elements, e.g. <8 x i16> => <4 x i16>
        // (x2)
        //   - each multiple of elements is supported natively
        //   - operation will use 'insert/extract' or swizzle
        .fewerElementsIf(
            [=](const LegalityQuery &Query) {
              auto SrcTy = Query.Types[SrcTyIdx];
              auto MaxElts = SrcTy.getScalarSizeInBits() == 32 ? 64 : 4;
              return SrcTy.getNumElements() > MaxElts;
            },
            [=](const LegalityQuery &Query) {
              auto SrcTy = Query.Types[SrcTyIdx];
              auto ScalarTy = SrcTy.getScalarType();
              return (SrcTy.getScalarSizeInBits() == 32)
                         ? std::pair(SrcTyIdx, LLT::fixed_vector(64, ScalarTy))
                         : std::pair(SrcTyIdx, LLT::fixed_vector(4, ScalarTy));
            })
        .lowerIf([=](const LegalityQuery &Query) {
          return Query.Types[SrcTyIdx].getScalarSizeInBits() != 32;
        })
        .alwaysLegal();
  }

  getActionDefinitionsBuilder(G_INSERT_SUBVECTOR)
      .customIf([=](const LegalityQuery &Query) {
        const LLT Ty = Query.Types[0];
        return (Ty.getScalarSizeInBits() == 32 ||
                Ty.getScalarSizeInBits() == 64);
      })
      .unsupported(); // no lower() implementation

  getActionDefinitionsBuilder(G_EXTRACT_SUBVECTOR)
      // A 2-element sub-vector extracted from a wider same-element vector is a
      // nameable composite sub-register slice (.xy / .zw); ISel lowers it to a
      // sub-register COPY. Mark it legal so the post-legalizer combiner may
      // produce it (see build_vector_from_unmerge_lanes).
      .legalIf([=](const LegalityQuery &Query) {
        const LLT Dst = Query.Types[0];
        const LLT Src = Query.Types[1];
        return Dst.isVector() && Src.isVector() && Dst.getNumElements() == 2 &&
               Src.getNumElements() > 2 && Src.getNumElements() <= 4 &&
               Src.getElementType() == Dst.getElementType() &&
               (Dst.getScalarSizeInBits() == 8 ||
                Dst.getScalarSizeInBits() == 16);
      })
      .customIf([=](const LegalityQuery &Query) {
        const LLT Ty = Query.Types[0];
        return (Ty.getScalarSizeInBits() == 32 ||
                Ty.getScalarSizeInBits() == 64);
      })
      .unsupported(); // no lower() implementation

  getActionDefinitionsBuilder({G_INSERT, G_EXTRACT}).lower();

  getActionDefinitionsBuilder(G_CONCAT_VECTORS)
      .legalIf([=](const LegalityQuery &Query) {
        // vector(big) <=> vector(lit)
        const LLT BigTy = Query.Types[0];
        return BigTy.isVector() && (BigTy.getNumElements() <= 4) &&
               (BigTy.getSizeInBits() <= 256); // v4s64
      })
      .customIf([=](const LegalityQuery &Query) {
        // return true if we want to use 'insert', instead of swizzle
        auto LitTy = Query.Types[1];
        auto BigTy = Query.Types[0];
        auto EltOk = BigTy.getScalarSizeInBits() == 32;
        auto VecOk = LitTy.isVector() && BigTy.isVector();
        return EltOk && VecOk &&
               ((LitTy.getNumElements() > 4) || (BigTy.getNumElements() > 4));
      })
      .clampMaxNumElements(1, I8, 4)
      .clampMaxNumElements(1, I16, 4)
      // .clampMaxNumElements(1, I32, 32) handled by customIf.
      .clampMaxNumElements(1, I64, 4);

  getActionDefinitionsBuilder(
      {G_VECREDUCE_SMIN, G_VECREDUCE_SMAX, G_VECREDUCE_UMIN, G_VECREDUCE_UMAX,
       G_VECREDUCE_ADD, G_VECREDUCE_MUL, G_VECREDUCE_OR, G_VECREDUCE_AND,
       G_VECREDUCE_XOR, G_VECREDUCE_FMUL, G_VECREDUCE_FMIN, G_VECREDUCE_FMAX,
       G_VECREDUCE_FMINIMUM, G_VECREDUCE_FMAXIMUM})
      // fewerElementsVectorReductions does not handle (SrcElts % DstElts != 0)
      .moreElementsIf(([=](const LegalityQuery &Query) {
                        auto NumElts = Query.Types[1].getNumElements();
                        return NumElts > 4 && NumElts % 4 != 0;
                      }),
                      [=](const LegalityQuery &Query) {
                        auto SrcTy = Query.Types[1];
                        auto NewNumElts =
                            llvm::PowerOf2Ceil(SrcTy.getNumElements());
                        auto NewSrcTy = LLT::fixed_vector(
                            NewNumElts, SrcTy.getScalarType());
                        return std::pair(1, NewSrcTy);
                      })
      .fewerElementsIf(vectorElementCountIsGreaterThan(1, 4),
                       changeElementCountTo(1, EC4))
      .scalarize(1)
      .lower();

  for (unsigned Op : {G_MERGE_VALUES, G_UNMERGE_VALUES}) {
    unsigned BigTyIdx = Op == G_UNMERGE_VALUES ? 1 : 0;
    unsigned LitTyIdx = Op == G_UNMERGE_VALUES ? 0 : 1;

    auto &Builder = getActionDefinitionsBuilder(Op);
    Builder.customIf([=](const LegalityQuery &Query) {
      // return true if we want to use 'extract', instead of swizzle
      auto BigTy = Query.Types[BigTyIdx];
      return BigTy.isVector() && (BigTy.getScalarSizeInBits() == 32) &&
             (BigTy.getNumElements() > 4);
    });

    Builder
        .legalIf([=](const LegalityQuery &Query) {
          // vector(big) <=> scalar/vector(lit)
          auto BigTy = Query.Types[BigTyIdx];
          auto LitTy = Query.Types[LitTyIdx];
          auto LitTyValid = LitTy.isScalar() ? BigTy.getScalarType() == LitTy
                                             : LitTy.getScalarSizeInBits() >= 8;
          // No register class exists for vectors with elements wider than
          // 64 bits, so <N x i128> and friends must not be marked legal here
          // (they would otherwise crash instruction selection when looking up
          // a vector register class). Cap the vector element size at 64.
          return LitTyValid && BigTy.isVector() &&
                 (BigTy.getScalarSizeInBits() <= 64) &&
                 (BigTy.getNumElements() <= 4) &&
                 (BigTy.getSizeInBits() <= 256); // v4s64
        })
        .widenScalarIf(
            [=](const LegalityQuery &Query) {
              auto BigTy = Query.Types[BigTyIdx];
              auto BigTySize = BigTy.getSizeInBits();
              return BigTy.isScalar() && BigTySize > 64 &&
                     !isPowerOf2_32(BigTySize);
            },
            [=](const LegalityQuery &Query) {
              auto BigTy = Query.Types[BigTyIdx];
              unsigned NewSizeInBits =
                  1 << Log2_32_Ceil(BigTy.getSizeInBits() + 1);
              return std::pair(BigTyIdx, LLT::integer(NewSizeInBits));
            })
        .lowerIf([=](const LegalityQuery &Query) {
          // lower to shift/mask if conversion would
          // result in a vector with >4 elements
          auto BigTy = Query.Types[BigTyIdx];
          auto LitTy = Query.Types[LitTyIdx];
          auto NumElts = BigTy.getSizeInBits() / LitTy.getScalarSizeInBits();
          return BigTy.isScalar() && (NumElts > 4);
        })
        .lowerIf(all(vectorElementCountIsGreaterThan(LitTyIdx, 4),
                     vectorElementCountIsGreaterThan(BigTyIdx, 4)))
        .fewerElementsIf(vectorElementCountIsGreaterThan(BigTyIdx, 4),
                         changeElementCountTo(BigTyIdx, EC4))
        .minScalarOrEltIf(scalarNarrowerThan(LitTyIdx, 16), LitTyIdx, I16)
        .legalIf([=](const LegalityQuery &Query) {
          return (Query.Types[BigTyIdx].isScalar() ||
                  Query.Types[BigTyIdx].isPointer()) &&
                 (Query.Types[LitTyIdx].isScalar() ||
                  Query.Types[LitTyIdx].isPointer());
        });
  }

  getActionDefinitionsBuilder(G_BUILD_VECTOR).alwaysLegal();

  getActionDefinitionsBuilder(G_IMPLICIT_DEF)
      .legalIf([=](const LegalityQuery &Query) {
        return isLegalType(Query.Types[0]);
      })
      .legalFor({I1})
      .widenScalarToNextPow2(0)
      .clampScalar(0, I16, I64)
      .scalarize(0);

  getActionDefinitionsBuilder(G_FREEZE)
      .legalFor(AllIntegers)
      .legalFor(AllFloats)
      .legalFor(AllPtrs)
      .widenScalarToNextPow2(0)
      .clampScalar(0, I32, I64)
      .scalarize(0);

  getActionDefinitionsBuilder(G_INTTOPTR)
      // List the common cases
      .legalForCartesianProduct(AddrSpaces64, {I64})
      .legalForCartesianProduct(AddrSpaces32, {I32})
      .scalarize(0)
      // Accept any address space as long as the size matches
      .legalIf(sameSize(0, 1))
      .widenScalarIf(smallerThan(1, 0),
                     [](const LegalityQuery &Query) {
                       return std::pair(
                           1, LLT::integer(Query.Types[0].getSizeInBits()));
                     })
      .narrowScalarIf(largerThan(1, 0), [](const LegalityQuery &Query) {
        return std::pair(1, LLT::integer(Query.Types[0].getSizeInBits()));
      });

  getActionDefinitionsBuilder(G_PTRTOINT)
      // List the common cases
      .legalForCartesianProduct(AddrSpaces64, {I64})
      .legalForCartesianProduct(AddrSpaces32, {I32})
      .scalarize(0)
      // Accept any address space as long as the size matches
      .legalIf(sameSize(0, 1))
      .widenScalarIf(smallerThan(0, 1),
                     [](const LegalityQuery &Query) {
                       return std::pair(
                           0, LLT::integer(Query.Types[1].getSizeInBits()));
                     })
      .narrowScalarIf(largerThan(0, 1), [](const LegalityQuery &Query) {
        return std::pair(0, LLT::integer(Query.Types[1].getSizeInBits()));
      });

  getActionDefinitionsBuilder(G_ICMP)
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .legalIf(all(
          typeIs(0, I1),
          LegalityPredicates::any(isPointer(1), typeInSet(1, {I16, I32, I64}))))
      .widenScalarToNextPow2(1)
      .scalarize(0)
      .clampScalar(1, I16, I64);

  getActionDefinitionsBuilder(G_FCMP)
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .scalarize(0)
      .custom();

  getActionDefinitionsBuilder({G_SCMP, G_UCMP})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .lower();

  getActionDefinitionsBuilder(G_SELECT)
      .legalIf(all(
          LegalityPredicates::any(isPointer(0), typeInSet(0, {I16, I32, I64})),
          typeIs(1, I1)))
      .legalIf(
          all(LegalityPredicates::any(isPointer(0), typeInSet(0, AllFloats)),
              typeIs(1, I1)))
      .scalarize(0)
      .clampScalar(0, I16, I64)
      .widenScalarToNextPow2(0);

  getActionDefinitionsBuilder(
      {G_ATOMICRMW_OR, G_ATOMICRMW_ADD, G_ATOMICRMW_AND, G_ATOMICRMW_MAX,
       G_ATOMICRMW_MIN, G_ATOMICRMW_SUB, G_ATOMICRMW_XOR, G_ATOMICRMW_UMAX,
       G_ATOMICRMW_UMIN, G_ATOMICRMW_UINC_WRAP, G_ATOMICRMW_UDEC_WRAP})
      // PISA supports up to acq_rel, anything that is not included in that
      // needs custom legalization.
      // In other words if acq_rel provides the same or stronger guarantees
      // than the requested ordering, then the operation is legal, otherwise
      // it needs custom legalization.
      // Note that this is not equivalent to isStrongerThan(Ordering, AcqRel)
      // because memory orderings do not form a total order.
      // For example, Acquire is neither stronger nor weaker than Release.
      .customIf([=](const LegalityQuery &Query) {
        return !isAtLeastOrStrongerThan(AtomicOrdering::AcquireRelease,
                                        Query.MMODescrs[0].Ordering);
      })
      .legalForCartesianProduct({I16, I32, I64},
                                {GlobalPtr, SharedPtr, GenericPtr});

  getActionDefinitionsBuilder(
      {G_ATOMICRMW_FADD, G_ATOMICRMW_FSUB, G_ATOMICRMW_FMIN, G_ATOMICRMW_FMAX})
      .customIf([=](const LegalityQuery &Query) {
        return !isAtLeastOrStrongerThan(AtomicOrdering::AcquireRelease,
                                        Query.MMODescrs[0].Ordering);
      })
      .legalForCartesianProduct(AllFloats, {GlobalPtr, SharedPtr, GenericPtr});

  getActionDefinitionsBuilder(G_ATOMICRMW_XCHG)
      .customIf([=](const LegalityQuery &Query) {
        return !isAtLeastOrStrongerThan(AtomicOrdering::AcquireRelease,
                                        Query.MMODescrs[0].Ordering);
      })
      .legalForCartesianProduct({I16, I32, I64, I128},
                                {GlobalPtr, SharedPtr, GenericPtr})
      .customIf([=](const LegalityQuery &Query) {
        return Query.Types[0].getScalarType().isPointer();
      });

  getActionDefinitionsBuilder(G_ATOMIC_CMPXCHG_WITH_SUCCESS).lower();
  // For cmpxchg in case of failure the strongest ordering we can do
  // directly is 'acquire', anything stronger needs custom legalization.
  getActionDefinitionsBuilder(G_ATOMIC_CMPXCHG)
      .customIf([=](const LegalityQuery &Query) {
        return !isAtLeastOrStrongerThan(AtomicOrdering::AcquireRelease,
                                        Query.MMODescrs[0].Ordering) ||
               !isAtLeastOrStrongerThan(AtomicOrdering::Acquire,
                                        Query.MMODescrs[0].FailureOrdering);
      })
      .alwaysLegal();

  getActionDefinitionsBuilder({G_UADDSAT, G_USUBSAT})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .minScalar(0, I16)
      .scalarize(0)
      .lower();

  getActionDefinitionsBuilder({G_SADDSAT, G_SSUBSAT})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .minScalar(0, I16)
      .legalFor({I16, I32, I64})
      .scalarize(0)
      .lower();

  getActionDefinitionsBuilder({G_UADDO, G_USUBO, G_UADDE, G_USUBE})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .scalarize(0)
      .clampScalar(0, I32, I32)
      .legalFor({{I32, I1}});

  getActionDefinitionsBuilder({G_SADDO, G_SSUBO, G_SADDE, G_SSUBE})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .lower();

  // pointer-handling.
  getActionDefinitionsBuilder(G_FRAME_INDEX).legalFor({PrivatePtr, SharedPtr});

  // control-flow. In some cases (e.g. constants) i1 may be promoted to i32.
  getActionDefinitionsBuilder(G_BR).alwaysLegal();
  getActionDefinitionsBuilder(G_BRCOND).legalFor({I1, I32});
  getActionDefinitionsBuilder(G_FENCE).alwaysLegal();
  getActionDefinitionsBuilder({G_TRAP, G_DEBUGTRAP, G_UBSANTRAP}).alwaysLegal();

  getActionDefinitionsBuilder({G_FCOS, G_FSIN, G_FTANH, G_FEXP2, G_FLOG2})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .legalFor({BF16, F16, F32})
      .scalarize(0);

  getActionDefinitionsBuilder({G_FEXP, G_FEXP10, G_FLOG, G_FLOG10, G_FPOW})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .customFor({BF16, F16, F32})
      .scalarize(0);

  getActionDefinitionsBuilder(G_FPOWI).lower();

  getActionDefinitionsBuilder(G_FCOPYSIGN)
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .lower();

  getActionDefinitionsBuilder({G_SMULH, G_UMULH})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .scalarize(0)
      .customFor({I64})
      .lower();

  getActionDefinitionsBuilder(G_BSWAP)
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .scalarize(0)
      .customIf([=](const LegalityQuery &Query) {
        const LLT Ty = Query.Types[0];
        unsigned BitSize = Ty.getSizeInBits();
        return Ty.isScalar() && (BitSize % 16 == 0);
      })
      .unsupported();

  getActionDefinitionsBuilder(G_CONSTANT_FOLD_BARRIER)
      .legalFor({I8, I16, I32, I64});

  getActionDefinitionsBuilder({G_SBFX, G_UBFX})
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, 4),
                       changeElementCountTo(0, EC4))
      .legalFor({{I32, I32}})
      .clampScalar(1, I32, I32)
      .clampScalar(0, I32, I32)
      .scalarize(0);

  getActionDefinitionsBuilder(G_DYN_STACKALLOC).legalFor({{PrivatePtr, I32}});

  getActionDefinitionsBuilder({G_READSTEADYCOUNTER, G_READCYCLECOUNTER})
      .legalFor({I64});

  verify(*ST.getInstrInfo());
}

// scalarize an intrinsic instruction with vector arguments
static SmallVector<MachineInstr *> scalarizeIntrinsic(MachineInstr &MI) {
  Intrinsic::ID IntrinsicID = cast<GIntrinsic>(MI).getIntrinsicID();
  MachineIRBuilder B(MI);
  auto &MRI = *B.getMRI();

  SmallVector<MachineInstr *> NewMIs;
  auto DstTy = MRI.getType(MI.getOperand(0).getReg());
  if (!DstTy.isVector()) {
    NewMIs.push_back(&MI);
    return NewMIs;
  }

  SmallVector<Register, 4> VecRegs;
  for (unsigned I = 0; I < DstTy.getNumElements(); I++) {
    SmallVector<MachineOperand, 4> Opnds;
    for (unsigned J = 2; J < MI.getNumOperands(); J++) { // dst, iid
      auto Opnd = MI.getOperand(J);
      if (Opnd.isReg()) {
        auto ArgTy = MRI.getType(Opnd.getReg());
        if (ArgTy.isVector()) {
          ArgTy = ArgTy.getScalarType();
          auto ArgReg = MRI.createGenericVirtualRegister(ArgTy);
          B.buildExtractVectorElementConstant(ArgReg, Opnd, I);
          Opnds.push_back(MachineOperand::CreateReg(ArgReg, false));
        } else { // use register operand as-is
          Opnds.push_back(Opnd);
        }
      } else { // use immediate operand as-is (e.g. rounding mode)
        Opnds.push_back(Opnd);
      }
    }
    auto DstReg = MRI.createGenericVirtualRegister(DstTy.getScalarType());
    auto Res = B.buildIntrinsic(IntrinsicID, DstReg);
    NewMIs.push_back(Res);
    Res.setMIFlags(MI.getFlags());
    for (auto It = Opnds.begin(), Ite = Opnds.end(); It != Ite; ++It)
      Res.add(*It);
    VecRegs.push_back(DstReg);
  }
  B.buildBuildVector(MI.getOperand(0), VecRegs);
  MI.eraseFromParent();
  return NewMIs;
}

// flog(x) = flog2(x) * ln(2)
static bool legalizeGFlog(MachineInstr &MI, MachineIRBuilder &B,
                          double Log2BaseInverted) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  LLT Ty = B.getMRI()->getType(Dst);
  unsigned Flags = MI.getFlags();

  auto &Semantics = getFltSemanticForLLT(Ty.getScalarType());
  APFloat APFLog2BaseInverted(Log2BaseInverted);
  bool LosesInfo; // ignored
  APFLog2BaseInverted.convert(Semantics, APFloat::rmNearestTiesToEven,
                              &LosesInfo);

  auto Log2Operand = B.buildFLog2(Ty, Src, Flags);
  auto Log2BaseInvertedOperand = B.buildFConstant(Ty, APFLog2BaseInverted);

  B.buildFMul(Dst, Log2Operand, Log2BaseInvertedOperand, Flags);
  MI.eraseFromParent();
  return true;
}

// fexp(x) = fexp2(x * log2(e))
static bool legalizeGFexp(MachineInstr &MI, MachineIRBuilder &B,
                          double Multiplicand) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  unsigned Flags = MI.getFlags();
  LLT Ty = B.getMRI()->getType(Dst);

  auto &Semantics = getFltSemanticForLLT(Ty.getScalarType());
  APFloat APFMultiplicand(Multiplicand);
  bool LosesInfo; // ignored
  APFMultiplicand.convert(Semantics, APFloat::rmNearestTiesToEven, &LosesInfo);

  auto K = B.buildFConstant(Ty, APFMultiplicand);
  auto Mul = B.buildFMul(Ty, Src, K, Flags);
  B.buildFExp2(Dst, Mul, Flags);
  MI.eraseFromParent();
  return true;
}

// GlobalISel doesn't currently have builtin support to legalize based on
// condition code like the SelectionDAG path does. We can move to that approach
// if and when it is available. For now, we custom legalize it based upon the
// approach in TargetLowering::LegalizeSetCCCondCode().
static bool legalizeGFcmp(MachineInstr &MI, MachineIRBuilder &B) {
  auto Pred = static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate());
  Register Dst = MI.getOperand(0).getReg();
  Register Op0 = MI.getOperand(2).getReg();
  Register Op1 = MI.getOperand(3).getReg();
  unsigned Flags = MI.getFlags();
  switch (Pred) {
  case CmpInst::FCMP_UNE:
  case CmpInst::FCMP_OEQ:
  case CmpInst::FCMP_OGT:
  case CmpInst::FCMP_OGE:
  case CmpInst::FCMP_OLT:
  case CmpInst::FCMP_OLE:
    // already legal
    break;
  case CmpInst::FCMP_ONE:
  case CmpInst::FCMP_UEQ: {
    // Without the explicit G_SEXT added here, the legalizer will typically
    // G_ANYEXT the G_FCMP compare result to i16. Given that a .reg destination
    // for fcmp is only available for 32-bit, we explicitly extend it here
    // so we can fold the resulting select into the fcmp.
    auto LHS =
        B.buildSExt(I32, B.buildFCmp(CmpInst::FCMP_OGT, I1, Op0, Op1, Flags));
    auto RHS =
        B.buildSExt(I32, B.buildFCmp(CmpInst::FCMP_OLT, I1, Op0, Op1, Flags));
    auto Result = B.buildOr(I32, LHS, RHS);
    if (Pred == CmpInst::FCMP_UEQ)
      Result = B.buildNot(I32, Result);
    B.buildICmp(CmpInst::ICMP_EQ, Dst, Result, B.buildConstant(I32, -1));
    MI.eraseFromParent();
    break;
  }
  case CmpInst::FCMP_ORD: {
    auto LHS =
        B.buildSExt(I32, B.buildFCmp(CmpInst::FCMP_OEQ, I1, Op0, Op0, Flags));
    auto RHS =
        B.buildSExt(I32, B.buildFCmp(CmpInst::FCMP_OEQ, I1, Op1, Op1, Flags));
    auto Result = B.buildAnd(I32, LHS, RHS);
    B.buildICmp(CmpInst::ICMP_EQ, Dst, Result, B.buildConstant(I32, -1));
    MI.eraseFromParent();
    break;
  }
  case CmpInst::FCMP_UNO: {
    // When checking if an op is NaN in OpenCL, the builtin generates an
    // fcmp.uno with a non-NaN constant (usually zero). In that case, we don't
    // need to generate two fcmps because only the non-const parameter is
    // relevant to this comparison
    auto Op0Cst = getFConstantVRegValWithLookThrough(Op0, *B.getMRI());
    bool Op0IsOrdConstant = Op0Cst && !Op0Cst.value().Value.isNaN();

    auto Op1Cst = getFConstantVRegValWithLookThrough(Op1, *B.getMRI());
    bool Op1IsOrdConstant = Op1Cst && !Op1Cst.value().Value.isNaN();

    if (Op0IsOrdConstant || Op1IsOrdConstant) {
      auto Reg = Op1IsOrdConstant ? Op0 : Op1;
      B.buildFCmp(CmpInst::FCMP_UNE, Dst, Reg, Reg, Flags);
    } else {
      // If the operands are both non-constant, we need to split this into two
      // fcmps to ensure it returns false if they are unequal
      auto LHS =
          B.buildSExt(I32, B.buildFCmp(CmpInst::FCMP_UNE, I1, Op0, Op0, Flags));
      auto RHS =
          B.buildSExt(I32, B.buildFCmp(CmpInst::FCMP_UNE, I1, Op1, Op1, Flags));
      auto Result = B.buildOr(I32, LHS, RHS);
      B.buildICmp(CmpInst::ICMP_EQ, Dst, Result, B.buildConstant(I32, -1));
    }
    MI.eraseFromParent();
    break;
  }
  case CmpInst::FCMP_UGT:
  case CmpInst::FCMP_UGE:
  case CmpInst::FCMP_ULT:
  case CmpInst::FCMP_ULE: {
    auto Cmp = B.buildSExt(I32, B.buildFCmp(FCmpInst::getInversePredicate(Pred),
                                            I1, Op0, Op1, Flags));
    auto Not = B.buildNot(I32, Cmp);
    B.buildICmp(CmpInst::ICMP_EQ, Dst, Not, B.buildConstant(I32, -1));
    MI.eraseFromParent();
    break;
  }
  default:
    llvm_unreachable("unknown predicate?");
  }
  return true;
}

static bool legalizeGTrunc(MachineInstr &MI, MachineIRBuilder &B) {
  [[maybe_unused]] auto &MRI = *B.getMRI();
  auto [Dst, DstTy, Src, SrcTy] = MI.getFirst2RegLLTs();
  if (DstTy.getSizeInBits() == 1) {
    // truncate ??? to i1
    // Since PISA does not support truncs to i1 (i8 is the minimum), we must
    // turn it into an i1 by using an icmp instruction.
    auto Zero = B.buildConstant(SrcTy, 0);
    auto One = B.buildConstant(SrcTy, 1);
    auto And = B.buildAnd(SrcTy, Src, One);
    B.buildICmp(CmpInst::ICMP_NE, Dst, And, Zero);
  } else {
    // truncate i128 to ???
    assert(SrcTy.getSizeInBits() == 128);
    auto Unmerge = B.buildUnmerge(I64, Src);
    if (DstTy.getSizeInBits() == 64)
      B.buildCopy(Dst, Unmerge.getReg(0));
    else
      B.buildTrunc(Dst, Unmerge.getReg(0));
  }
  MI.eraseFromParent();
  return true;
}

static bool legalizeGExt(MachineInstr &MI, MachineIRBuilder &B) {
  auto &MRI = *B.getMRI();
  auto [Dst, DstTy, Src, SrcTy] = MI.getFirst2RegLLTs();

  if (MRI.getType(Src).getSizeInBits() == 1) {
    // i8 = G_*EXT i1
    auto Zero = B.buildConstant(DstTy, 0);
    int64_t ExtendedVal = (MI.getOpcode() == TargetOpcode::G_SEXT) ||
                                  (MI.getOpcode() == TargetOpcode::G_ANYEXT)
                              ? -1
                              : 1;
    auto One = B.buildConstant(DstTy, ExtendedVal);
    B.buildSelect(Dst, Src, One, Zero);
  } else {
    // any G_*EXT where source and destination are byte size
    auto DstSize = DstTy.getScalarSizeInBits();
    auto SrcSize = SrcTy.getScalarSizeInBits();
    assert((DstSize % 8 == 0) && "destination size is not byte size");
    assert((SrcSize % 8 == 0) && "source size is not byte size");
    auto EltSize =
        ((DstSize % 32 == 0) && (SrcSize % 32 == 0))
            ? 32
            : (((DstSize % 16 == 0) && (SrcSize % 16 == 0)) ? 16 : 8);
    unsigned NumDstElts = DstSize / EltSize;
    unsigned NumSrcElts = SrcSize / EltSize;
    LLT EltTy = LLT::integer(EltSize);
    LLT VecDstTy = LLT::fixed_vector(NumDstElts, EltTy);

    auto VecZero = MRI.createGenericVirtualRegister(VecDstTy);
    SmallVector<APInt> Zeros(NumDstElts, APInt(EltSize, 0));
    B.buildBuildVectorConstant(VecZero, Zeros);

    Register VecSrc;
    if (NumSrcElts == 1) {
      SmallVector<Register> Ops(NumDstElts, Src);
      VecSrc = MRI.createGenericVirtualRegister(VecDstTy);
      B.buildBuildVector(VecSrc, Ops); // Splat scalar into vector
    } else {
      LLT VecSrcTy = LLT::fixed_vector(NumSrcElts, EltTy);
      VecSrc = MRI.createGenericVirtualRegister(VecSrcTy);
      B.buildBitcast(VecSrc, Src);
    }

    SmallVector<int> Mask;
    for (unsigned I = 0; I < NumDstElts; I++) {
      Mask.push_back((I < NumSrcElts) ? I
                                      : MRI.getType(VecSrc).getNumElements());
    }

    auto VecDst = MRI.createGenericVirtualRegister(VecDstTy);
    B.buildShuffleVector(VecDst, VecSrc, VecZero, Mask);

    if (MI.getOpcode() == TargetOpcode::G_SEXT) {
      auto CastReg = MRI.createGenericVirtualRegister(DstTy);
      auto ShiftReg = MRI.createGenericVirtualRegister(DstTy);
      auto ShiftAmt = B.buildConstant(I32, DstSize - SrcSize);
      B.buildBitcast(CastReg, VecDst);
      B.buildShl(ShiftReg, CastReg, ShiftAmt);
      B.buildAShr(Dst, ShiftReg, ShiftAmt);
    } else {
      B.buildBitcast(Dst, VecDst);
    }
  }
  MI.eraseFromParent();
  return true;
}

static bool legalizeGItofp(MachineInstr &MI, MachineIRBuilder &B) {
  auto [Dst, DstTy, Src, SrcTy] = MI.getFirst2RegLLTs();
  assert(SrcTy.isScalar() && SrcTy.getSizeInBits() == 1 &&
         "Unexpected source type");
  assert(DstTy.isScalar() && DstTy.getSizeInBits() == 16 &&
         "Unexpected destination type");

  unsigned Opc = MI.getOpcode();
  assert((Opc == TargetOpcode::G_SITOFP || Opc == TargetOpcode::G_UITOFP) &&
         "Unexpected instruction opcode");

  const fltSemantics &Semantics =
      DstTy == LLT::bfloat16() ? APFloat::BFloat() : APFloat::IEEEhalf();

  auto TrueVal =
      APFloat::getOne(Semantics, /*Negative=*/Opc == TargetOpcode::G_SITOFP);
  auto FalseVal = APFloat::getZero(Semantics);

  auto True = B.buildFConstant(DstTy, TrueVal);
  auto False = B.buildFConstant(DstTy, FalseVal);
  B.buildSelect(Dst, Src, True, False);
  MI.eraseFromParent();
  return true;
}

static void updateRegInDebugValue(Register OriginalVal, Register NewVal,
                                  MachineRegisterInfo &MRI) {
  llvm::SmallVector<MachineOperand *, 5> Opnds;
  for (auto &Instr : MRI.use_instructions(OriginalVal)) {
    if (!Instr.isDebugValue())
      continue;
    for (auto &Opnd : Instr.operands()) {
      if (Opnd.isReg() && Opnd.getReg() == OriginalVal)
        Opnds.push_back(&Opnd);
    }
  }
  for (auto *Opnd : Opnds)
    Opnd->setReg(NewVal);
  return;
}

static bool legalizeGExtload(MachineInstr &MI, MachineIRBuilder &B) {
  auto &LoadMI = cast<GExtLoad>(MI);
  Register DstReg = LoadMI.getDstReg();
  Register PtrReg = LoadMI.getPointerReg();
  LLT MemTy = LoadMI.getMMO().getMemoryType();

  // legalizer will create scalar type here, e.g. s16
  LLT EltTy = LLT::integer(MemTy.getScalarSizeInBits());
  MemTy = MemTy.isVector() ? LLT::fixed_vector(MemTy.getNumElements(), EltTy)
                           : EltTy;

  // Narrow load + extension: G_{S,Z}EXTLOAD(DstTy, ptr) ->
  //   %narrow = G_LOAD MemTy, ptr
  //   DstReg  = G_{S,Z}EXT DstTy, %narrow
  auto NarrowLoad = B.buildLoad(MemTy, PtrReg, LoadMI.getMMO());
  if (isa<GSExtLoad>(MI))
    B.buildSExt(DstReg, NarrowLoad);
  else
    B.buildZExt(DstReg, NarrowLoad);
  LoadMI.eraseFromParent();
  return true;
}

static bool legalizeGLoad(MachineInstr &MI, MachineIRBuilder &B,
                          LegalizerHelper &Helper) {
  auto &MRI = *B.getMRI();
  GISelChangeObserver &Observer = Helper.Observer;
  auto &ValMO = MI.getOperand(0);
  Register Val = ValMO.getReg();
  MachineMemOperand &MMO = **MI.memoperands_begin();
  unsigned AddressSpace = MMO.getAddrSpace();
  LLT CurTy = MRI.getType(Val);
  auto CurTySize = CurTy.getSizeInBits();

  if (!CurTy.isVector() && ((CurTySize % 8) != 0)) {
    // Widen sub-byte scalar load/store to multiple of 8 bits.
    auto NewSize = (CurTySize + 7) & ~7;
    auto NewTy = LLT::integer(NewSize);
    if (MI.getOpcode() == TargetOpcode::G_LOAD) {
      // For loads: widen the load and truncate result.
      Helper.widenScalar(MI, 0, NewTy);
      MI.memoperands()[0]->setType(NewTy);
    } else if (CurTySize > 1) {
      // For stores of i2+: fold through G_TRUNC/G_BITCAST chains to find a
      // byte-sized source, avoiding G_ANYEXT from sub-byte types (which can't
      // be legalized for sources wider than i1).
      Register SrcReg = Val;
      MachineInstr *Def = MRI.getVRegDef(SrcReg);
      while (Def &&
             (Def->getOpcode() == TargetOpcode::G_TRUNC ||
              Def->getOpcode() == TargetOpcode::G_BITCAST) &&
             MRI.getType(Def->getOperand(1).getReg()).getSizeInBits() <
                 NewSize) {
        SrcReg = Def->getOperand(1).getReg();
        Def = MRI.getVRegDef(SrcReg);
      }
      // If the walk stops on a cast, it is always a G_TRUNC: a G_BITCAST
      // preserves its operand's (sub-byte) width, so it can never be the def
      // with a source >= NewSize that ends the walk. A non-cast terminating
      // def (e.g. G_CONSTANT) is handled by the G_INSERT widening below.
      if (Def && Def->getOpcode() == TargetOpcode::G_TRUNC) {
        // Found wider source via the G_TRUNC/G_BITCAST chain
        Register WiderReg = Def->getOperand(1).getReg();
        LLT WiderTy = MRI.getType(WiderReg);
        Register StoreReg;
        if (WiderTy.getSizeInBits() == NewSize)
          StoreReg = WiderReg;
        else
          StoreReg = B.buildTrunc(NewTy, WiderReg).getReg(0);
        Observer.changingInstr(MI);
        ValMO.setReg(StoreReg);
        MMO.setType(NewTy);
        Observer.changedInstr(MI);
      } else {
        // No wider source found via G_TRUNC/G_BITCAST chain.
        // Use G_INSERT into undef to widen without G_ANYEXT (which can't be
        // legalized for non-byte-aligned sub-byte sources > i1).
        Register UndefReg = B.buildUndef(NewTy).getReg(0);
        Register WideReg = B.buildInsert(NewTy, UndefReg, Val, 0).getReg(0);
        Observer.changingInstr(MI);
        ValMO.setReg(WideReg);
        MMO.setType(NewTy);
        Observer.changedInstr(MI);
      }
    } else {
      // For stores of i1: G_ANYEXT from i1 is custom-legalized (produces
      // proper masking), so use Helper.widenScalar directly.
      Helper.widenScalar(MI, 0, NewTy);
      MI.memoperands()[0]->setType(NewTy);
    }
  } else if (CurTy.isVector() && (CurTy.getScalarSizeInBits() > 1) &&
             (CurTy.getScalarSizeInBits() < 8)) {
    // Vectors with sub-byte elements: bitcast to scalar, then widen to
    // multiple of 8 bits (minimum 8).
    // e.g. <2 x i4> -> i8, <2 x i2> -> i4 -> i8, <65 x i2> -> i130 -> i136
    unsigned ScalarSize = CurTySize;
    unsigned NewSize = std::max(8u, ((ScalarSize + 7) & ~7u));
    auto NewTy = LLT::integer(NewSize);
    if (MI.getOpcode() == TargetOpcode::G_LOAD) {
      Register NewVal = MRI.createGenericVirtualRegister(NewTy);
      Observer.changingInstr(MI);
      ValMO.setReg(NewVal);
      MMO.setType(NewTy);
      Observer.changedInstr(MI);
      B.setInsertPt(B.getMBB(), ++B.getInsertPt());
      if (ScalarSize == NewSize) {
        B.buildBitcast(Val, NewVal);
      } else {
        auto Trunc = B.buildTrunc(LLT::integer(ScalarSize), NewVal);
        B.buildBitcast(Val, Trunc);
      }
    } else {
      // Store: bitcast vector to scalar integer. If already byte-aligned,
      // update MI directly. Otherwise, let the legalizer re-process as
      // scalar sub-byte store on the next iteration.
      if (ScalarSize == NewSize) {
        Register NewVal = MRI.createGenericVirtualRegister(NewTy);
        B.buildBitcast(NewVal, Val);
        Observer.changingInstr(MI);
        ValMO.setReg(NewVal);
        MMO.setType(NewTy);
        Observer.changedInstr(MI);
      } else {
        // Vector total size is not byte-aligned. Bitcast to scalar and
        // insert into a wider byte-aligned integer using G_INSERT.
        // Upper bits are don't-care for stores.
        Register CastReg =
            B.buildBitcast(LLT::integer(ScalarSize), Val).getReg(0);
        Register UndefReg = B.buildUndef(NewTy).getReg(0);
        Register NewVal = B.buildInsert(NewTy, UndefReg, CastReg, 0).getReg(0);
        Observer.changingInstr(MI);
        ValMO.setReg(NewVal);
        MMO.setType(NewTy);
        Observer.changedInstr(MI);
      }
    }
  } else if (shouldWidenLoad(MI.getOpcode(), CurTy, AddressSpace,
                             MMO.getAlign().value() * 8)) {
    assert(CurTy.getScalarSizeInBits() == 32 &&
           "ShouldWidenLoad: Only 32-bit elements reach here");
    assert(
        (AddressSpace == static_cast<unsigned>(PISAAS::AddressSpace::GLOBAL) ||
         AddressSpace == static_cast<unsigned>(PISAAS::AddressSpace::SHARED) ||
         AddressSpace ==
             static_cast<unsigned>(PISAAS::AddressSpace::CONSTANT)) &&
        "ShouldWidenLoad: Only global,shared,constant loads should reach here");
    // Get alignment in bytes
    Align AlignInBytes = MMO.getAlign();
    Register OriginalVal = ValMO.getReg();
    Register NewVal;
    auto NumElts = CurTy.getNumElements();
    // If alignment is at least 8 bytes and number of elements is 5 or 6,
    // widen to 3 elements of i64. Otherwise, widen to 8 elements of i32.
    bool CanWidenToI64 = (AlignInBytes.value() >= 8) && (NumElts <= 6);
    auto NewTy = CanWidenToI64 ? LLT::fixed_vector(3, LLT::integer(64))
                               : LLT::fixed_vector(8, LLT::integer(32));
    auto VecN32Ty = LLT::fixed_vector(NumElts, LLT::integer(32));

    // Create the new widened load/store
    Observer.changingInstr(MI);
    NewVal = MRI.createGenericVirtualRegister(NewTy);
    MMO.setType(NewTy);
    ValMO.setReg(NewVal);
    Observer.changedInstr(MI);
    B.setInsertPt(B.getMBB(), ++B.getInsertPt());
    // DBG_VALUE should be associated with the original load
    updateRegInDebugValue(OriginalVal, NewVal, MRI);

    Register ResultReg;
    if (CanWidenToI64) {
      // Handle vectors with 5 or 6 elements of i32 with alignment >= 8 bytes
      // Legalize to 3 elements of i64 for better hardware utilization
      auto V6i32Ty = LLT::fixed_vector(6, LLT::integer(32));
      auto ExtrVal = MRI.createGenericVirtualRegister(V6i32Ty);
      // Bitcast to 6xi32 first, then extract the
      // first 5 elements
      B.buildBitcast(ExtrVal, NewVal);
      ResultReg = B.buildExtractSubvector(VecN32Ty, ExtrVal, 0).getReg(0);
    } else {
      // Handle vectors with 5, 6, or 7 elements of i32 with alignment <= 8
      // bytes or 7 elements. Expand them to 8 elements for better hardware
      // utilization
      assert(NumElts == 7 || AlignInBytes.value() < 8);
      // After loading the 8-element vector, extract the needed elements
      ResultReg = B.buildExtractSubvector(VecN32Ty, NewVal, 0).getReg(0);
    }
    if (CurTy.getScalarType().isPointer()) {
      B.buildIntToPtr(Val, ResultReg);
    } else {
      B.buildCopy(Val, ResultReg);
    }
  } else if (CurTy.getScalarType().isPointer()) {
    // load/store of ptr requires inttoptr/ptrtoint
    auto EltSize = CurTy.getScalarSizeInBits();
    LLT NewTy = CurTy.changeElementType(LLT::integer(EltSize));
    Register NewVal = MRI.createGenericVirtualRegister(NewTy);
    MMO.setType(NewTy);
    ValMO.setReg(NewVal);
    if (MI.getOpcode() == TargetOpcode::G_LOAD) {
      B.setInsertPt(B.getMBB(), ++B.getInsertPt());
      B.buildIntToPtr(Val, NewVal);
    } else {
      B.buildPtrToInt(NewVal, Val);
    }
  } else if (CurTy.getScalarSizeInBits() == 1) {
    auto BitSize = CurTy.getSizeInBits();
    auto NumEltsI8 = (BitSize + 7) / 8;
    auto NewBitSize = NumEltsI8 * 8;
    assert(CurTy.isVector() &&
           "Expected only vector of i1 to reach here, scalar was extended to "
           "i8 on widen scalars to be multiple of 8");
    bool NoExtensionNeeded = (BitSize == NewBitSize);
    LLT NewI8Ty = (NumEltsI8 > 1) ? LLT::fixed_vector(NumEltsI8, I8) : I8;
    Register NewI8Val = MRI.createGenericVirtualRegister(NewI8Ty);
    Register OriginalVal = ValMO.getReg();
    Observer.changingInstr(MI);
    MMO.setType(NewI8Ty);
    ValMO.setReg(NewI8Val);
    Observer.changedInstr(MI);
    if (MI.getOpcode() == TargetOpcode::G_LOAD) {
      updateRegInDebugValue(OriginalVal, NewI8Val, MRI);
      B.setInsertPt(B.getMBB(), ++B.getInsertPt());
      if (NoExtensionNeeded)
        B.buildBitcast(Val, NewI8Val);
      else {
        LLT NewI1Ty = LLT::fixed_vector(NewBitSize, I1);
        Register NewI1Val = MRI.createGenericVirtualRegister(NewI1Ty);
        B.buildBitcast(NewI1Val, NewI8Val);
        B.buildDeleteTrailingVectorElements(Val, NewI1Val);
      }
    } else {
      if (NoExtensionNeeded)
        B.buildBitcast(NewI8Val, Val);
      else {
        LLT NewI1Ty = LLT::fixed_vector(NewBitSize, I1);
        Register NewI1Val = MRI.createGenericVirtualRegister(NewI1Ty);
        B.buildPadVectorWithUndefElements(NewI1Val, Val);
        B.buildBitcast(NewI8Val, NewI1Val);
      }
    }
  } else {
    llvm_unreachable("unhandled load/store case");
  }
  return true;
}

static bool legalizeGFrem(MachineInstr &MI, MachineIRBuilder &B) {
  auto &MRI = *B.getMRI();
  Register DstReg = MI.getOperand(0).getReg();
  Register Src0Reg = MI.getOperand(1).getReg();
  Register Src1Reg = MI.getOperand(2).getReg();
  auto Flags = MI.getFlags();
  auto FmAfn = Flags & MachineInstr::FmAfn;
  LLT Ty = MRI.getType(DstReg);

  auto DivFlags = Flags;
  if (FmAfn) {
    DivFlags &= ~MachineInstr::FmAfn;
    DivFlags |= MachineInstr::FmArcp;
  }
  auto Div = B.buildFDiv(Ty, Src0Reg, Src1Reg, DivFlags);
  auto Trunc = B.buildIntrinsicTrunc(Ty, Div, Flags);
  auto Neg = B.buildFNeg(Ty, Trunc, Flags);
  if (!FmAfn) {
    auto FMA = B.buildFMA(Ty, Neg, Src1Reg, Src0Reg, Flags);

    auto &Semantics = getFltSemanticForLLT(Ty.getScalarType());
    auto InfC = B.buildFConstant(Ty, APFloat::getInf(Semantics));

    auto XAbs = B.buildIntrinsic(Intrinsic::pisa_fabs, {Ty})
                    .addUse(Src0Reg)
                    .setMIFlags(Flags);
    auto YAbs = B.buildIntrinsic(Intrinsic::pisa_fabs, {Ty})
                    .addUse(Src1Reg)
                    .setMIFlags(Flags);
    // Using pisa_fabs is safe here: the result is only compared against Inf
    // via OEQ, which is false for any NaN regardless of signaling/quiet.
    auto XFCmp = B.buildFCmp(FCmpInst::FCMP_OEQ, I1, XAbs, InfC, Flags);
    auto YFCmp = B.buildFCmp(FCmpInst::FCMP_OEQ, I1, YAbs, InfC, Flags);
    auto Sel = B.buildSelect(Ty, YFCmp, Src0Reg, FMA);
    B.buildSelect(DstReg, XFCmp, FMA, Sel);
  } else {
    B.buildFMA(DstReg, Neg, Src1Reg, Src0Reg, Flags);
  }
  MI.eraseFromParent();
  return true;
}

// IEEE 754 fabs: clear the sign bit via bitwise AND.
// This is used when NaN inputs cannot be ruled out, ensuring the sign bit
// is cleared without quieting the NaN (unlike the PISA fabs instruction).
static bool legalizeFAbs(MachineInstr &MI, MachineIRBuilder &B) {
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  MachineRegisterInfo &MRI = *B.getMRI();
  LLT Ty = MRI.getType(DstReg);

  // <2 x half> / <2 x bfloat>: pack into a single 32-bit AND.
  if (Ty.isVector() && Ty.getNumElements() == 2 &&
      Ty.getScalarSizeInBits() == 16) {
    auto Src32 = B.buildBitcast(I32, SrcReg);
    // 0x7FFF7FFF: clears the sign bit of each 16-bit element.
    auto Mask = B.buildConstant(I32, 0x7FFF7FFF);
    auto And = B.buildAnd(I32, Src32, Mask);
    B.buildBitcast(DstReg, And);
    MI.eraseFromParent();
    return true;
  }

  unsigned BitWidth = Ty.getSizeInBits();

  // Sign bit mask: all ones except the MSB.
  APInt Mask = APInt::getSignedMaxValue(BitWidth);
  LLT IntTy = LLT::integer(BitWidth);

  // G_AND is only legal on (any-)scalar integer LLTs, so for typed-float
  // operands we bitcast through the integer LLT. LLT::operator== treats an
  // any-scalar as equal to a typed float of the same width, so distinguish
  // by isFloat() instead of by inequality.
  bool IsTypedFloat = Ty.isFloat();

  Register IntSrc = SrcReg;
  if (IsTypedFloat)
    IntSrc = B.buildBitcast(IntTy, SrcReg).getReg(0);

  auto MaskCst = B.buildConstant(IntTy, Mask);
  auto And = B.buildAnd(IntTy, IntSrc, MaskCst);

  if (IsTypedFloat)
    B.buildBitcast(DstReg, And);
  else
    B.buildCopy(DstReg, And);

  MI.eraseFromParent();
  return true;
}

// Support for bf/hf type is limited to fdiv.fast
// If non-afn division is requested, we extend args to float,
// perform the division and truncate the result back to hf/bf
static bool legalizeGFdiv(MachineInstr &MI, MachineIRBuilder &B) {
  // natively supported
  if (MI.getFlag(MachineInstr::FmArcp))
    return true;

  // perform converts
  auto &MRI = *B.getMRI();
  auto [DstReg, Src0Reg, Src1Reg] = MI.getFirst3Regs();

  auto Src0Tmp = MRI.createGenericVirtualRegister(F32);
  auto Src1Tmp = MRI.createGenericVirtualRegister(F32);
  auto DstTmp = MRI.createGenericVirtualRegister(F32);

  auto FPExt0 = B.buildFPExt(Src0Tmp, Src0Reg);
  auto FPExt1 = B.buildFPExt(Src1Tmp, Src1Reg);
  auto FDiv = B.buildFDiv(DstTmp, FPExt0, FPExt1);
  B.buildFPTrunc(DstReg, FDiv);

  MI.eraseFromParent();
  return true;
}

static bool legalizeGInsertVectorElt(LegalizerHelper &Helper, MachineInstr &MI,
                                     MachineIRBuilder &B) {
  auto &MRI = *B.getMRI();
  Register SrcReg = MI.getOperand(1).getReg();
  Register EltReg = MI.getOperand(2).getReg();
  Register IndexReg = MI.getOperand(3).getReg();

  LLT SrcTy = MRI.getType(SrcReg);
  LLT EltTy = MRI.getType(EltReg);
  LLT IndexTy = MRI.getType(IndexReg);

  int EltSize = EltTy.getSizeInBits();
  int NumElts = SrcTy.getNumElements();

  if (EltSize == 1) {
    // Handle insertion to <n x i1>
    int Size = MRI.getType(SrcReg).getSizeInBits();
    assert(isPowerOf2_32(Size) && "need to extend source to be power of 2");

    LLT ScalarTy = LLT::integer(Size);

    auto SScalarReg = MRI.createGenericVirtualRegister(ScalarTy);
    auto DScalarReg = MRI.createGenericVirtualRegister(ScalarTy);
    auto MaskReg = MRI.createGenericVirtualRegister(ScalarTy);
    auto NotReg = MRI.createGenericVirtualRegister(ScalarTy);
    auto AndReg = MRI.createGenericVirtualRegister(ScalarTy);
    auto ShiftReg = MRI.createGenericVirtualRegister(ScalarTy);
    auto EltZExtReg = MRI.createGenericVirtualRegister(ScalarTy);
    auto ShiftAmountReg = IndexReg;

    B.buildBitcast(SScalarReg, SrcReg);
    auto Value = getIConstantVRegValWithLookThrough(IndexReg, MRI);

    if (Value.has_value()) { // constant index
      B.buildConstant(MaskReg, 1ull << Value->Value.getZExtValue());
      ShiftAmountReg = MRI.createGenericVirtualRegister(ScalarTy);
      B.buildConstant(ShiftAmountReg, Value->Value.getZExtValue());
    } else { // non-constant index
      auto ConstReg = MRI.createGenericVirtualRegister(ScalarTy);
      B.buildConstant(ConstReg, 1ull);
      B.buildShl(MaskReg, ConstReg, IndexReg);
    }

    B.buildNot(NotReg, MaskReg);
    B.buildAnd(AndReg, SScalarReg, NotReg);
    B.buildZExt(EltZExtReg, EltReg);
    B.buildShl(ShiftReg, EltZExtReg, ShiftAmountReg);
    B.buildOr(DScalarReg, AndReg, ShiftReg);
    B.buildBitcast(MI.getOperand(0), DScalarReg);
    MI.eraseFromParent();
    return true;
  }

  assert((EltSize == 8 || EltSize == 16 || EltSize == 64) &&
         "unexpected element size");

  // If the index is constant, narrow the vector down to 4 elements.
  if (auto MaybeValue = getIConstantVRegValWithLookThrough(IndexReg, MRI)) {
    if (NumElts <= 4) {
      SmallVector<Register, 4> Elements;
      for (int I = 0; I < NumElts; ++I)
        Elements.push_back(MRI.createGenericVirtualRegister(EltTy));

      B.buildUnmerge(Elements, SrcReg);
      Elements[MaybeValue->Value.getZExtValue()] = EltReg;
      B.buildBuildVector(MI.getOperand(0), Elements);
      MI.eraseFromParent();
      return true;
    }

    // Narrow to 4 elements.
    auto Res = Helper.fewerElementsVector(MI, 0, LLT::fixed_vector(4, EltTy));
    return Res != LegalizerHelper::UnableToLegalize;
  }

  if (EltSize <= 16) {
    // Handle insertion to <n x i8> and <n x i16>, where the vector size is not
    // a multiple of 32 bits. The vector is extended to the next multiple of 32
    // bits, and then bitcast to a vector of i32 for the insertion. The vector
    // cannot be narrowed here because the index is not constant, and we don't
    // know which elements will be inserted.
    if (NumElts * EltSize % 32 != 0) {
      int NewNumElts = alignTo(NumElts, 32 / EltSize);
      LLT NewVecTy = LLT::fixed_vector(NewNumElts, EltTy);
      auto Res = Helper.moreElementsVector(MI, 0, NewVecTy);
      if (Res == LegalizerHelper::UnableToLegalize)
        return false;
      NumElts = NewNumElts;
      B.setInsertPt(*MI.getParent(), MI);
    }

    // Now the vector size is a multiple of 32 bits, we can bitcast to a vector
    // of s32 and insert the element.
    // When the element type is float (e.g. f16), bitcastInsertVectorElt will
    // emit G_ZEXT on the element, but G_ZEXT of a float type is invalid.
    // Work around by converting to integer types, delegating, and bitcasting
    // back.
    Register OrigDst = MI.getOperand(0).getReg();
    Register IntDst;
    bool NeedFloatBitcast = EltTy.isFloat();
    if (NeedFloatBitcast) {
      LLT IntEltTy = LLT::integer(EltSize);
      LLT IntVecTy = LLT::fixed_vector(NumElts, IntEltTy);

      // Bitcast element from float to integer.
      Register IntElt = MRI.createGenericVirtualRegister(IntEltTy);
      B.buildBitcast(IntElt, MI.getOperand(2).getReg());
      MI.getOperand(2).setReg(IntElt);

      // Bitcast source vector to integer element type.
      Register IntSrc = MRI.createGenericVirtualRegister(IntVecTy);
      B.buildBitcast(IntSrc, MI.getOperand(1).getReg());
      MI.getOperand(1).setReg(IntSrc);

      // Replace destination with integer vector type.
      IntDst = MRI.createGenericVirtualRegister(IntVecTy);
      MI.getOperand(0).setReg(IntDst);
    }
    int NewNumElts = NumElts * EltSize / 32;
    LLT NewVecTy = NewNumElts == 1 ? I32 : LLT::fixed_vector(NewNumElts, I32);
    auto Res = Helper.bitcastInsertVectorElt(MI, 0, NewVecTy);
    if (NeedFloatBitcast && Res != LegalizerHelper::UnableToLegalize) {
      MachineInstr *DefMI = MRI.getVRegDef(IntDst);
      B.setInsertPt(*DefMI->getParent(), std::next(DefMI->getIterator()));
      B.buildBitcast(OrigDst, IntDst);
    }
    return Res != LegalizerHelper::UnableToLegalize;
  }

  assert(EltSize == 64 && "unexpected element size");
  LLT NewVecTy = LLT::fixed_vector(NumElts * 2, I32);

  // Compute the low and high indices as low = index * 2, high = low + 1
  auto One = B.buildConstant(IndexTy, 1).getReg(0);
  auto LowIndexReg = B.buildShl(IndexTy, IndexReg, One).getReg(0);
  auto HighIndexReg = B.buildAdd(IndexTy, LowIndexReg, One).getReg(0);

  // Split the 64-bit element into two 32-bit elements
  auto EltLowReg = MRI.createGenericVirtualRegister(I32);
  auto EltHighReg = MRI.createGenericVirtualRegister(I32);
  B.buildUnmerge({EltLowReg, EltHighReg}, EltReg);

  // Bitcast the source vector to s32 vector
  auto BitcastSrcReg = B.buildBitcast(NewVecTy, SrcReg).getReg(0);

  // Insert the low and high parts
  auto InsertLowReg = B.buildInsertVectorElement(NewVecTy, BitcastSrcReg,
                                                 EltLowReg, LowIndexReg)
                          .getReg(0);
  auto InsertHighReg = B.buildInsertVectorElement(NewVecTy, InsertLowReg,
                                                  EltHighReg, HighIndexReg)
                           .getReg(0);

  // Bitcast back to the original vector type
  B.buildBitcast(MI.getOperand(0), InsertHighReg);
  MI.eraseFromParent();
  return true;
}

static bool legalizeGExtractVectorElt(LegalizerHelper &Helper, MachineInstr &MI,
                                      MachineIRBuilder &B) {
  auto &MRI = *B.getMRI();
  Register VecReg = MI.getOperand(1).getReg();
  LLT VecTy = MRI.getType(VecReg);
  LLT EltTy = VecTy.getScalarType();
  int EltSize = VecTy.getScalarSizeInBits();
  int NumElts = VecTy.getNumElements();

  if (EltSize == 1) {
    // Handle extraction of <n x i1>
    auto Size = MRI.getType(VecReg).getSizeInBits();
    auto CastReg = MRI.createGenericVirtualRegister(LLT::integer(Size));
    B.buildBitcast(CastReg, VecReg);
    auto ShiftReg = MRI.createGenericVirtualRegister(LLT::integer(Size));
    B.buildLShr(ShiftReg, CastReg, MI.getOperand(2));
    B.buildTrunc(MI.getOperand(0), ShiftReg);
    MI.eraseFromParent();
    return true;
  }

  assert((EltSize == 8 || EltSize == 16 || EltSize == 64) &&
         "unexpected element size");

  // If the index is constant, narrow the vector down to 4 elements.
  Register IndexReg = MI.getOperand(2).getReg();
  if (auto MaybeValue = getIConstantVRegValWithLookThrough(IndexReg, MRI)) {
    if (NumElts <= 4) {
      Register UnmergeReg = B.buildUnmerge(EltTy, VecReg)
                                .getReg(MaybeValue->Value.getZExtValue());
      B.buildCopy(MI.getOperand(0), UnmergeReg);
      MI.eraseFromParent();
      return true;
    }

    // Narrow to 4 elements.
    auto Res = Helper.fewerElementsVector(MI, 1, LLT::fixed_vector(4, EltTy));
    return Res != LegalizerHelper::UnableToLegalize;
  }

  // Handle extraction from <n x i8> and <n x i16>, where the vector size is not
  // a multiple of 32 bits. The vector is extended to the next multiple of 32
  // bits, and then bitcast to a vector of i32 for the extraction. The vector
  // cannot be narrowed here because the index is not constant, and we don't
  // know which elements will be extracted.
  if (NumElts * EltSize % 32 != 0) {
    int NewNumElts = alignTo(NumElts, 32 / EltSize);
    LLT NewVecTy = LLT::fixed_vector(NewNumElts, EltTy);
    auto Res = Helper.moreElementsVector(MI, 1, NewVecTy);
    if (Res == LegalizerHelper::UnableToLegalize)
      return false;
    NumElts = NewNumElts;
  }

  // Now the vector size is a multiple of 32 bits, we can bitcast to a vector
  // of s32 and extract the element.
  // When the element type is float (e.g. f16), bitcastExtractVectorElt will
  // emit G_TRUNC to the original element type, but G_TRUNC to a float type is
  // invalid. Work around this by replacing the destination with an integer
  // type, delegating to the helper, and then bitcasting back to float.
  Register OrigDst = MI.getOperand(0).getReg();
  Register IntDst;
  bool NeedFloatBitcast = EltTy.isFloat();
  if (NeedFloatBitcast) {
    LLT IntEltTy = LLT::integer(EltSize);
    IntDst = MRI.createGenericVirtualRegister(IntEltTy);
    MI.getOperand(0).setReg(IntDst);
    // Also patch the source vector to integer element type so bitcast is valid.
    LLT IntVecTy = LLT::fixed_vector(NumElts, IntEltTy);
    Register IntVec = MRI.createGenericVirtualRegister(IntVecTy);
    B.buildBitcast(IntVec, MI.getOperand(1).getReg());
    MI.getOperand(1).setReg(IntVec);
    VecTy = IntVecTy;
  }
  int NewNumElts = NumElts * EltSize / 32;
  LLT NewVecTy = NewNumElts == 1 ? I32 : LLT::fixed_vector(NewNumElts, I32);
  auto Res = Helper.bitcastExtractVectorElt(MI, 1, NewVecTy);
  if (NeedFloatBitcast && Res != LegalizerHelper::UnableToLegalize) {
    // MI has been erased by the helper. IntDst now has an integer-typed def
    // from the helper's lowered sequence. Bitcast it back to the original
    // float type. Reset the insert point since MI was erased.
    MachineInstr *DefMI = MRI.getVRegDef(IntDst);
    B.setInsertPt(*DefMI->getParent(), std::next(DefMI->getIterator()));
    B.buildBitcast(OrigDst, IntDst);
  }
  return Res != LegalizerHelper::UnableToLegalize;
}

static bool legalizeGBswap(MachineInstr &MI, MachineIRBuilder &B) {
  auto &MRI = *B.getMRI();
  auto [Dst, Src] = MI.getFirst2Regs();
  const LLT Ty = MRI.getType(Src);
  unsigned BitSize = Ty.getScalarSizeInBits();

  assert(BitSize % 16 == 0 && "bswap only supported for multiples of 16 bits");

  // Masks for byte swapping
  static const std::array<int, 2> SwapMask16 = {1, 0};
  static const std::array<int, 4> SwapMask32 = {3, 2, 1, 0};

  // Helper lambda for byte-swapping: returns a tuple describing how to swap
  // bytes within each chunk. The tuple contains:
  // - ChunkSize: the size in bits of each chunk (either 16 or 32).
  // - ChunkByteSwapMask: the shuffle mask used to reverse the byte order within
  // a chunk.
  // - ChunkShuffleVecTy: the vector type used for shuffling bytes within a
  // chunk.
  auto GetSwapProps =
      [&](unsigned BitSize) -> std::tuple<unsigned, ArrayRef<int>, LLT> {
    return (BitSize % 32 == 0)
               ? std::make_tuple(32u, ArrayRef<int>(SwapMask32), V4I8)
               : std::make_tuple(16u, ArrayRef<int>(SwapMask16), V2I8);
  };

  auto [ChunkSize, ChunkByteSwapMask, ChunkShuffleVecTy] =
      GetSwapProps(BitSize);

  // For Src types that are multiples of 32 bits, the value is divided into
  // 32-bit chunks. Each chunk is byte-swapped, and the resulting chunks are
  // built into a vector in reverse order. For Src types that are multiples of
  // 16 bits (but not 32), the value is divided into 16-bit chunks. Each chunk
  // is byte-swapped and reassembled in reverse order.
  if (BitSize == 16 || BitSize == 32) {
    assert(ChunkSize == BitSize &&
           "Single chunk case: ChunkSize must equal BitSize");
    auto VecReg = B.buildBitcast(ChunkShuffleVecTy, Src);
    auto ShufReg = B.buildShuffleVector(ChunkShuffleVecTy, VecReg, VecReg,
                                        ChunkByteSwapMask);
    B.buildBitcast(Dst, ShufReg);
  } else {
    unsigned NumChunks = BitSize / ChunkSize;
    LLT ChunkTy = LLT::integer(ChunkSize);
    LLT VecTy = LLT::fixed_vector(NumChunks, ChunkTy);
    auto VecReg = B.buildBitcast(VecTy, Src);

    SmallVector<Register, 8> SwappedChunks;
    for (int I = NumChunks - 1; I >= 0; --I) {
      auto Index = B.buildConstant(I32, I);
      auto ChunkReg = B.buildExtractVectorElement(ChunkTy, VecReg, Index);
      auto ChunkVec = B.buildBitcast(ChunkShuffleVecTy, ChunkReg);
      auto SwappedVec = B.buildShuffleVector(ChunkShuffleVecTy, ChunkVec,
                                             ChunkVec, ChunkByteSwapMask);
      auto SwappedChunk = B.buildBitcast(ChunkTy, SwappedVec);
      SwappedChunks.push_back(SwappedChunk.getReg(0));
    }

    auto FinalVec = B.buildBuildVector(VecTy, SwappedChunks);
    B.buildBitcast(Dst, FinalVec);
  }

  MI.eraseFromParent();
  return true;
}

static bool legalizeGFpow(MachineInstr &MI, MachineIRBuilder &B) {
  auto &MRI = *B.getMRI();

  auto [Dst, Src0, Src1] = MI.getFirst3Regs();
  auto DstTy = MRI.getType(Dst);
  assert(DstTy.isScalar() &&
         (DstTy.getSizeInBits() == 32 || DstTy.getSizeInBits() == 16));

  unsigned Flags = MI.getFlags();

  // can only do approximation of pow()
  auto AllowApprox = MI.getFlag(MachineInstr::FmAfn);
  if (!AllowApprox)
    llvm_unreachable("not implemented (fpow)");

  auto LogReg = MRI.createGenericVirtualRegister(DstTy);
  auto MulReg = MRI.createGenericVirtualRegister(DstTy);
  auto FExp2Reg = Dst;

  B.buildFLog2(LogReg, Src0, Flags);
  B.buildFMul(MulReg, LogReg, Src1, Flags);
  B.buildFExp2(FExp2Reg, MulReg, Flags);

  MI.eraseFromParent();
  return true;
}

static bool legalizeGFldexp(MachineInstr &MI, MachineIRBuilder &B) {
  auto &MRI = *B.getMRI();
  auto [Dst, Src0, Src1] = MI.getFirst3Regs();
  unsigned Flags = MI.getFlags();

  LLT XTy = MRI.getType(Src0);
  LLT NTy = MRI.getType(Src1);
  LLT Src1Ty = MRI.getType(Src1);

  auto AllowApprox =
      XTy.getSizeInBits() <= 32 && MI.getFlag(MachineInstr::FmAfn);
  auto IsBFloat16 = XTy == LLT::bfloat16();
  if (AllowApprox) {
    auto RegLLT = MRI.getType(Dst);
    auto FpReg = MRI.createGenericVirtualRegister(RegLLT);
    auto ExpReg = MRI.createGenericVirtualRegister(RegLLT);

    B.buildSITOFP(FpReg, Src1);
    B.buildFExp2(ExpReg, FpReg, Flags);
    B.buildFMul(Dst, Src0, ExpReg, Flags);

    MI.eraseFromParent();
    return true;
  }

  int NClampRangeVal, NShiftVal, NDivBy3ShiftVal, NDivBy3MulVal;
  if (XTy.getSizeInBits() == 16 && !IsBFloat16) {
    NClampRangeVal = 14;
    NShiftVal = 10;
    NDivBy3ShiftVal = 8;
    NDivBy3MulVal = 0x56;
  } else if (XTy.getSizeInBits() == 32 || IsBFloat16) {
    NClampRangeVal = 126;
    NShiftVal = IsBFloat16 ? 7 : 23;
    NDivBy3ShiftVal = 16;
    NDivBy3MulVal = 0x5556;
  } else {
    // double precision
    NClampRangeVal = 1022;
    NShiftVal = 52;
    NDivBy3ShiftVal = 16;
    NDivBy3MulVal = 0x5556;
  }

  // Limit range of n (such that all inputs can be handled correctly)
  // For FP32, |n|>128+126+23 will definitely lead to overflow/underflow
  // |n|<=126*3 is a sufficiently wide range for n (and FP32 x)
  // For FP64, |n|>1024+1022+52 will definitely lead to overflow/underflow
  // |n|<=1022*3 is a sufficiently wide range for n (and FP64 x)

  auto ClampMax = B.buildConstant(NTy, -NClampRangeVal * 3);
  auto NClampedMax = B.buildSMax(NTy, Src1, ClampMax);
  auto ClampMin = B.buildConstant(NTy, NClampRangeVal * 3);
  auto NClamped = B.buildSMin(NTy, NClampedMax, ClampMin);

  auto AddConst = B.buildConstant(NTy, (NClampRangeVal + 1) * 3);
  auto N = B.buildAdd(NTy, NClamped, AddConst);
  if (XTy.getSizeInBits() == 16 && !IsBFloat16) {
    NTy = I16;
    N = B.buildTrunc(NTy, N);
  }

  // for fp16, n/3 performed as a 8x8-bit->16-bit integer MUL and SHR by 8.
  // for others, n/3, performed as a 16x16-bit->32-bit integer MUL and SHR by 16
  // (both LSHR or ASHR work, n is positive at this point)
  auto MulConst = B.buildConstant(NTy, NDivBy3MulVal);
  auto NMul = B.buildMul(NTy, N, MulConst);
  auto ShrConst = B.buildConstant(I32, NDivBy3ShiftVal);
  auto K0 = B.buildLShr(NTy, NMul, ShrConst);

  auto NMinusK0 = B.buildSub(NTy, N, K0);
  auto K1 = B.buildSub(NTy, NMinusK0, K0);

  if (XTy.getSizeInBits() == 64) {
    NTy = I64;
    K0 = B.buildZExt(NTy, K0);
    K1 = B.buildZExt(NTy, K1);
  } else if (IsBFloat16) {
    NTy = I16;
    K0 = B.buildTrunc(NTy, K0);
    K1 = B.buildTrunc(NTy, K1);
  }

  auto ShlConst = B.buildConstant(I32, NShiftVal);
  auto SK0I = B.buildShl(NTy, K0, ShlConst);
  auto SK1I = B.buildShl(NTy, K1, ShlConst);
  auto SK0 = B.buildBitcast(XTy, SK0I);
  auto SK1 = B.buildBitcast(XTy, SK1I);

  SrcOp SwapperX(Src0), SwapperSK1(SK1);
  if (XTy.getSizeInBits() > 16) {
    // Swap Src0 with SK1 if n is sufficiently small for SK1 * SK0 * SK0 not to
    // overflow (inf). This prevents a potential underflow that can happen with
    // Src0 * SK0 * SK0.
    int SmallThresholdVal = NClampRangeVal / 3;
    auto SmallThresholdConst = B.buildConstant(Src1Ty, SmallThresholdVal);
    auto Src1Abs = B.buildAbs(Src1Ty, NClamped);
    auto IsSrc1Small = B.buildICmp(CmpInst::Predicate::ICMP_SLT, I1, Src1Abs,
                                   SmallThresholdConst);

    SwapperX = B.buildSelect(XTy, IsSrc1Small, SK1, Src0);
    SwapperSK1 = B.buildSelect(XTy, IsSrc1Small, Src0, SK1);
  }

  auto Res0 = B.buildFMul(XTy, SwapperX, SK0, Flags);
  auto Res1 = B.buildFMul(XTy, Res0, SK0, Flags);
  B.buildFMul(Dst, Res1, SwapperSK1, Flags);

  MI.eraseFromParent();
  return true;
}

// PISA specification provides no support for 16bit fsqrt with rounding mode
// - extend to 32bit value
// - perform square root with rounding mode
// - truncate to 16bit value
static bool legalizeIntrinsicFSqrt(LegalizerHelper &Helper, MachineInstr &MI) {
  MachineIRBuilder &B = Helper.MIRBuilder;
  auto &MRI = *B.getMRI();

  SmallVector<MachineInstr *, 4> MIs;
  Intrinsic::ID IntrinsicID = cast<GIntrinsic>(MI).getIntrinsicID();
  if (MRI.getType(MI.getOperand(0).getReg()).isVector()) {
    MIs = scalarizeIntrinsic(MI);
  } else {
    MIs.push_back(&MI);
  }

  for (auto It = MIs.begin(), Ite = MIs.end(); It != Ite; ++It) {
    auto *MI = *It;
    MachineIRBuilder MIB(*MI);

    auto Dst = MI->getOperand(0).getReg();
    auto Src = MI->getOperand(2).getReg();
    auto Imm = MI->getOperand(3).getImm();

    if (MRI.getType(Dst).getScalarSizeInBits() != 16)
      continue; // already legal

    auto Src32 = MRI.createGenericVirtualRegister(F32);
    auto Dst32 = MRI.createGenericVirtualRegister(F32);
    MIB.buildFPExt(Src32, Src);
    MIB.buildIntrinsic(IntrinsicID, Dst32).addReg(Src32).addImm(Imm);
    MIB.buildFPTrunc(Dst, Dst32);
    MI->eraseFromParent();
  }
  return true;
}

static bool legalizeIntrinsicFDiv(LegalizerHelper &Helper, MachineInstr &MI) {
  MachineIRBuilder &B = Helper.MIRBuilder;
  auto &MRI = *B.getMRI();

  Intrinsic::ID IntrinsicID = cast<GIntrinsic>(MI).getIntrinsicID();
  SmallVector<MachineInstr *> MIs = scalarizeIntrinsic(MI);

  // fdiv only supports 32/64 width
  for (auto It = MIs.begin(), Ite = MIs.end(); It != Ite; ++It) {
    auto *MI = *It;
    MachineIRBuilder MIB(*MI);

    auto Dst = MI->getOperand(0).getReg();
    auto Src0 = MI->getOperand(2).getReg();
    auto Src1 = MI->getOperand(3).getReg();

    if (MRI.getType(Dst).getScalarSizeInBits() != 16)
      continue;

    // s16 A = FDIV s16 B, s16 C
    // => s32 B' = FEXT s16 B
    // => s32 C' = FEXT s16 C
    // => s32 A' = FDIV s32 B', s32 C'
    // => s16 A = FTRUNC s32 A'
    auto Src032 = MRI.createGenericVirtualRegister(F32);
    auto Src132 = MRI.createGenericVirtualRegister(F32);
    auto Dst32 = MRI.createGenericVirtualRegister(F32);
    MIB.buildFPExt(Src032, Src0);
    MIB.buildFPExt(Src132, Src1);
    MIB.buildIntrinsic(IntrinsicID, Dst32)
        .addReg(Src032)
        .addReg(Src132)
        .add(MI->getOperand(4));
    MIB.buildFPTrunc(Dst, Dst32);

    MI->eraseFromParent();
  }
  return true;
}

static SmallVector<Register> splitVectorByGrain(MachineIRBuilder &B,
                                                Register Src, unsigned Grain) {
  auto &MRI = *B.getMRI();
  auto SrcTy = MRI.getType(Src);
  auto EltTy = SrcTy.getScalarType();

  auto SliceTy = LLT::fixed_vector(Grain, EltTy);

  if (SrcTy.isScalar()) {
    auto SliceUndef = MRI.createGenericVirtualRegister(SliceTy);
    auto Slice = MRI.createGenericVirtualRegister(SliceTy);
    B.buildUndef(SliceUndef);
    B.buildInsertVectorElement(Slice, SliceUndef, Src, B.buildConstant(I32, 0));
    return {Slice};
  }

  const unsigned NumElts = SrcTy.getNumElements();

  SmallVector<Register> Elts;
  for (unsigned I = 0; I < NumElts; I += Grain) {
    auto Slice = MRI.createGenericVirtualRegister(SliceTy);
    B.buildUndef(Slice);

    for (unsigned J = 0; J < std::min(Grain, NumElts - I); ++J) {
      auto Idx = B.buildConstant(I32, J).getReg(0);
      auto Elt =
          B.buildExtractVectorElementConstant(EltTy, Src, I + J).getReg(0);
      Slice = B.buildInsertVectorElement(SliceTy, Slice, Elt, Idx).getReg(0);
    }
    Elts.push_back(Slice);
  }
  return Elts;
}

static void joinVectorByGrain(MachineIRBuilder &B, Register Dst,
                              ArrayRef<Register> Srcs, unsigned Grain) {
  auto &MRI = *B.getMRI();
  auto DstTy = MRI.getType(Dst);
  auto EltTy = DstTy.getScalarType();

  if (DstTy.isScalar()) {
    auto Src = Srcs[0];
    B.buildExtractVectorElementConstant(Dst, Src, 0);
    return;
  }

  const unsigned NumElts = DstTy.getNumElements();

  auto TmpDst = B.buildUndef(DstTy).getReg(0);

  for (unsigned I = 0; I < NumElts; I += Grain) {
    auto &Src = Srcs[I / Grain];

    for (unsigned J = 0; J < std::min(Grain, NumElts - I); ++J) {
      auto Idx = B.buildConstant(I32, I + J).getReg(0);
      auto Elt = B.buildExtractVectorElementConstant(EltTy, Src, J).getReg(0);

      TmpDst = B.buildInsertVectorElement(DstTy, TmpDst, Elt, Idx).getReg(0);
    }
  }

  B.buildCopy(Dst, TmpDst);
}

static bool legalizeIntrinsicBfn(LegalizerHelper &Helper, MachineInstr &MI) {
  MachineIRBuilder &B = Helper.MIRBuilder;
  auto &MRI = *B.getMRI();
  auto IntrinsicID = cast<GIntrinsic>(MI).getIntrinsicID();
  assert(IntrinsicID == Intrinsic::pisa_bfn);

  const auto BfnOpcode = MI.getOperand(2);

  const auto OrigDst = MI.getOperand(0).getReg();
  const auto OrigSrc0 = MI.getOperand(3).getReg();
  const auto OrigSrc1 = MI.getOperand(4).getReg();
  const auto OrigSrc2 = MI.getOperand(5).getReg();

  const auto Ty = MRI.getType(OrigDst);

  const auto BitWidth = Ty.getScalarSizeInBits();
  switch (BitWidth) {
  default:
    llvm_unreachable("unexpected bitwidth");
  case 8:
  case 16: {
    const auto Grain = 32 / BitWidth;
    MachineIRBuilder MIB(MI);

    auto Srcs0 = splitVectorByGrain(MIB, OrigSrc0, Grain);
    auto Srcs1 = splitVectorByGrain(MIB, OrigSrc1, Grain);
    auto Srcs2 = splitVectorByGrain(MIB, OrigSrc2, Grain);

    auto GrainTy = MRI.getType(Srcs0[0]);
    SmallVector<Register> Dsts;

    for (auto [Src0, Src1, Src2] : zip(Srcs0, Srcs1, Srcs2)) {
      auto Dst = MRI.createGenericVirtualRegister(I32);
      auto Src0Cast = MRI.createGenericVirtualRegister(I32);
      auto Src1Cast = MRI.createGenericVirtualRegister(I32);
      auto Src2Cast = MRI.createGenericVirtualRegister(I32);

      auto DstCast = MRI.createGenericVirtualRegister(GrainTy);

      MIB.buildBitcast(Src0Cast, Src0);
      MIB.buildBitcast(Src1Cast, Src1);
      MIB.buildBitcast(Src2Cast, Src2);

      MIB.buildIntrinsic(IntrinsicID, Dst)
          .add(BfnOpcode)
          .addReg(Src0Cast)
          .addReg(Src1Cast)
          .addReg(Src2Cast);

      MIB.buildBitcast(DstCast, Dst);
      Dsts.push_back(DstCast);
    }

    joinVectorByGrain(MIB, OrigDst, Dsts, Grain);
    MI.eraseFromParent();
  } break;
  case 32:
    scalarizeIntrinsic(MI);
    return true;
  case 64: {
    auto MIs = scalarizeIntrinsic(MI);

    for (auto *MI : MIs) {
      MachineIRBuilder MIB(*MI);
      auto Dst = MI->getOperand(0).getReg();
      auto Src0 = MI->getOperand(3).getReg();
      auto Src1 = MI->getOperand(4).getReg();
      auto Src2 = MI->getOperand(5).getReg();

      auto DstV2I32 = MRI.createGenericVirtualRegister(V2I32);
      auto Src0V2I32 = MRI.createGenericVirtualRegister(V2I32);
      auto Src1V2I32 = MRI.createGenericVirtualRegister(V2I32);
      auto Src2V2I32 = MRI.createGenericVirtualRegister(V2I32);

      MIB.buildUndef(DstV2I32);

      MIB.buildBitcast(Src0V2I32, Src0);
      MIB.buildBitcast(Src1V2I32, Src1);
      MIB.buildBitcast(Src2V2I32, Src2);

      for (int I = 0; I < 2; I++) {
        auto DstI32 = MRI.createGenericVirtualRegister(I32);
        auto Src0I32 = MRI.createGenericVirtualRegister(I32);
        auto Src1I32 = MRI.createGenericVirtualRegister(I32);
        auto Src2I32 = MRI.createGenericVirtualRegister(I32);
        auto Idx = MRI.createGenericVirtualRegister(I32);

        MIB.buildExtractVectorElementConstant(Src0I32, Src0V2I32, I);
        MIB.buildExtractVectorElementConstant(Src1I32, Src1V2I32, I);
        MIB.buildExtractVectorElementConstant(Src2I32, Src2V2I32, I);

        MIB.buildIntrinsic(IntrinsicID, DstI32)
            .add(MI->getOperand(2))
            .addReg(Src0I32)
            .addReg(Src1I32)
            .addReg(Src2I32);

        MIB.buildConstant(Idx, I);

        auto DstNext = MRI.createGenericVirtualRegister(V2I32);
        MIB.buildInsertVectorElement(DstNext, DstV2I32, DstI32, Idx);
        DstV2I32 = DstNext;
      }

      MIB.buildBitcast(Dst, DstV2I32);
      MI->eraseFromParent();
    }
  } break;
  }

  return true;
}

static bool legalizeIntrinsicRE(LegalizerHelper &Helper, MachineInstr &MI) {
  MachineIRBuilder &B = Helper.MIRBuilder;

  auto RndMode = MI.getOperand(MI.getNumOperands() - 1).getImm();
  if (static_cast<RoundingMode>(RndMode) != RoundingMode::NearestTiesToEven)
    return false; // only .re is supported

  Intrinsic::ID IntrinsicID = cast<GIntrinsic>(MI).getIntrinsicID();
  switch (IntrinsicID) {
  default:
    return false;
  case Intrinsic::pisa_log_rnd:
    B.buildFLog(MI.getOperand(0), MI.getOperand(2), MI.getFlags());
    break;
  case Intrinsic::pisa_log2_rnd:
    B.buildFLog2(MI.getOperand(0), MI.getOperand(2), MI.getFlags());
    break;
  case Intrinsic::pisa_log10_rnd:
    B.buildInstr(TargetOpcode::G_FLOG10, {MI.getOperand(0)}, {MI.getOperand(2)},
                 MI.getFlags());
    break;
  case Intrinsic::pisa_sin_rnd:
    B.buildInstr(TargetOpcode::G_FSIN, {MI.getOperand(0)}, {MI.getOperand(2)},
                 MI.getFlags());
    break;
  case Intrinsic::pisa_cos_rnd:
    B.buildInstr(TargetOpcode::G_FCOS, {MI.getOperand(0)}, {MI.getOperand(2)},
                 MI.getFlags());
    break;
  case Intrinsic::pisa_tanh_rnd:
    B.buildInstr(TargetOpcode::G_FTANH, {MI.getOperand(0)}, {MI.getOperand(2)},
                 MI.getFlags());
    break;
  case Intrinsic::pisa_exp_rnd:
    B.buildInstr(TargetOpcode::G_FEXP, {MI.getOperand(0)}, {MI.getOperand(2)},
                 MI.getFlags());
    break;
  case Intrinsic::pisa_exp2_rnd:
    B.buildFExp2(MI.getOperand(0), MI.getOperand(2), MI.getFlags());
    break;
  case Intrinsic::pisa_pow_rnd:
    B.buildFPow(MI.getOperand(0), MI.getOperand(2), MI.getOperand(3),
                MI.getFlags());
    break;
  }
  MI.eraseFromParent();
  return true;
}

static bool legalizeIntrinsicI2F(LegalizerHelper &Helper, MachineInstr &MI) {
  MachineIRBuilder &B = Helper.MIRBuilder;
  auto &MRI = *B.getMRI();

  Intrinsic::ID IntrinsicID = cast<GIntrinsic>(MI).getIntrinsicID();
  SmallVector<MachineInstr *> MIs = scalarizeIntrinsic(MI);

  // @llvm.experimental.constrained.sitofp.f32.i1
  for (auto It = MIs.begin(), Ite = MIs.end(); It != Ite; ++It) {
    auto *MI = *It;
    MachineIRBuilder MIB(*MI);

    auto SrcReg = MI->getOperand(2).getReg();
    auto SrcTy = MRI.getType(SrcReg);

    if (SrcTy.getSizeInBits() >= 8)
      continue;

    auto ExtReg = MRI.createGenericVirtualRegister(I8);
    if (IntrinsicID == Intrinsic::pisa_uitofp)
      MIB.buildZExt(ExtReg, SrcReg);
    else
      MIB.buildSExt(ExtReg, SrcReg);
    auto NewMI = MIB.buildIntrinsic(IntrinsicID, MI->getOperand(0).getReg())
                     .addReg(ExtReg)
                     .add(MI->getOperand(3))
                     .add(MI->getOperand(4));
    NewMI.setMIFlags(MI->getFlags());
    MI->eraseFromParent();
  }
  return true;
}

// Legalize dp4a_uu with saturation enabled:
//   dp4a_uu(acc, src1, src2, sat=true)
// => tmp = dp4a_uu(0, src1, src2, sat=false)
//    dst = G_UADDSAT(tmp, acc)
static bool legalizeIntrinsicDp4a(LegalizerHelper &Helper, MachineInstr &MI) {
  unsigned Sat = MI.getOperand(5).getImm();
  if (Sat == 0)
    return true;

  MachineIRBuilder &B = Helper.MIRBuilder;
  auto &MRI = *B.getMRI();
  B.setInstrAndDebugLoc(MI);

  Register Dst = MI.getOperand(0).getReg();
  Register Acc = MI.getOperand(2).getReg();
  Register Src1 = MI.getOperand(3).getReg();
  Register Src2 = MI.getOperand(4).getReg();

  Register Zero = B.buildConstant(I32, 0).getReg(0);
  Register Tmp = MRI.createGenericVirtualRegister(I32);
  B.buildIntrinsic(Intrinsic::pisa_dp4a_uu, ArrayRef<Register>{Tmp})
      .addUse(Zero)
      .addUse(Src1)
      .addUse(Src2)
      .addImm(0);
  B.buildInstr(TargetOpcode::G_UADDSAT, {DstOp(Dst)}, {SrcOp(Tmp), SrcOp(Acc)});

  MI.eraseFromParent();
  return true;
}

bool PISALegalizerInfo::legalizeIntrinsic(LegalizerHelper &Helper,
                                          MachineInstr &MI) const {
  Intrinsic::ID IntrinsicID = cast<GIntrinsic>(MI).getIntrinsicID();
  switch (IntrinsicID) {
  case Intrinsic::pisa_dp4a_uu:
    return legalizeIntrinsicDp4a(Helper, MI);
  case Intrinsic::pisa_fsqrt_rnd:
    return legalizeIntrinsicFSqrt(Helper, MI);
  case Intrinsic::pisa_fdiv_rnd:
    return legalizeIntrinsicFDiv(Helper, MI);
  case Intrinsic::pisa_bfn:
    return legalizeIntrinsicBfn(Helper, MI);
  case Intrinsic::pisa_log_rnd:
  case Intrinsic::pisa_log2_rnd:
  case Intrinsic::pisa_log10_rnd:
  case Intrinsic::pisa_sin_rnd:
  case Intrinsic::pisa_cos_rnd:
  case Intrinsic::pisa_tanh_rnd:
  case Intrinsic::pisa_exp_rnd:
  case Intrinsic::pisa_exp2_rnd:
  case Intrinsic::pisa_pow_rnd:
    return legalizeIntrinsicRE(Helper, MI);
  case Intrinsic::pisa_ired:
  case Intrinsic::pisa_fred:
  case Intrinsic::pisa_frcp:
  case Intrinsic::pisa_frsqrt:
  case Intrinsic::pisa_fabs:
  case Intrinsic::pisa_smad:
  case Intrinsic::pisa_fptosi_rnd:
  case Intrinsic::pisa_fptoui_rnd:
  case Intrinsic::pisa_fadd:
  case Intrinsic::pisa_fsub:
  case Intrinsic::pisa_fmul:
  case Intrinsic::pisa_fma:
  case Intrinsic::pisa_ftrunc:
  case Intrinsic::pisa_frnd_rnd:
    scalarizeIntrinsic(MI);
    return true;
  case Intrinsic::pisa_sitofp:
  case Intrinsic::pisa_uitofp:
    return legalizeIntrinsicI2F(Helper, MI);
  default:
    return true;
  }
}

static bool legalizeGConcatVectors(MachineInstr &MI, MachineIRBuilder &B) {
  auto *MRI = B.getMRI();

  auto Dst = MI.getOperand(0).getReg();
  auto DstTy = MRI->getType(Dst);
  assert(DstTy.getScalarSizeInBits() == 32);
  unsigned Idx = 0;

  auto TDst = MRI->createGenericVirtualRegister(DstTy);
  B.buildInstr(TargetOpcode::IMPLICIT_DEF).addDef(TDst);
  for (unsigned I = 1; I < MI.getNumOperands(); I++) {
    auto Src = MI.getOperand(I).getReg();
    auto SrcTy = B.getMRI()->getType(Src);
    auto NewDst = MRI->createGenericVirtualRegister(DstTy);
    B.buildInsertSubvector(NewDst, TDst, Src, Idx);
    Idx += SrcTy.getNumElements();
    TDst = NewDst;
  }
  B.buildCopy(Dst, TDst);
  MI.eraseFromParent();
  return true;
}

static bool legalizeGUnmergeValues(LegalizerHelper &Helper, MachineInstr &MI,
                                   MachineIRBuilder &B) {
  auto *MRI = B.getMRI();

  auto Src = MI.getOperand(MI.getNumOperands() - 1).getReg();
  auto DstTy = MRI->getType(MI.getOperand(0).getReg());
  assert(MRI->getType(Src).getScalarSizeInBits() == 32);

  unsigned Idx = 0;
  for (unsigned I = 0; I < MI.getNumOperands() - 1; I++) {
    auto Dst = MI.getOperand(I).getReg();
    if (DstTy.isVector()) {
      // <2 x s32>, <2 x s32> = G_UNMERGE_VALUES <4 x s32>
      B.buildExtractSubvector(Dst, Src, Idx);
      Idx += DstTy.getNumElements();
    } else {
      // s32, s32, s32, s32 = G_UNMERGE_VALUES <4 x s32>
      B.buildExtractVectorElementConstant(Dst, Src, Idx);
      Idx += 1;
    }
  }
  MI.eraseFromParent();
  return true;
}

// NOLINTNEXTLINE(readability-identifier-naming)
static bool legalizeGInsertSubvector(MachineInstr &MI, MachineIRBuilder &B) {
  auto *MRI = B.getMRI();

  auto DstReg = MI.getOperand(0).getReg();
  auto VecReg = MI.getOperand(1).getReg();
  auto SubVecReg = MI.getOperand(2).getReg();
  auto Idx = MI.getOperand(3).getImm();

  LLT DstTy = MRI->getType(DstReg);

  if (DstTy.getScalarSizeInBits() == 64) {
    LLT VecTy = MRI->getType(VecReg);
    LLT SubVecTy = MRI->getType(SubVecReg);
    LLT CastedVecTy =
        LLT::fixed_vector(2 * VecTy.getNumElements(), LLT::integer(32));
    LLT CastedSubVecTy =
        LLT::fixed_vector(2 * SubVecTy.getNumElements(), LLT::integer(32));
    LLT CastedDstTy =
        LLT::fixed_vector(2 * DstTy.getNumElements(), LLT::integer(32));

    // Update vec
    auto CastedVecReg = MRI->createGenericVirtualRegister(CastedVecTy);
    B.buildBitcast(CastedVecReg, VecReg);
    MI.getOperand(1).setReg(CastedVecReg);

    // Update subvec
    auto CastedSubVecReg = MRI->createGenericVirtualRegister(CastedSubVecTy);
    B.buildBitcast(CastedSubVecReg, SubVecReg);
    MI.getOperand(2).setReg(CastedSubVecReg);

    // Update dst
    auto CastedDstReg = MRI->createGenericVirtualRegister(CastedDstTy);
    MI.getOperand(0).setReg(CastedDstReg);

    // Update index
    MI.getOperand(3).setImm(2 * Idx);

    B.setInsertPt(B.getMBB(), ++B.getInsertPt());
    B.buildBitcast(DstReg, CastedDstReg);
    return legalizeGInsertSubvector(MI, B);
  }

  assert(DstTy.getScalarSizeInBits() == 32 && "Unexpected scalar size");

  unsigned NumDstElems = DstTy.getNumElements();
  if (NumDstElems <= 64)
    return true; // Already legal

  // Split destination vector into legal 32-element chunks
  unsigned NumChunks = NumDstElems / 32;
  LLT ChunkTy = LLT::vector(ElementCount::getFixed(32), DstTy.getScalarType());

  SmallVector<Register, 4> Chunks;
  for (unsigned I = 0; I < NumChunks; ++I)
    Chunks.push_back(MRI->createGenericVirtualRegister(ChunkTy));

  B.setInsertPt(*MI.getParent(), MI);
  B.buildUnmerge(Chunks, VecReg);

  // Determine which chunk the subvector belongs in
  unsigned ChunkIdx = Idx / 32;
  unsigned OffsetInChunk = Idx % 32;

  // Per LLVM spec:
  // "Idx must be a constant multiple of subvec’s known minimum vector length"
  assert(ChunkIdx < Chunks.size() && "Subvector index out of bounds");
  assert(OffsetInChunk + MRI->getType(SubVecReg).getNumElements() <= 32 &&
         "Subvector spans multiple chunks");

  // Insert subvector into the appropriate chunk
  Register ModifiedChunk = MRI->createGenericVirtualRegister(ChunkTy);
  B.buildInsertSubvector(ModifiedChunk, Chunks[ChunkIdx], SubVecReg,
                         OffsetInChunk);
  Chunks[ChunkIdx] = ModifiedChunk;

  // Rebuild the final vector
  Register FinalVec = MRI->createGenericVirtualRegister(DstTy);
  B.buildConcatVectors(FinalVec, Chunks);
  B.buildCopy(DstReg, FinalVec);

  MI.eraseFromParent();
  return true;
}

// NOLINTNEXTLINE(readability-identifier-naming)
static bool legalizeGExtractSubvector(MachineInstr &MI, MachineIRBuilder &B) {
  auto *MRI = B.getMRI();

  auto DstReg = MI.getOperand(0).getReg();
  auto SrcReg = MI.getOperand(1).getReg();
  auto Idx = MI.getOperand(2).getImm();

  LLT SrcTy = MRI->getType(SrcReg);

  if (SrcTy.getScalarSizeInBits() == 64) {
    LLT DstTy = MRI->getType(DstReg);
    LLT CastedSrcTy =
        LLT::fixed_vector(2 * SrcTy.getNumElements(), LLT::integer(32));
    LLT CastedDstTy =
        LLT::fixed_vector(2 * DstTy.getNumElements(), LLT::integer(32));

    // Update src
    auto CastedSrcReg = MRI->createGenericVirtualRegister(CastedSrcTy);
    B.buildBitcast(CastedSrcReg, SrcReg);
    MI.getOperand(1).setReg(CastedSrcReg);

    // Update dst
    auto CastedDstReg = MRI->createGenericVirtualRegister(CastedDstTy);
    MI.getOperand(0).setReg(CastedDstReg);

    // Update index
    MI.getOperand(2).setImm(2 * Idx);

    B.setInsertPt(B.getMBB(), ++B.getInsertPt());
    B.buildBitcast(DstReg, CastedDstReg);
    return legalizeGExtractSubvector(MI, B);
  }

  assert(SrcTy.getScalarSizeInBits() == 32 && "Unexpected scalar size");

  unsigned NumSrcElems = SrcTy.getNumElements();
  if (NumSrcElems <= 64)
    return true; // Already legal

  // Define 32-element legal vector type
  unsigned NumChunks = NumSrcElems / 32;
  LLT ChunkTy = LLT::vector(ElementCount::getFixed(32), SrcTy.getScalarType());

  // Create registers for each chunk
  SmallVector<Register, 4> Chunks;
  for (unsigned I = 0; I < NumChunks; ++I)
    Chunks.push_back(MRI->createGenericVirtualRegister(ChunkTy));

  B.setInsertPt(*MI.getParent(), MI);
  B.buildUnmerge(Chunks, SrcReg);

  // Determine chunk index and offset
  unsigned ChunkIdx = Idx / 32;
  unsigned OffsetInChunk = Idx % 32;

  // Per LLVM spec:
  // "Idx must be a constant multiple of the known-minimum vector length of the
  // result type"
  assert(ChunkIdx < Chunks.size() && "Subvector index out of bounds");
  assert(OffsetInChunk + MRI->getType(DstReg).getNumElements() <= 32 &&
         "Subvector spans multiple legal vector chunks");

  B.buildExtractSubvector(DstReg, Chunks[ChunkIdx], OffsetInChunk);
  MI.eraseFromParent();
  return true;
}

static StringRef getSyncScopeStr(LLVMContext &Ctx, SyncScope::ID ScopeID) {
  // Map dynamically assigned PISA SyncScope ID to its scope name.
  static DenseMap<SyncScope::ID, StringRef> ScopeID2Name;
  auto InitializeScopeID2Name = [&]() {
    static const StringMap<StringRef> ScopeName2EncodeName = {
        {"workgroup", "workgroup"},
        {"gpu", "gpu"},
        {"system", "system"},
        // using workgroup scope (see PISAScopeSelector pass).
        {"subgroup", "workgroup"},
        {"workitem", "workgroup"},
    };
    for (const auto &[Name, EncodeName] : ScopeName2EncodeName) {
      auto ID = Ctx.getOrInsertSyncScopeID(Name);
      ScopeID2Name.emplace_or_assign(ID, EncodeName);
    }
  };
  static llvm::once_flag InitializeScopeID2NameFlag;
  std::call_once(InitializeScopeID2NameFlag, InitializeScopeID2Name);

  // Use the original SyncScope ID to look up its scope name in the map.
  auto It = ScopeID2Name.find(ScopeID);
  return It != ScopeID2Name.end() ? It->second : StringRef("gpu");
}

// NOLINTNEXTLINE(readability-identifier-naming)
static bool legalizeGAtomicrmw(MachineInstr &MI, MachineIRBuilder &B) {
  const MachineMemOperand *MemOp = *MI.memoperands_begin();
  AtomicOrdering AO = MemOp->getSuccessOrdering();
  AtomicOrdering AOF = MemOp->getFailureOrdering();
  if (isAtLeastOrStrongerThan(AtomicOrdering::AcquireRelease, AO) &&
      isAtLeastOrStrongerThan(AtomicOrdering::Release, AOF))
    return true;

  auto &Ctx = B.getMF().getFunction().getContext();
  llvm::SmallString<16> FenceScopeStr =
      getSyncScopeStr(Ctx, MemOp->getSyncScopeID());
  unsigned AddressSpace = MemOp->getAddrSpace();
  if (AddressSpace == unsigned(PISAAS::AddressSpace::SHARED))
    FenceScopeStr += "-shared";
  else if (AddressSpace == unsigned(PISAAS::AddressSpace::GENERIC))
    FenceScopeStr += "-generic";
  else
    FenceScopeStr += "-global";

  B.buildFence(
      static_cast<unsigned>(llvm::AtomicOrdering::SequentiallyConsistent),
      Ctx.getOrInsertSyncScopeID(FenceScopeStr));

  if (!isAtLeastOrStrongerThan(AtomicOrdering::AcquireRelease, AO))
    AO = AtomicOrdering::Monotonic;
  if (!isAtLeastOrStrongerThan(AtomicOrdering::Release, AOF))
    AOF = AtomicOrdering::Monotonic;

  MachineMemOperand *NewMemOp = B.getMF().getMachineMemOperand(
      MemOp->getPointerInfo(), MemOp->getFlags(), MemOp->getSize(),
      MemOp->getAlign(), MemOp->getAAInfo(), MemOp->getRanges(),
      MemOp->getSyncScopeID(), AO, AOF);

  if (MI.getOpcode() == TargetOpcode::G_ATOMIC_CMPXCHG)
    B.buildAtomicCmpXchg(MI.getOperand(0), MI.getOperand(1), MI.getOperand(2),
                         MI.getOperand(3), *NewMemOp)
        .setMIFlags(MI.getFlags());
  else
    B.buildAtomicRMW(MI.getOpcode(), MI.getOperand(0), MI.getOperand(1),
                     MI.getOperand(2), *NewMemOp)
        .setMIFlags(MI.getFlags());

  B.buildFence(
      static_cast<unsigned>(llvm::AtomicOrdering::SequentiallyConsistent),
      Ctx.getOrInsertSyncScopeID(FenceScopeStr));

  MI.eraseFromParent();
  return true;
}

// NOLINTNEXTLINE(readability-identifier-naming)
static bool legalizeGAtomicrmwXchg(MachineInstr &MI, MachineIRBuilder &B) {
  auto &MRI = *B.getMRI();
  auto &Dst = MI.getOperand(0);
  LLT CurTy = MRI.getType(Dst.getReg());
  if (CurTy.getScalarType().isPointer()) {
    auto &Src = MI.getOperand(2);
    LLT NewTy = LLT::integer(CurTy.getScalarSizeInBits());
    Register NewSrc = MRI.createGenericVirtualRegister(NewTy);
    Register NewDst = MRI.createGenericVirtualRegister(NewTy);
    B.buildPtrToInt(NewSrc, Src);
    B.buildAtomicRMWXchg(NewDst, MI.getOperand(1).getReg(), NewSrc,
                         **MI.memoperands_begin());
    B.buildIntToPtr(Dst, NewDst);
    MI.eraseFromParent();
  } else {
    legalizeGAtomicrmw(MI, B);
  }
  return true;
}

// NOLINTNEXTLINE(readability-identifier-naming)
static bool legalizeGShuffleVector(LegalizerHelper &Helper, MachineInstr &MI,
                                   MachineIRBuilder &B) {
  auto &MRI = *B.getMRI();
  auto &Dst = MI.getOperand(0);
  auto &Src0 = MI.getOperand(1);
  auto &Src1 = MI.getOperand(2);
  ArrayRef<int> Mask = MI.getOperand(3).getShuffleMask();
  assert(MRI.getType(Dst.getReg()).getScalarSizeInBits() == 32);
  assert(isPowerOf2_32(MRI.getType(Dst.getReg()).getNumElements()));

  auto UseExtract = true;
  // indices must be consecutive
  int PrevIdx = -1;
  for (int Idx : Mask) {
    if ((PrevIdx != -1) && (Idx != (PrevIdx + 1)))
      UseExtract = false;
    PrevIdx = Idx;
  }

  // starting index must be aligned to destination size
  if (Mask[0] % Mask.size())
    UseExtract = false;

  // indices can not straddle the arguments
  auto SrcSize = MRI.getType(Src0.getReg()).getNumElements();
  if ((Mask[0] < SrcSize) && ((Mask[0] + Mask.size()) > SrcSize))
    UseExtract = false;

  if (SrcSize > 8 && SrcSize != 16 && SrcSize != 32 && (SrcSize % 64))
    UseExtract = false;

  if (UseExtract) {
    auto Src = (Mask[0] < SrcSize) ? Src0 : Src1;
    auto Idx = (Mask[0] < SrcSize) ? Mask[0] : Mask[0] - SrcSize;
    B.buildExtractSubvector(Dst, Src, Idx);
    MI.eraseFromParent();
    return true;
  }

  // lower if unable to use extract/insert
  auto Res = Helper.lowerShuffleVector(MI);
  return Res != LegalizerHelper::UnableToLegalize;
}

// NOLINTNEXTLINE(readability-identifier-naming)
static bool legalizeGIsFpclass(LegalizerHelper &Helper, MachineInstr &MI,
                               MachineIRBuilder &MIRBuilder) {
  auto [DstReg, DstTy, SrcReg, SrcTy] = MI.getFirst2RegLLTs();
  FPClassTest OriginalMask =
      static_cast<FPClassTest>(MI.getOperand(2).getImm());
  auto Mask = OriginalMask;
  auto IsInvertedCheck = false;

  if (Mask == fcNone) {
    MIRBuilder.buildConstant(DstReg, 0);
    MI.eraseFromParent();
    return true;
  }
  if (Mask == fcAllFlags) {
    MIRBuilder.buildConstant(DstReg, 1);
    MI.eraseFromParent();
    return true;
  }

  // support bfloat types
  auto &Semantics = getFltSemanticForLLT(SrcTy.getScalarType());

  unsigned BitSize = SrcTy.getScalarSizeInBits();
  LLT IntTy = LLT::integer(BitSize);
  if (SrcTy.isVector())
    IntTy = LLT::vector(SrcTy.getElementCount(), IntTy);
  auto AsInt = MIRBuilder.buildBitcast(IntTy, SrcReg);

  // Various masks.
  APInt SignBit = APInt::getSignMask(BitSize);
  APInt ValueMask = APInt::getSignedMaxValue(BitSize);     // All bits but sign.
  APInt Inf = APFloat::getInf(Semantics).bitcastToAPInt(); // Exp and int bit.
  APInt ExpMask = Inf;
  APInt AllOneMantissa = APFloat::getLargest(Semantics).bitcastToAPInt() & ~Inf;
  APInt QNaNBitMask =
      APInt::getOneBitSet(BitSize, AllOneMantissa.getActiveBits() - 1);
  APInt InvertionMask = APInt::getAllOnes(DstTy.getScalarSizeInBits());

  auto SignBitC = MIRBuilder.buildConstant(IntTy, SignBit);
  auto ValueMaskC = MIRBuilder.buildConstant(IntTy, ValueMask);
  auto InfC = MIRBuilder.buildConstant(IntTy, Inf);
  auto ExpMaskC = MIRBuilder.buildConstant(IntTy, ExpMask);
  auto ZeroC = MIRBuilder.buildConstant(IntTy, 0);

  auto Abs = MIRBuilder.buildAnd(IntTy, AsInt, ValueMaskC);
  auto Sign =
      MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_NE, DstTy, AsInt, Abs);

  auto Res = MIRBuilder.buildConstant(DstTy, 0);
  // Clang doesn't support capture of structured bindings:
  LLT DstTyCopy = DstTy;
  const auto AppendToRes = [&](MachineInstrBuilder ToAppend) {
    Res = MIRBuilder.buildOr(DstTyCopy, Res, ToAppend);
  };

  // Tests that involve more than one class should be processed first.
  if ((Mask & fcFinite) == fcFinite) {
    // finite(V) ==> abs(V) u< exp_mask
    AppendToRes(MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_ULT, DstTy, Abs,
                                     ExpMaskC));
    Mask &= ~fcFinite;
  } else if ((Mask & fcFinite) == fcPosFinite) {
    // finite(V) && V > 0 ==> V u< exp_mask
    AppendToRes(MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_ULT, DstTy, AsInt,
                                     ExpMaskC));
    Mask &= ~fcPosFinite;
  } else if ((Mask & fcFinite) == fcNegFinite) {
    // finite(V) && V < 0 ==> abs(V) u< exp_mask && signbit == 1
    auto Cmp = MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_ULT, DstTy, Abs,
                                    ExpMaskC);
    auto And = MIRBuilder.buildAnd(DstTy, Cmp, Sign);
    AppendToRes(And);
    Mask &= ~fcNegFinite;
  }

  if (FPClassTest PartialCheck = Mask & (fcZero | fcSubnormal)) {
    // fcZero | fcSubnormal => test all exponent bits are 0
    // TODO: Handle sign bit specific cases
    // TODO: Handle inverted case
    if (PartialCheck == (fcZero | fcSubnormal)) {
      auto ExpBits = MIRBuilder.buildAnd(IntTy, AsInt, ExpMaskC);
      AppendToRes(MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_EQ, DstTy,
                                       ExpBits, ZeroC));
      Mask &= ~PartialCheck;
    }
  }

  if (Mask == OriginalMask) {
    // combination of classes above did not yield any
    // optimizations, see if inverse will be less ops
    auto InvertedMask = (unsigned)~Mask;
    if (llvm::popcount((unsigned)Mask) > llvm::popcount(InvertedMask)) {
      Mask = ~Mask;
      IsInvertedCheck = true;
    }
  }

  // Check for individual classes.
  if (FPClassTest PartialCheck = Mask & fcZero) {
    if (PartialCheck == fcPosZero)
      AppendToRes(MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_EQ, DstTy,
                                       AsInt, ZeroC));
    else if (PartialCheck == fcZero)
      AppendToRes(
          MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_EQ, DstTy, Abs, ZeroC));
    else // fcNegZero
      AppendToRes(MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_EQ, DstTy,
                                       AsInt, SignBitC));
  }

  if (FPClassTest PartialCheck = Mask & fcSubnormal) {
    // issubnormal(V) ==> unsigned(abs(V) - 1) u< (all mantissa bits set)
    // issubnormal(V) && V>0 ==> unsigned(V - 1) u< (all mantissa bits set)
    auto V = (PartialCheck == fcPosSubnormal) ? AsInt : Abs;
    auto OneC = MIRBuilder.buildConstant(IntTy, 1);
    auto VMinusOne = MIRBuilder.buildSub(IntTy, V, OneC);
    auto SubnormalRes =
        MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_ULT, DstTy, VMinusOne,
                             MIRBuilder.buildConstant(IntTy, AllOneMantissa));
    if (PartialCheck == fcNegSubnormal)
      SubnormalRes = MIRBuilder.buildAnd(DstTy, SubnormalRes, Sign);
    AppendToRes(SubnormalRes);
  }

  if (FPClassTest PartialCheck = Mask & fcInf) {
    if (PartialCheck == fcPosInf)
      AppendToRes(MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_EQ, DstTy,
                                       AsInt, InfC));
    else if (PartialCheck == fcInf)
      AppendToRes(
          MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_EQ, DstTy, Abs, InfC));
    else { // fcNegInf
      APInt NegInf = APFloat::getInf(Semantics, true).bitcastToAPInt();
      auto NegInfC = MIRBuilder.buildConstant(IntTy, NegInf);
      AppendToRes(MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_EQ, DstTy,
                                       AsInt, NegInfC));
    }
  }

  if (FPClassTest PartialCheck = Mask & fcNan) {
    auto InfWithQnanBitC =
        MIRBuilder.buildConstant(IntTy, std::move(Inf) | QNaNBitMask);
    if (PartialCheck == fcNan) {
      // isnan(V) ==> abs(V) u> int(inf)
      AppendToRes(
          MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_UGT, DstTy, Abs, InfC));
    } else if (PartialCheck == fcQNan) {
      // isquiet(V) ==> abs(V) u>= (unsigned(Inf) | quiet_bit)
      AppendToRes(MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_UGE, DstTy, Abs,
                                       InfWithQnanBitC));
    } else { // fcSNan
      // issignaling(V) ==> abs(V) u> unsigned(Inf) &&
      //                    abs(V) u< (unsigned(Inf) | quiet_bit)
      auto IsNan =
          MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_UGT, DstTy, Abs, InfC);
      auto IsNotQnan = MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_ULT, DstTy,
                                            Abs, InfWithQnanBitC);
      AppendToRes(MIRBuilder.buildAnd(DstTy, IsNan, IsNotQnan));
    }
  }

  if (FPClassTest PartialCheck = Mask & fcNormal) {
    // isnormal(V) ==> (0 u< exp u< max_exp) ==> (unsigned(exp-1) u<
    // (max_exp-1))
    APInt ExpLSB = ExpMask & ~(ExpMask.shl(1));
    auto ExpMinusOne = MIRBuilder.buildSub(
        IntTy, Abs, MIRBuilder.buildConstant(IntTy, ExpLSB));
    APInt MaxExpMinusOne = std::move(ExpMask) - ExpLSB;
    auto NormalRes =
        MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_ULT, DstTy, ExpMinusOne,
                             MIRBuilder.buildConstant(IntTy, MaxExpMinusOne));
    if (PartialCheck == fcNegNormal)
      NormalRes = MIRBuilder.buildAnd(DstTy, NormalRes, Sign);
    else if (PartialCheck == fcPosNormal) {
      auto PosSign = MIRBuilder.buildXor(
          DstTy, Sign, MIRBuilder.buildConstant(DstTy, InvertionMask));
      NormalRes = MIRBuilder.buildAnd(DstTy, NormalRes, PosSign);
    }
    AppendToRes(NormalRes);
  }

  if (IsInvertedCheck)
    MIRBuilder.buildNot(DstReg, Res);
  else
    MIRBuilder.buildCopy(DstReg, Res);
  MI.eraseFromParent();
  return true;
}

static bool legalizeGMulh(LegalizerHelper &Helper, MachineInstr &MI,
                          MachineIRBuilder &B) {
  auto &MRI = *B.getMRI();
  auto [Dst, Src0, Src1] = MI.getFirst3Regs();
  auto IsSigned = MI.getOpcode() == TargetOpcode::G_SMULH;
  auto DstTy = MRI.getType(Dst);
  assert(DstTy.getSizeInBits() == 64);

  auto SourceA = MRI.createGenericVirtualRegister(DstTy);
  auto SourceB = MRI.createGenericVirtualRegister(DstTy);

  auto Const32 = MRI.createGenericVirtualRegister(DstTy);
  auto Const63 = MRI.createGenericVirtualRegister(DstTy);
  auto Const0 = MRI.createGenericVirtualRegister(DstTy);
  auto Mask32 = MRI.createGenericVirtualRegister(DstTy);
  B.buildConstant(Const32, 32);
  B.buildConstant(Const63, 63);
  B.buildConstant(Const0, 0);
  B.buildConstant(Mask32, 0xFFFFFFFF);

  auto ASign = MRI.createGenericVirtualRegister(DstTy);
  auto BSign = MRI.createGenericVirtualRegister(DstTy);
  auto ResultSign = MRI.createGenericVirtualRegister(DstTy);
  B.buildAShr(ASign, Src0, Const63);
  B.buildAShr(BSign, Src1, Const63);
  B.buildXor(ResultSign, ASign, BSign);

  if (IsSigned) {
    auto ASignXor = MRI.createGenericVirtualRegister(DstTy);
    auto BSignXor = MRI.createGenericVirtualRegister(DstTy);
    B.buildXor(ASignXor, Src0, ASign);
    B.buildXor(BSignXor, Src1, BSign);
    B.buildSub(SourceA, ASignXor, ASign);
    B.buildSub(SourceB, BSignXor, BSign);
  } else {
    B.buildCopy(SourceA, Src0);
    B.buildCopy(SourceB, Src1);
  }

  auto LoSrc0 = MRI.createGenericVirtualRegister(DstTy);
  auto HiSrc0 = MRI.createGenericVirtualRegister(DstTy);
  auto LoSrc1 = MRI.createGenericVirtualRegister(DstTy);
  auto HiSrc1 = MRI.createGenericVirtualRegister(DstTy);
  B.buildLShr(HiSrc0, SourceA, Const32);
  B.buildLShr(HiSrc1, SourceB, Const32);
  B.buildAnd(LoSrc0, SourceA, Mask32);
  B.buildAnd(LoSrc1, SourceB, Mask32);

  auto ALobLo = MRI.createGenericVirtualRegister(DstTy);
  auto ALobHi = MRI.createGenericVirtualRegister(DstTy);
  auto AHibLo = MRI.createGenericVirtualRegister(DstTy);
  auto AHibHi = MRI.createGenericVirtualRegister(DstTy);
  B.buildMul(AHibHi, HiSrc0, HiSrc1);
  B.buildMul(AHibLo, HiSrc0, LoSrc1);
  B.buildMul(ALobHi, LoSrc0, HiSrc1);
  B.buildMul(ALobLo, LoSrc0, LoSrc1);

  auto ALobLoHi = MRI.createGenericVirtualRegister(DstTy);
  auto ALobHiLo = MRI.createGenericVirtualRegister(DstTy);
  auto AHibLoSum0 = MRI.createGenericVirtualRegister(DstTy);
  auto AHibLoSum1 = MRI.createGenericVirtualRegister(DstTy);
  B.buildLShr(ALobLoHi, ALobLo, Const32);
  B.buildAnd(ALobHiLo, ALobHi, Mask32);
  B.buildAdd(AHibLoSum0, ALobLoHi, ALobHiLo);
  B.buildAdd(AHibLoSum1, AHibLo, AHibLoSum0);

  auto ALobLoMasked = MRI.createGenericVirtualRegister(DstTy);
  auto AHibLoShiftedL = MRI.createGenericVirtualRegister(DstTy);
  auto ALobHiShiftedR = MRI.createGenericVirtualRegister(DstTy);
  auto AHibLoShiftedR = MRI.createGenericVirtualRegister(DstTy);
  auto ShiftedSum = MRI.createGenericVirtualRegister(DstTy);
  auto DstLo = MRI.createGenericVirtualRegister(DstTy);
  auto DstHi = MRI.createGenericVirtualRegister(DstTy);

  B.buildAnd(ALobLoMasked, ALobLo, Mask32);
  B.buildShl(AHibLoShiftedL, AHibLoSum1, Const32);
  B.buildOr(DstLo, AHibLoShiftedL, ALobLoMasked);
  B.buildLShr(ALobHiShiftedR, ALobHi, Const32);
  B.buildLShr(AHibLoShiftedR, AHibLoSum1, Const32);
  B.buildAdd(ShiftedSum, ALobHiShiftedR, AHibLoShiftedR);
  B.buildAdd(DstHi, AHibHi, ShiftedSum);

  if (IsSigned) {
    // ulong mask = -resultSign;
    // hi = hi ^ mask;
    // lo = lo ^ mask;
    // lo += resultSign;  // Add 1 if resultSign is negative, otherwise add 0
    // hi += (lo < resultSign);  // Adjust hi if lo overflowed
    auto Mask = MRI.createGenericVirtualRegister(DstTy);
    B.buildNeg(Mask, ResultSign);
    auto HiXorMask = MRI.createGenericVirtualRegister(DstTy);
    auto LoXorMask = MRI.createGenericVirtualRegister(DstTy);
    B.buildXor(HiXorMask, DstHi, Mask);
    B.buildXor(LoXorMask, DstLo, Mask);

    auto LoAddResult = MRI.createGenericVirtualRegister(DstTy);
    auto LoAddShiftResult = MRI.createGenericVirtualRegister(DstTy);
    B.buildAdd(LoAddResult, LoXorMask, ResultSign);
    B.buildShl(LoAddShiftResult, LoAddResult, ResultSign);
    B.buildAdd(Dst, HiXorMask, LoAddShiftResult);
  } else {
    B.buildCopy(Dst, DstHi);
  }
  MI.eraseFromParent();
  return true;
}

// Legalizes an addrspacecast operation between pointers in non-generic address
// spaces. Ensures that null pointers are preserved during the cast by replacing
// the addrspacecast with a null pointer of the destination type.
static bool legalizeGAddrspaceCast(MachineInstr &MI, MachineIRBuilder &B) {
  auto &Dst = MI.getOperand(0);
  LLT DstTy = B.getMRI()->getType(Dst.getReg());
  B.buildConstant(
      Dst, PISATargetMachine::getNullPointerValue(DstTy.getAddressSpace()));
  MI.eraseFromParent();
  return true;
}

// NOLINTNEXTLINE(readability-identifier-naming)
static bool legalizeGBitcast(MachineInstr &MI, MachineIRBuilder &B) {
  auto [DstReg, DstTy, SrcReg, SrcTy] = MI.getFirst2RegLLTs();
  auto *MRI = B.getMRI();

  assert(SrcTy.getSizeInBits() == DstTy.getSizeInBits());
  if (SrcTy.isPointer() != DstTy.isPointer()) {
    // legalize bitcast between pointers and non-pointers
    if (SrcTy.isPointer()) {
      // <2 x i32> G_BITCAST (p1)
      auto IntSize = SrcTy.getSizeInBits();
      auto IntTy = LLT::integer(IntSize);
      auto IntReg = MRI->createGenericVirtualRegister(IntTy);
      B.buildPtrToInt(IntReg, SrcReg);
      if (IntTy == DstTy)
        B.buildCopy(DstReg, IntReg);
      else
        B.buildBitcast(DstReg, IntReg);
    } else {
      // (p1) G_BITCAST <2 x i32>
      auto IntSize = DstTy.getSizeInBits();
      auto IntTy = LLT::integer(IntSize);
      auto IntReg = MRI->createGenericVirtualRegister(IntTy);
      if (IntTy == SrcTy)
        B.buildCopy(IntReg, SrcReg);
      else
        B.buildBitcast(IntReg, SrcReg);
      B.buildIntToPtr(DstReg, IntReg);
    }
  } else {
    // Legalizes a bitcast operation between vector types by decomposing the
    // source vector into scalar elements and reassembling them into the
    // destination vector type.
    assert(SrcTy.isVector() && DstTy.isVector());
    unsigned SrcScalarSize = SrcTy.getScalarSizeInBits();
    unsigned DstScalarSize = DstTy.getScalarSizeInBits();

    if (SrcScalarSize == DstScalarSize) {
      // Same scalar size (e.g. <5 x f16> to <5 x i16>): extract each element
      // and reinterpret as the destination scalar type.
      SmallVector<Register, 4> DstElements;
      for (unsigned I = 0; I < SrcTy.getNumElements(); I++) {
        auto SrcElemReg =
            MRI->createGenericVirtualRegister(SrcTy.getScalarType());
        B.buildExtractVectorElementConstant(SrcElemReg, SrcReg, I);
        auto DstElemReg =
            MRI->createGenericVirtualRegister(DstTy.getScalarType());
        B.buildBitcast(DstElemReg, SrcElemReg);
        DstElements.push_back(DstElemReg);
      }
      B.buildBuildVector(DstReg, DstElements);
    } else {
      // Different scalar sizes with non-divisible element counts: decompose
      // via GCD-sized pieces.
      unsigned CommonPieceSize = std::gcd(DstScalarSize, SrcScalarSize);
      SmallVector<Register, 4> ScalarPieces;
      for (unsigned SrcElemIdx = 0; SrcElemIdx < SrcTy.getNumElements();
           SrcElemIdx++) {
        auto SrcElemReg =
            MRI->createGenericVirtualRegister(SrcTy.getScalarType());
        B.buildExtractVectorElementConstant(SrcElemReg, SrcReg, SrcElemIdx);
        auto Unmerge =
            B.buildUnmerge(LLT::integer(CommonPieceSize), SrcElemReg);
        for (unsigned PieceIdx = 0; PieceIdx != Unmerge->getNumOperands() - 1;
             PieceIdx++)
          ScalarPieces.push_back(Unmerge.getReg(PieceIdx));
      }
      unsigned NumPiecesPerDstElem = DstScalarSize / CommonPieceSize;
      SmallVector<Register, 4> DstElements;
      for (unsigned PieceStartIdx = 0; PieceStartIdx < ScalarPieces.size();
           PieceStartIdx += NumPiecesPerDstElem) {
        auto DstElemReg =
            MRI->createGenericVirtualRegister(DstTy.getScalarType());
        SmallVector<Register> DstElemPieces(
            ScalarPieces.begin() + PieceStartIdx,
            ScalarPieces.begin() + PieceStartIdx + NumPiecesPerDstElem);
        B.buildMergeValues(DstElemReg, DstElemPieces);
        DstElements.push_back(DstElemReg);
      }
      B.buildBuildVector(DstReg, DstElements);
    }
  }
  MI.eraseFromParent();
  return true;
}

bool PISALegalizerInfo::legalizeCustom(
    LegalizerHelper &Helper, MachineInstr &MI,
    LostDebugLocObserver &LocObserver) const {
  MachineIRBuilder &B = Helper.MIRBuilder;
  switch (MI.getOpcode()) {
  case TargetOpcode::G_FLOG:
    return legalizeGFlog(MI, B, numbers::ln2f);
  case TargetOpcode::G_FLOG10:
    return legalizeGFlog(MI, B, numbers::ln2f / numbers::ln10f);
  case TargetOpcode::G_FEXP:
    return legalizeGFexp(MI, B, numbers::log2e);
  case TargetOpcode::G_FEXP10:
    return legalizeGFexp(MI, B, numbers::ln10f / numbers::ln2f);
  case TargetOpcode::G_FCMP:
    return legalizeGFcmp(MI, B);
  case TargetOpcode::G_TRUNC:
    return legalizeGTrunc(MI, B);
  case TargetOpcode::G_ZEXT:
  case TargetOpcode::G_SEXT:
  case TargetOpcode::G_ANYEXT:
    return legalizeGExt(MI, B);
  case TargetOpcode::G_SITOFP:
  case TargetOpcode::G_UITOFP:
    return legalizeGItofp(MI, B);
  case TargetOpcode::G_STORE:
  case TargetOpcode::G_LOAD:
    return legalizeGLoad(MI, B, Helper);
  case TargetOpcode::G_SEXTLOAD:
  case TargetOpcode::G_ZEXTLOAD:
    return legalizeGExtload(MI, B);
  case TargetOpcode::G_FDIV:
    return legalizeGFdiv(MI, B);
  case TargetOpcode::G_FREM:
    return legalizeGFrem(MI, B);
  case TargetOpcode::G_INSERT_VECTOR_ELT:
    return legalizeGInsertVectorElt(Helper, MI, B);
  case TargetOpcode::G_EXTRACT_VECTOR_ELT:
    return legalizeGExtractVectorElt(Helper, MI, B);
  case TargetOpcode::G_INSERT_SUBVECTOR:
    return legalizeGInsertSubvector(MI, B);
  case TargetOpcode::G_EXTRACT_SUBVECTOR:
    return legalizeGExtractSubvector(MI, B);
  case TargetOpcode::G_CONCAT_VECTORS:
    return legalizeGConcatVectors(MI, B);
  case TargetOpcode::G_UNMERGE_VALUES:
    return legalizeGUnmergeValues(Helper, MI, B);
  case TargetOpcode::G_BSWAP:
    return legalizeGBswap(MI, B);
  case TargetOpcode::G_FPOW:
    return legalizeGFpow(MI, B);
  case TargetOpcode::G_FLDEXP:
  case TargetOpcode::G_STRICT_FLDEXP:
    return legalizeGFldexp(MI, B);
  case TargetOpcode::G_ATOMICRMW_XCHG:
    return legalizeGAtomicrmwXchg(MI, B);
  case TargetOpcode::G_SHUFFLE_VECTOR:
    return legalizeGShuffleVector(Helper, MI, B);
  case TargetOpcode::G_IS_FPCLASS:
    return legalizeGIsFpclass(Helper, MI, B);
  case TargetOpcode::G_UMULH:
  case TargetOpcode::G_SMULH:
    return legalizeGMulh(Helper, MI, B);
  case TargetOpcode::G_ADDRSPACE_CAST:
    return legalizeGAddrspaceCast(MI, B);
  case TargetOpcode::G_BITCAST:
    return legalizeGBitcast(MI, B);
  case TargetOpcode::G_ATOMIC_CMPXCHG:
  case TargetOpcode::G_ATOMICRMW_ADD:
  case TargetOpcode::G_ATOMICRMW_SUB:
  case TargetOpcode::G_ATOMICRMW_AND:
  case TargetOpcode::G_ATOMICRMW_OR:
  case TargetOpcode::G_ATOMICRMW_XOR:
  case TargetOpcode::G_ATOMICRMW_MIN:
  case TargetOpcode::G_ATOMICRMW_MAX:
  case TargetOpcode::G_ATOMICRMW_UMIN:
  case TargetOpcode::G_ATOMICRMW_UMAX:
  case TargetOpcode::G_ATOMICRMW_UINC_WRAP:
  case TargetOpcode::G_ATOMICRMW_UDEC_WRAP:
  case TargetOpcode::G_ATOMICRMW_FADD:
  case TargetOpcode::G_ATOMICRMW_FSUB:
  case TargetOpcode::G_ATOMICRMW_FMIN:
  case TargetOpcode::G_ATOMICRMW_FMAX:
    return legalizeGAtomicrmw(MI, B);
  case TargetOpcode::G_FABS:
    return legalizeFAbs(MI, B);
  }
  assert(0 && "unhandled!");
  return false;
}
