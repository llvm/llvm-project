//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/RuntimeLibcalls.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

TEST(RuntimeLibcallsTest, LibcallImplByName) {
  EXPECT_TRUE(RTLIB::RuntimeLibcallsInfo::lookupLibcallImplName("").empty());
  EXPECT_TRUE(
      RTLIB::RuntimeLibcallsInfo::lookupLibcallImplName("unknown").empty());
  EXPECT_TRUE(
      RTLIB::RuntimeLibcallsInfo::lookupLibcallImplName("Unsupported").empty());
  EXPECT_TRUE(
      RTLIB::RuntimeLibcallsInfo::lookupLibcallImplName("unsupported").empty());

  for (RTLIB::LibcallImpl LC : RTLIB::libcall_impls()) {
    StringRef Name = RTLIB::RuntimeLibcallsInfo::getLibcallImplName(LC);
    EXPECT_TRUE(is_contained(
        RTLIB::RuntimeLibcallsInfo::lookupLibcallImplName(Name), LC));
  }

  // Test first libcall name
  EXPECT_EQ(
      RTLIB::impl_arm64ec__Unwind_Resume,
      *RTLIB::RuntimeLibcallsInfo::lookupLibcallImplName("#_Unwind_Resume")
           .begin());
  // Test longest libcall names
  EXPECT_EQ(RTLIB::impl___hexagon_memcpy_likely_aligned_min32bytes_mult8bytes,
            *RTLIB::RuntimeLibcallsInfo::lookupLibcallImplName(
                 "__hexagon_memcpy_likely_aligned_min32bytes_mult8bytes")
                 .begin());

  {
    auto SquirtleSquad =
        RTLIB::RuntimeLibcallsInfo::lookupLibcallImplName("sqrtl");
    ASSERT_EQ(size(SquirtleSquad), 3);
    auto I = SquirtleSquad.begin();
    EXPECT_EQ(*I++, RTLIB::impl_sqrtl_f128);
    EXPECT_EQ(*I++, RTLIB::impl_sqrtl_f80);
    EXPECT_EQ(*I++, RTLIB::impl_sqrtl_ppcf128);
  }

  // Last libcall
  {
    auto Truncs = RTLIB::RuntimeLibcallsInfo::lookupLibcallImplName("truncl");
    ASSERT_EQ(size(Truncs), 3);
    auto I = Truncs.begin();
    EXPECT_EQ(*I++, RTLIB::impl_truncl_f128);
    EXPECT_EQ(*I++, RTLIB::impl_truncl_f80);
    EXPECT_EQ(*I++, RTLIB::impl_truncl_ppcf128);
  }
}

TEST(RuntimeLibcallsTest, LibcallForIntrinsic) {
  LLVMContext Ctx;
  using Info = RTLIB::RuntimeLibcallsInfo;
  auto UnarySig = [&](Type *Ty) { return FunctionType::get(Ty, {Ty}, false); };
  auto BinarySig = [&](Type *Ty) {
    return FunctionType::get(Ty, {Ty, Ty}, false);
  };
  auto FpIntSig = [&](Type *Ty) {
    return FunctionType::get(Ty, {Ty, Type::getInt32Ty(Ctx)}, false);
  };
  auto ConstrainedUnarySig = [&](Type *Ty) {
    return FunctionType::get(
        Ty, {Ty, Type::getMetadataTy(Ctx), Type::getMetadataTy(Ctx)}, false);
  };
  auto ConstrainedFmaSig = [&](Type *Ty) {
    return FunctionType::get(
        Ty, {Ty, Ty, Ty, Type::getMetadataTy(Ctx), Type::getMetadataTy(Ctx)},
        false);
  };

  // Per-type resolution.
  EXPECT_EQ(Info::getLibcallForIntrinsic(Intrinsic::sin,
                                         UnarySig(Type::getFloatTy(Ctx))),
            RTLIB::SIN_F32);
  EXPECT_EQ(Info::getLibcallForIntrinsic(Intrinsic::sin,
                                         UnarySig(Type::getDoubleTy(Ctx))),
            RTLIB::SIN_F64);
  EXPECT_EQ(Info::getLibcallForIntrinsic(Intrinsic::sin,
                                         UnarySig(Type::getFP128Ty(Ctx))),
            RTLIB::SIN_F128);

  EXPECT_EQ(Info::getLibcallForIntrinsic(Intrinsic::cos,
                                         UnarySig(Type::getDoubleTy(Ctx))),
            RTLIB::COS_F64);
  EXPECT_EQ(Info::getLibcallForIntrinsic(Intrinsic::pow,
                                         BinarySig(Type::getFloatTy(Ctx))),
            RTLIB::POW_F32);
  EXPECT_EQ(Info::getLibcallForIntrinsic(Intrinsic::ldexp,
                                         FpIntSig(Type::getDoubleTy(Ctx))),
            RTLIB::LDEXP_F64);
  EXPECT_EQ(Info::getLibcallForIntrinsic(Intrinsic::sqrt,
                                         UnarySig(Type::getFloatTy(Ctx))),
            RTLIB::SQRT_F32);
  EXPECT_EQ(Info::getLibcallForIntrinsic(Intrinsic::exp,
                                         UnarySig(Type::getDoubleTy(Ctx))),
            RTLIB::EXP_F64);
  EXPECT_EQ(Info::getLibcallForIntrinsic(Intrinsic::log,
                                         UnarySig(Type::getX86_FP80Ty(Ctx))),
            RTLIB::LOG_F80);
  EXPECT_EQ(Info::getLibcallForIntrinsic(Intrinsic::copysign,
                                         BinarySig(Type::getDoubleTy(Ctx))),
            RTLIB::COPYSIGN_F64);
  EXPECT_EQ(Info::getLibcallForIntrinsic(Intrinsic::maxnum,
                                         BinarySig(Type::getDoubleTy(Ctx))),
            RTLIB::FMAX_F64);

  // Constrained variants resolve to the same libcall as their plain twin.
  EXPECT_EQ(
      Info::getLibcallForIntrinsic(Intrinsic::experimental_constrained_sin,
                                   ConstrainedUnarySig(Type::getFloatTy(Ctx))),
      RTLIB::SIN_F32);
  EXPECT_EQ(
      Info::getLibcallForIntrinsic(Intrinsic::experimental_constrained_fma,
                                   ConstrainedFmaSig(Type::getDoubleTy(Ctx))),
      RTLIB::FMA_F64);
  EXPECT_EQ(
      Info::getLibcallForIntrinsic(Intrinsic::experimental_constrained_pow,
                                   ConstrainedUnarySig(Type::getDoubleTy(Ctx))),
      RTLIB::POW_F64);

  // Intrinsics whose result is not the floating-point type resolve from the
  // first floating-point argument instead.
  Type *Dbl = Type::getDoubleTy(Ctx);
  EXPECT_EQ(Info::getLibcallForIntrinsic(
                Intrinsic::lround,
                FunctionType::get(Type::getInt32Ty(Ctx), {Dbl}, false)),
            RTLIB::LROUND_F64);
  EXPECT_EQ(Info::getLibcallForIntrinsic(
                Intrinsic::llround,
                FunctionType::get(Type::getInt64Ty(Ctx), {Dbl}, false)),
            RTLIB::LLROUND_F64);
  EXPECT_EQ(
      Info::getLibcallForIntrinsic(
          Intrinsic::frexp,
          FunctionType::get(StructType::get(Ctx, {Dbl, Type::getInt32Ty(Ctx)}),
                            {Dbl}, false)),
      RTLIB::FREXP_F64);
  EXPECT_EQ(
      Info::getLibcallForIntrinsic(
          Intrinsic::sincos,
          FunctionType::get(StructType::get(Ctx, {Dbl, Dbl}), {Dbl}, false)),
      RTLIB::SINCOS_F64);
  EXPECT_EQ(
      Info::getLibcallForIntrinsic(
          Intrinsic::modf,
          FunctionType::get(StructType::get(Ctx, {Dbl, Dbl}), {Dbl}, false)),
      RTLIB::MODF_F64);

  // Unmapped intrinsic.
  EXPECT_EQ(Info::getLibcallForIntrinsic(Intrinsic::fabs,
                                         UnarySig(Type::getDoubleTy(Ctx))),
            RTLIB::UNKNOWN_LIBCALL);

  // Mapped intrinsic, but a type with no matching libcall.
  EXPECT_EQ(Info::getLibcallForIntrinsic(Intrinsic::sin,
                                         UnarySig(Type::getHalfTy(Ctx))),
            RTLIB::UNKNOWN_LIBCALL);

  // Vectors are not currently mapped.
  EXPECT_EQ(Info::getLibcallForIntrinsic(
                Intrinsic::sin,
                UnarySig(FixedVectorType::get(Type::getFloatTy(Ctx), 4))),
            RTLIB::UNKNOWN_LIBCALL);

  // Mapped intrinsic, but a signature with no floating-point argument/result.
  auto *NonFPSig =
      FunctionType::get(Type::getVoidTy(Ctx), {Type::getInt32Ty(Ctx)}, false);
  EXPECT_EQ(Info::getLibcallForIntrinsic(Intrinsic::sin, NonFPSig),
            RTLIB::UNKNOWN_LIBCALL);
}

} // namespace
