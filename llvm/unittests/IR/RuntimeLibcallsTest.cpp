//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/RuntimeLibcalls.h"
#include "llvm/ADT/STLExtras.h"
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

  // Per-type resolution.
  EXPECT_EQ(Info::getLibcallForIntrinsic(Intrinsic::sin, Type::getFloatTy(Ctx)),
            RTLIB::SIN_F32);
  EXPECT_EQ(
      Info::getLibcallForIntrinsic(Intrinsic::sin, Type::getDoubleTy(Ctx)),
      RTLIB::SIN_F64);
  EXPECT_EQ(Info::getLibcallForIntrinsic(Intrinsic::sin, Type::getFP128Ty(Ctx)),
            RTLIB::SIN_F128);

  EXPECT_EQ(
      Info::getLibcallForIntrinsic(Intrinsic::cos, Type::getDoubleTy(Ctx)),
      RTLIB::COS_F64);
  EXPECT_EQ(Info::getLibcallForIntrinsic(Intrinsic::pow, Type::getFloatTy(Ctx)),
            RTLIB::POW_F32);
  EXPECT_EQ(
      Info::getLibcallForIntrinsic(Intrinsic::ldexp, Type::getDoubleTy(Ctx)),
      RTLIB::LDEXP_F64);
  EXPECT_EQ(
      Info::getLibcallForIntrinsic(Intrinsic::sqrt, Type::getFloatTy(Ctx)),
      RTLIB::SQRT_F32);
  EXPECT_EQ(
      Info::getLibcallForIntrinsic(Intrinsic::exp, Type::getDoubleTy(Ctx)),
      RTLIB::EXP_F64);
  EXPECT_EQ(
      Info::getLibcallForIntrinsic(Intrinsic::log, Type::getX86_FP80Ty(Ctx)),
      RTLIB::LOG_F80);
  EXPECT_EQ(
      Info::getLibcallForIntrinsic(Intrinsic::copysign, Type::getDoubleTy(Ctx)),
      RTLIB::COPYSIGN_F64);
  EXPECT_EQ(
      Info::getLibcallForIntrinsic(Intrinsic::maxnum, Type::getDoubleTy(Ctx)),
      RTLIB::FMAX_F64);

  // Constrained variants resolve to the same libcall as their plain twin.
  EXPECT_EQ(Info::getLibcallForIntrinsic(
                Intrinsic::experimental_constrained_sin, Type::getFloatTy(Ctx)),
            RTLIB::SIN_F32);
  EXPECT_EQ(
      Info::getLibcallForIntrinsic(Intrinsic::experimental_constrained_fma,
                                   Type::getDoubleTy(Ctx)),
      RTLIB::FMA_F64);
  EXPECT_EQ(
      Info::getLibcallForIntrinsic(Intrinsic::experimental_constrained_pow,
                                   Type::getDoubleTy(Ctx)),
      RTLIB::POW_F64);

  // Unmapped intrinsic.
  EXPECT_EQ(
      Info::getLibcallForIntrinsic(Intrinsic::fabs, Type::getDoubleTy(Ctx)),
      RTLIB::UNKNOWN_LIBCALL);

  // Mapped intrinsic, but a type with no matching libcall.
  EXPECT_EQ(Info::getLibcallForIntrinsic(Intrinsic::sin, Type::getHalfTy(Ctx)),
            RTLIB::UNKNOWN_LIBCALL);
}

} // namespace
