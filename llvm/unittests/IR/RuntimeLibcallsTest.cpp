//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/RuntimeLibcalls.h"
#include "llvm/ADT/STLExtras.h"
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
    EXPECT_EQ(*I++, RTLIB::impl_sqrt_f128);
    EXPECT_EQ(*I++, RTLIB::impl_sqrt_f80);
    EXPECT_EQ(*I++, RTLIB::impl_sqrt_ppcf128);
  }

  // Last libcall
  {
    auto Truncs = RTLIB::RuntimeLibcallsInfo::lookupLibcallImplName("truncl");
    ASSERT_EQ(size(Truncs), 3);
    auto I = Truncs.begin();
    EXPECT_EQ(*I++, RTLIB::impl_trunc_f128);
    EXPECT_EQ(*I++, RTLIB::impl_trunc_f80);
    EXPECT_EQ(*I++, RTLIB::impl_trunc_ppcf128);
  }
}

} // namespace
