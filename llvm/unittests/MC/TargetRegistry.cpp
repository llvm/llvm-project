//===- unittests/MC/TargetRegistry.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The target registry code lives in Support, but it relies on linking in all
// LLVM targets. We keep this test with the MC tests, which already do that, to
// keep the SupportTests target small.

#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(TargetRegistry, TargetHasArchType) {
  // Presence of at least one target will be asserted when done with the loop,
  // else this would pass by accident if InitializeAllTargetInfos were omitted.
  int Count = 0;

  llvm::InitializeAllTargetInfos();

  for (const Target &T : TargetRegistry::targets()) {
    StringRef Name = T.getName();
    // There is really no way (at present) to ask a Target whether it targets
    // a specific architecture, because the logic for that is buried in a
    // predicate.
    // We can't ask the predicate "Are you a function that always returns
    // false?"
    // So given that the cpp backend truly has no target arch, it is skipped.
    if (Name != "cpp") {
      Triple::ArchType Arch = Triple::getArchTypeForLLVMName(Name);
      EXPECT_NE(Arch, Triple::UnknownArch);
      ++Count;
    }
  }
  ASSERT_NE(Count, 0);
}

TEST(TargetRegistry, PreserveSubArchInfo) {
  auto T = Triple("mipsisa64r6el-unknown-linux-gnuabi64");
  EXPECT_EQ(Triple::mips64el, T.getArch());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUABI64, T.getEnvironment());
  std::string Error;
  if (!TargetRegistry::lookupTarget("mips64el", T, Error))
    GTEST_SKIP();
  EXPECT_EQ(Triple::mips64el, T.getArch());
  // The SubArch information needs to be preserved.
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUABI64, T.getEnvironment());
}

TEST(TargetRegistry, UpdateSubArchInfo) {
  auto T = Triple("mipsisa64r6el-unknown-linux-gnuabi64");
  EXPECT_EQ(Triple::mips64el, T.getArch());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUABI64, T.getEnvironment());
  std::string Error;
  if (!TargetRegistry::lookupTarget("aarch64", T, Error))
    GTEST_SKIP();
  EXPECT_EQ(Triple::aarch64, T.getArch());
  // The SubArch will be updated to the new value corresponding to "aarch64".
  EXPECT_EQ(Triple::ARMSubArch_v8, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUABI64, T.getEnvironment());
  if (!TargetRegistry::lookupTarget("riscv64", T, Error))
    GTEST_SKIP();
  EXPECT_EQ(Triple::riscv64, T.getArch());
  // The SubArch will be updated to the new value corresponding to "riscv64".
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUABI64, T.getEnvironment());
}

} // end namespace
