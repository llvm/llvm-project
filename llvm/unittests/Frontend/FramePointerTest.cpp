//===- llvm/unittests/Frontend/FramePointerTest.cpp - FP tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/Driver/CodeGenOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::driver;

namespace {

TEST(FramePointerTest, TargetDefaults) {
  struct TestCase {
    const char *Triple;
    bool Optimized;
    FramePointerKind Expected;
  };
  const TestCase Cases[] = {
      {"i386-unknown-linux", false, FramePointerKind::All},
      {"i386-unknown-linux", true, FramePointerKind::None},
      {"thumb-arm-none-eabi", false, FramePointerKind::None},
      {"thumbv6m-apple-none-macho", false, FramePointerKind::NonLeafNoReserve},
      {"riscv64-unknown-linux-android", true,
       FramePointerKind::NonLeafNoReserve},
  };

  for (const TestCase &Case : Cases) {
    FramePointerOptions Opts;
    Opts.Optimized = Case.Optimized;
    EXPECT_EQ(Case.Expected, getFramePointerKind(Triple(Case.Triple), Opts))
        << Case.Triple;
  }
}

TEST(FramePointerTest, ExplicitOptions) {
  FramePointerOptions Opts;
  Opts.Optimized = true;
  Opts.EnableFramePointer = true;
  EXPECT_EQ(FramePointerKind::All,
            getFramePointerKind(Triple("i386-unknown-linux"), Opts));

  Opts.EnableLeafFramePointer = false;
  EXPECT_EQ(FramePointerKind::NonLeafNoReserve,
            getFramePointerKind(Triple("i386-unknown-linux"), Opts));

  Opts.ReserveFramePointerRegister = true;
  EXPECT_EQ(FramePointerKind::NonLeaf,
            getFramePointerKind(Triple("i386-unknown-linux"), Opts));

  Opts.EnableFramePointer = false;
  EXPECT_EQ(FramePointerKind::Reserved,
            getFramePointerKind(Triple("i386-unknown-linux"), Opts));
}

TEST(FramePointerTest, InstrumentationRequiresFramePointer) {
  FramePointerOptions Opts;
  Opts.Optimized = true;
  Opts.InstrumentationRequiresFramePointer = true;
  EXPECT_EQ(FramePointerKind::All,
            getFramePointerKind(Triple("i386-unknown-linux"), Opts));
}

TEST(FramePointerTest, FrameChain) {
  FramePointerOptions Opts;
  Opts.MaintainValidFrameChain = true;
  EXPECT_EQ(FramePointerKind::Reserved,
            getFramePointerKind(Triple("arm-arm-none-eabi"), Opts));

  Opts.FramePointerImpliesLeaf = true;
  Opts.EnableFramePointer = true;
  EXPECT_EQ(FramePointerKind::All,
            getFramePointerKind(Triple("arm-arm-none-eabi"), Opts));
}

TEST(FramePointerTest, AArch64WindowsMaintainsFrameChain) {
  FramePointerOptions Opts;
  Opts.EnableFramePointer = false;
  EXPECT_EQ(FramePointerKind::Reserved,
            getFramePointerKind(Triple("aarch64-pc-windows-msvc"), Opts));
}

} // namespace
