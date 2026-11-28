//===-- SparcTargetParserTest.cpp - SPARC TargetParser unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/SparcTargetParser.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(SparcTargetParser, ValidCPU) {
  EXPECT_TRUE(Sparc::isValidCPU("generic"));
  EXPECT_TRUE(Sparc::isValidCPU("v8"));
  EXPECT_TRUE(Sparc::isValidCPU("v9"));
  EXPECT_TRUE(Sparc::isValidCPU("leon3"));
  EXPECT_TRUE(Sparc::isValidCPU("niagara4"));
  EXPECT_FALSE(Sparc::isValidCPU(""));
  EXPECT_FALSE(Sparc::isValidCPU("pwr8"));

  SmallVector<StringRef> CPUs;
  Sparc::fillValidCPUList(CPUs);
  EXPECT_TRUE(is_contained(CPUs, "leon5"));
  EXPECT_TRUE(is_contained(CPUs, "ultrasparc3"));
  EXPECT_FALSE(is_contained(CPUs, "pwr8"));
}

TEST(SparcTargetParser, NormalizedCPU) {
  EXPECT_EQ(Sparc::getNormalizedSparcTargetCPU(Triple("sparc-unknown-linux")),
            "v8");
  EXPECT_EQ(Sparc::getNormalizedSparcTargetCPU(Triple("sparcel-unknown-elf")),
            "v8");
  EXPECT_EQ(
      Sparc::getNormalizedSparcTargetCPU(Triple("sparc64-unknown-openbsd")),
      "v9");
  EXPECT_EQ(Sparc::getNormalizedSparcTargetCPU(Triple("sparcv9-sun-solaris")),
            "v9");
  EXPECT_EQ(Sparc::getNormalizedSparcTargetCPU(Triple("sparc-unknown-linux"),
                                               "leon3"),
            "leon3");
}

TEST(SparcTargetParser, DefaultFeatures) {
  EXPECT_FALSE(Sparc::getSparcDefaultTargetFeatures("pwr8").has_value());

  std::optional<StringMap<bool>> V9 =
      Sparc::getSparcDefaultTargetFeatures("v9");
  ASSERT_TRUE(V9.has_value());
  EXPECT_TRUE(V9->lookup("v9"));
  EXPECT_FALSE(V9->lookup("vis"));

  // Implied features are included.
  std::optional<StringMap<bool>> Niagara4 =
      Sparc::getSparcDefaultTargetFeatures("niagara4");
  ASSERT_TRUE(Niagara4.has_value());
  EXPECT_TRUE(Niagara4->lookup("crypto"));
  EXPECT_TRUE(Niagara4->lookup("osa2011"));
  EXPECT_TRUE(Niagara4->lookup("vis3"));

  std::optional<StringMap<bool>> Leon3 =
      Sparc::getSparcDefaultTargetFeatures("leon3");
  ASSERT_TRUE(Leon3.has_value());
  EXPECT_TRUE(Leon3->lookup("leon"));
  EXPECT_TRUE(Leon3->lookup("hasleoncasa"));
  EXPECT_FALSE(Leon3->lookup("v9"));

  // UT699 does not have CASA.
  std::optional<StringMap<bool>> UT699 =
      Sparc::getSparcDefaultTargetFeatures("ut699");
  ASSERT_TRUE(UT699.has_value());
  EXPECT_TRUE(UT699->lookup("leon"));
  EXPECT_FALSE(UT699->lookup("hasleoncasa"));
}

TEST(SparcTargetParser, ValidFeatureName) {
  EXPECT_TRUE(Sparc::isValidFeatureName("v9"));
  EXPECT_TRUE(Sparc::isValidFeatureName("v8plus"));
  EXPECT_TRUE(Sparc::isValidFeatureName("hasleoncasa"));
  EXPECT_FALSE(Sparc::isValidFeatureName("altivec"));
}

} // namespace
