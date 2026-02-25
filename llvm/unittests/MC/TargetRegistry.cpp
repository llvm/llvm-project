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

TEST(TargetRegistry, IsValidFeatureListFormat) {
  // Valid strings

  // Empty string is a valid feature string
  EXPECT_TRUE(Target::isValidFeatureListFormat(""));

  EXPECT_TRUE(Target::isValidFeatureListFormat("+some_feature"));
  EXPECT_TRUE(Target::isValidFeatureListFormat("-some_feature"));
  EXPECT_TRUE(
      Target::isValidFeatureListFormat("+feature1,-feature2,+feature3"));
  EXPECT_TRUE(Target::isValidFeatureListFormat("+123"));

  // Invalid strings

  // Feature don't start with '+' or '-'
  EXPECT_FALSE(Target::isValidFeatureListFormat("invalid_string"));
  EXPECT_FALSE(Target::isValidFeatureListFormat("+good,bad"));
  EXPECT_FALSE(Target::isValidFeatureListFormat("bad,+good"));

  // String has spaces
  EXPECT_FALSE(Target::isValidFeatureListFormat(" "));
  EXPECT_FALSE(Target::isValidFeatureListFormat(", "));
  EXPECT_FALSE(Target::isValidFeatureListFormat(" avx"));
  EXPECT_FALSE(Target::isValidFeatureListFormat("+avx, -sse"));

  // Redundant commas
  EXPECT_FALSE(Target::isValidFeatureListFormat("+feature1,,+feature2"));
  EXPECT_FALSE(Target::isValidFeatureListFormat(",+feature"));
  EXPECT_FALSE(Target::isValidFeatureListFormat("-feature,"));
  EXPECT_FALSE(
      Target::isValidFeatureListFormat("+feature1,,,+feature2,,+feature3"));

  // Feature consists only of '+' or '-'
  EXPECT_FALSE(Target::isValidFeatureListFormat("+"));
  EXPECT_FALSE(Target::isValidFeatureListFormat("-"));
  EXPECT_FALSE(Target::isValidFeatureListFormat("+avx,-"));

  // Only commas
  EXPECT_FALSE(Target::isValidFeatureListFormat(","));
  EXPECT_FALSE(Target::isValidFeatureListFormat(",,"));
  EXPECT_FALSE(Target::isValidFeatureListFormat(",,,"));
}

} // end namespace
