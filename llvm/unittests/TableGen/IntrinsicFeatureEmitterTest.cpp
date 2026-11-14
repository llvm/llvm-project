//===- IntrinsicFeatureEmitterTest.cpp - Generated table tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/CodeGenTypes/MachineValueType.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "gtest/gtest.h"

using namespace llvm;

// The TableGen input uses a synthetic target and intrinsics. Supply the small
// portion of their C++ declarations needed by the generated table.
namespace TestTarget {
enum { FeatureExample = 0 };
}

namespace llvm::Intrinsic {
constexpr ID test_one_guard = 100000;
constexpr ID test_typed = 100001;
} // namespace llvm::Intrinsic

namespace {

struct IntrinsicFeatureGuard {
  unsigned FirstClause;
  unsigned NumClauses;
  const char *Expression;
};

struct IntrinsicFeatureVariant {
  unsigned FirstType;
  unsigned NumResults;
  unsigned NumInputs;
  unsigned FirstGuard;
  unsigned NumGuards;
  bool AppliesToAllTypes;
};

#include "IntrinsicFeatureChecksSingleGuard.inc"

TEST(IntrinsicFeatureEmitterTest, SingleGuard) {
  ASSERT_EQ(ArrayRef(TestTargetIntrinsicFeatureClauses).size(), 1u);
  EXPECT_EQ(TestTargetIntrinsicFeatureClauses[0],
            FeatureBitset{TestTarget::FeatureExample});

  ASSERT_EQ(ArrayRef(TestTargetIntrinsicFeatureGuards).size(), 1u);
  const auto &Guard = TestTargetIntrinsicFeatureGuards[0];
  EXPECT_EQ(Guard.FirstClause, 0u);
  EXPECT_EQ(Guard.NumClauses, 1u);
  EXPECT_STREQ(Guard.Expression, "example");

  ASSERT_EQ(ArrayRef(TestTargetIntrinsicTypes).size(), 2u);
  ASSERT_EQ(ArrayRef(TestTargetIntrinsicGuardIndices).size(), 2u);
  ASSERT_EQ(ArrayRef(TestTargetIntrinsicVariants).size(), 2u);

  auto Untyped = getTestTargetIntrinsicVariants(Intrinsic::test_one_guard);
  ASSERT_EQ(Untyped.size(), 1u);
  EXPECT_TRUE(Untyped[0].AppliesToAllTypes);
  EXPECT_EQ(Untyped[0].NumGuards, 1u);
  EXPECT_EQ(TestTargetIntrinsicGuardIndices[Untyped[0].FirstGuard], 0u);

  auto Typed = getTestTargetIntrinsicVariants(Intrinsic::test_typed);
  ASSERT_EQ(Typed.size(), 1u);
  EXPECT_FALSE(Typed[0].AppliesToAllTypes);
  EXPECT_EQ(Typed[0].NumResults, 1u);
  EXPECT_EQ(Typed[0].NumInputs, 1u);
  EXPECT_EQ(Typed[0].NumGuards, 1u);
  EXPECT_EQ(TestTargetIntrinsicTypes[Typed[0].FirstType], MVT::i32);
  EXPECT_EQ(TestTargetIntrinsicTypes[Typed[0].FirstType + 1], MVT::i32);
  EXPECT_EQ(TestTargetIntrinsicGuardIndices[Typed[0].FirstGuard], 0u);

  EXPECT_TRUE(getTestTargetIntrinsicVariants(0).empty());
}

} // namespace
