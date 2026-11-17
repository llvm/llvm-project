//===- IntrinsicFeatureEmitterTest.cpp - Generated table tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGenTypes/MachineValueType.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IntrinsicFeatureGuard.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "gtest/gtest.h"
#include <cassert>

using namespace llvm;

// The TableGen input uses a synthetic target and intrinsics. Supply the small
// portion of their C++ declarations needed by the generated table.
namespace TestTarget {
enum { FeatureExample = 0, FeatureOther = 1 };
}

namespace llvm::Intrinsic {
constexpr ID test_one_guard = 100000;
constexpr ID test_typed = 100001;
constexpr ID test_two_guards = 100002;
constexpr ID test_type_specific_guards = 100003;
} // namespace llvm::Intrinsic

namespace {

#include "IntrinsicFeatureChecksGuards.inc"

TEST(IntrinsicFeatureEmitterTest, SingleGuard) {
  ArrayRef<IntrinsicFeatureVariant> Untyped =
      getTestTargetIntrinsicVariants(Intrinsic::test_one_guard);
  ASSERT_EQ(Untyped.size(), 1u);
  EXPECT_TRUE(Untyped[0].AppliesToAllTypes);
  EXPECT_EQ(Untyped[0].NumGuards, 1u);
  const IntrinsicFeatureGuard &UntypedGuard =
      getTestTargetIntrinsicFeatureGuard(Untyped[0], 0);
  EXPECT_EQ(UntypedGuard.FirstClause, 0u);
  EXPECT_EQ(UntypedGuard.NumClauses, 1u);
  EXPECT_STREQ(UntypedGuard.Expression, "example");
  ArrayRef<FeatureBitset> Alternatives =
      getTestTargetIntrinsicFeatureAlternatives(UntypedGuard);
  ASSERT_EQ(Alternatives.size(), 1u);
  EXPECT_EQ(Alternatives[0], FeatureBitset{TestTarget::FeatureExample});

  ArrayRef<IntrinsicFeatureVariant> Typed =
      getTestTargetIntrinsicVariants(Intrinsic::test_typed);
  ASSERT_EQ(Typed.size(), 1u);
  EXPECT_FALSE(Typed[0].AppliesToAllTypes);
  EXPECT_EQ(Typed[0].NumTypes, 0u);
  EXPECT_EQ(Typed[0].NumGuards, 1u);
  EXPECT_EQ(&getTestTargetIntrinsicFeatureGuard(Typed[0], 0), &UntypedGuard);

  EXPECT_TRUE(getTestTargetIntrinsicVariants(0).empty());
}

TEST(IntrinsicFeatureEmitterTest, MultipleGuards) {
  ArrayRef<IntrinsicFeatureVariant> Variants =
      getTestTargetIntrinsicVariants(Intrinsic::test_two_guards);
  ASSERT_EQ(Variants.size(), 1u);
  EXPECT_TRUE(Variants[0].AppliesToAllTypes);
  ASSERT_EQ(Variants[0].NumGuards, 2u);

  const IntrinsicFeatureGuard &First =
      getTestTargetIntrinsicFeatureGuard(Variants[0], 0);
  const IntrinsicFeatureGuard &Second =
      getTestTargetIntrinsicFeatureGuard(Variants[0], 1);
  EXPECT_STREQ(First.Expression, "example");
  EXPECT_STREQ(Second.Expression, "other");
  EXPECT_NE(&First, &Second);

  ArrayRef<FeatureBitset> FirstAlternatives =
      getTestTargetIntrinsicFeatureAlternatives(First);
  ASSERT_EQ(FirstAlternatives.size(), 1u);
  EXPECT_EQ(FirstAlternatives[0], FeatureBitset{TestTarget::FeatureExample});
  ArrayRef<FeatureBitset> SecondAlternatives =
      getTestTargetIntrinsicFeatureAlternatives(Second);
  ASSERT_EQ(SecondAlternatives.size(), 1u);
  EXPECT_EQ(SecondAlternatives[0], FeatureBitset{TestTarget::FeatureOther});
}

TEST(IntrinsicFeatureEmitterTest, TypeSpecificGuards) {
  ArrayRef<IntrinsicFeatureVariant> Variants =
      getTestTargetIntrinsicVariants(Intrinsic::test_type_specific_guards);
  ASSERT_EQ(Variants.size(), 2u);
  EXPECT_FALSE(Variants[0].AppliesToAllTypes);
  EXPECT_FALSE(Variants[1].AppliesToAllTypes);
  EXPECT_EQ(Variants[0].NumTypes, 1u);
  EXPECT_EQ(Variants[1].NumTypes, 1u);
  EXPECT_NE(Variants[0].FirstType, Variants[1].FirstType);

  const IntrinsicFeatureGuard &I32Guard =
      getTestTargetIntrinsicFeatureGuard(Variants[0], 0);
  const IntrinsicFeatureGuard &I64Guard =
      getTestTargetIntrinsicFeatureGuard(Variants[1], 0);
  EXPECT_STREQ(I32Guard.Expression, "example");
  EXPECT_STREQ(I64Guard.Expression, "other");

  ArrayRef<FeatureBitset> I32Alternatives =
      getTestTargetIntrinsicFeatureAlternatives(I32Guard);
  ArrayRef<FeatureBitset> I64Alternatives =
      getTestTargetIntrinsicFeatureAlternatives(I64Guard);
  ASSERT_EQ(I32Alternatives.size(), 1u);
  ASSERT_EQ(I64Alternatives.size(), 1u);
  EXPECT_EQ(I32Alternatives[0], FeatureBitset{TestTarget::FeatureExample});
  EXPECT_EQ(I64Alternatives[0], FeatureBitset{TestTarget::FeatureOther});
}

} // namespace
