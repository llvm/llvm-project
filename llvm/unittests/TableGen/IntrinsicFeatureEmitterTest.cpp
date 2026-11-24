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
#include "llvm/IR/GuardedIntrinsicInfo.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "gtest/gtest.h"
#include <cassert>
#include <iterator>

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

#include "IntrinsicFeatureChecksGuards.inc"

namespace {

struct TestSubtarget {
  FeatureBitset Bits;
  bool Enabled;
  const FeatureBitset &getFeatureBits() const { return Bits; }
};

static const GuardedIntrinsicInfo<TestSubtarget> &getInfo() {
  return getTestTargetGeneratedGuardedIntrinsicInfo<TestSubtarget>();
}

TEST(IntrinsicFeatureEmitterTest, SingleGuard) {
  ArrayRef<IntrinsicFeatureVariant> Untyped =
      getInfo().getVariants(Intrinsic::test_one_guard);
  ASSERT_EQ(Untyped.size(), 1u);
  EXPECT_EQ(Untyped[0].NumOverloadTypesToCheck, 0u);
  EXPECT_EQ(Untyped[0].NumGuards, 1u);
  const IntrinsicFeatureGuard &UntypedGuard = getInfo().getGuard(Untyped[0], 0);
  EXPECT_EQ(UntypedGuard.FirstClause, 0u);
  EXPECT_EQ(UntypedGuard.NumClauses, 1u);
  EXPECT_STREQ(UntypedGuard.Expression, "example");
  ArrayRef<FeatureBitset> Alternatives =
      getInfo().getAlternatives(UntypedGuard);
  ASSERT_EQ(Alternatives.size(), 1u);
  EXPECT_EQ(Alternatives[0], FeatureBitset{TestTarget::FeatureExample});

  ArrayRef<IntrinsicFeatureVariant> Typed =
      getInfo().getVariants(Intrinsic::test_typed);
  ASSERT_EQ(Typed.size(), 1u);
  EXPECT_EQ(Typed[0].NumOverloadTypesToCheck, 0u);
  EXPECT_EQ(Typed[0].NumGuards, 1u);
  EXPECT_EQ(&getInfo().getGuard(Typed[0], 0), &UntypedGuard);

  EXPECT_TRUE(getInfo().getVariants(static_cast<Intrinsic::ID>(0)).empty());
}

TEST(IntrinsicFeatureEmitterTest, MultipleGuards) {
  ArrayRef<IntrinsicFeatureVariant> Variants =
      getInfo().getVariants(Intrinsic::test_two_guards);
  ASSERT_EQ(Variants.size(), 1u);
  EXPECT_EQ(Variants[0].NumOverloadTypesToCheck, 0u);
  ASSERT_EQ(Variants[0].NumGuards, 2u);

  const IntrinsicFeatureGuard &First = getInfo().getGuard(Variants[0], 0);
  const IntrinsicFeatureGuard &Second = getInfo().getGuard(Variants[0], 1);
  EXPECT_STREQ(First.Expression, "example");
  EXPECT_STREQ(Second.Expression, "other");
  EXPECT_NE(&First, &Second);

  ArrayRef<FeatureBitset> FirstAlternatives = getInfo().getAlternatives(First);
  ASSERT_EQ(FirstAlternatives.size(), 1u);
  EXPECT_EQ(FirstAlternatives[0], FeatureBitset{TestTarget::FeatureExample});
  ArrayRef<FeatureBitset> SecondAlternatives =
      getInfo().getAlternatives(Second);
  ASSERT_EQ(SecondAlternatives.size(), 1u);
  EXPECT_EQ(SecondAlternatives[0], FeatureBitset{TestTarget::FeatureOther});

  TestSubtarget ST{{TestTarget::FeatureOther}, false};
  EXPECT_EQ(getInfo().checkGuard(First, ST),
            IntrinsicGuardCheckResult::MissingFeatures);
  EXPECT_EQ(getInfo().checkGuard(Second, ST),
            IntrinsicGuardCheckResult::Inactive);
  EXPECT_FALSE(getInfo().isVariantAvailable(Variants[0], ST));
  ST.Bits = {TestTarget::FeatureExample};
  EXPECT_TRUE(getInfo().isVariantAvailable(Variants[0], ST));
  ST.Bits = {TestTarget::FeatureOther};
  ST.Enabled = true;
  EXPECT_EQ(getInfo().checkGuard(Second, ST),
            IntrinsicGuardCheckResult::Available);
  EXPECT_FALSE(getInfo().isVariantAvailable(Variants[0], ST));
  ST.Bits = {TestTarget::FeatureExample, TestTarget::FeatureOther};
  EXPECT_TRUE(getInfo().isVariantAvailable(Variants[0], ST));
}

TEST(IntrinsicFeatureEmitterTest, TypeSpecificGuards) {
  ArrayRef<IntrinsicFeatureVariant> Variants =
      getInfo().getVariants(Intrinsic::test_type_specific_guards);
  ASSERT_EQ(Variants.size(), 2u);
  EXPECT_EQ(Variants[0].NumOverloadTypesToCheck, 1u);
  EXPECT_EQ(Variants[1].NumOverloadTypesToCheck, 1u);
  EXPECT_NE(Variants[0].FirstType, Variants[1].FirstType);
  ASSERT_EQ(Variants[0].NumGuards, 2u);
  ASSERT_EQ(Variants[1].NumGuards, 2u);

  const IntrinsicFeatureGuard &I32First = getInfo().getGuard(Variants[0], 0);
  const IntrinsicFeatureGuard &I32Second = getInfo().getGuard(Variants[0], 1);
  const IntrinsicFeatureGuard &I64First = getInfo().getGuard(Variants[1], 0);
  const IntrinsicFeatureGuard &I64Second = getInfo().getGuard(Variants[1], 1);
  EXPECT_STREQ(I32First.Expression, "example");
  EXPECT_STREQ(I32Second.Expression, "other");
  EXPECT_EQ(&I32First, &I64Second);
  EXPECT_NE(&I32Second, &I64First);
  EXPECT_EQ(I32First.Index, I32Second.Index);
  EXPECT_NE(I32Second.Index, I64First.Index);

  ArrayRef<FeatureBitset> I32Alternatives = getInfo().getAlternatives(I32First);
  ArrayRef<FeatureBitset> I64Alternatives = getInfo().getAlternatives(I64First);
  ASSERT_EQ(I32Alternatives.size(), 1u);
  ASSERT_EQ(I64Alternatives.size(), 1u);
  EXPECT_EQ(I32Alternatives[0], FeatureBitset{TestTarget::FeatureExample});
  EXPECT_EQ(I64Alternatives[0], FeatureBitset{TestTarget::FeatureOther});
}

} // namespace
