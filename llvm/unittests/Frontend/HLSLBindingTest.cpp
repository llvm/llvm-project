//===------ HLSLBindingTest.cpp - Resource binding tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/HLSLBinding.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/DXILABI.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::dxil;

MATCHER_P(HasSpecificValue, Value, "") {
  return arg.has_value() && *arg == Value;
}

static void
checkExpectedSpaceAndFreeRanges(hlsl::BindingInfo::RegisterSpace &RegSpace,
                                uint32_t ExpSpace,
                                ArrayRef<uint32_t> ExpValues) {
  EXPECT_EQ(RegSpace.Space, ExpSpace);
  EXPECT_EQ(RegSpace.FreeRanges.size() * 2, ExpValues.size());
  unsigned I = 0;
  for (auto &R : RegSpace.FreeRanges) {
    EXPECT_EQ(R.LowerBound, ExpValues[I]);
    EXPECT_EQ(R.UpperBound, ExpValues[I + 1]);
    I += 2;
  }
}

TEST(HLSLBindingTest, TestTrivialCase) {
  hlsl::BindingInfoBuilder Builder;

  Builder.trackBinding(ResourceClass::UAV, /*Space=*/0, /*LowerBound=*/5,
                       /*UpperBound=*/5, /*Cookie=*/nullptr);
  bool HasOverlap;
  hlsl::BindingInfo Info = Builder.calculateBindingInfo(HasOverlap);

  EXPECT_FALSE(HasOverlap);

  // check that UAV has exactly one gap
  hlsl::BindingInfo::BindingSpaces &UAVSpaces =
      Info.getBindingSpaces(ResourceClass::UAV);
  EXPECT_EQ(UAVSpaces.RC, ResourceClass::UAV);
  EXPECT_EQ(UAVSpaces.Spaces.size(), 1u);
  checkExpectedSpaceAndFreeRanges(UAVSpaces.Spaces[0], 0, {0u, 4u, 6u, ~0u});

  // check that other kinds of register spaces are all available
  for (auto RC :
       {ResourceClass::SRV, ResourceClass::CBuffer, ResourceClass::Sampler}) {
    hlsl::BindingInfo::BindingSpaces &Spaces = Info.getBindingSpaces(RC);
    EXPECT_EQ(Spaces.RC, RC);
    EXPECT_EQ(Spaces.Spaces.size(), 0u);
  }
}

TEST(HLSLBindingTest, TestManyBindings) {
  hlsl::BindingInfoBuilder Builder;

  // cbuffer CB                 : register(b3) { int a; }
  // RWBuffer<float4> A[5]      : register(u10, space20);
  // StructuredBuffer<int> B    : register(t5);
  // RWBuffer<float> C          : register(u5);
  // StructuredBuffer<int> D[5] : register(t0);
  // RWBuffer<float> E[2]       : register(u2);
  // SamplerState S1            : register(s5, space2);
  // SamplerState S2            : register(s4, space2);
  Builder.trackBinding(ResourceClass::CBuffer, /*Space=*/0, /*LowerBound=*/3,
                       /*UpperBound=*/3, /*Cookie=*/nullptr);
  Builder.trackBinding(ResourceClass::UAV, /*Space=*/20, /*LowerBound=*/10,
                       /*UpperBound=*/14, /*Cookie=*/nullptr);
  Builder.trackBinding(ResourceClass::SRV, /*Space=*/0, /*LowerBound=*/5,
                       /*UpperBound=*/5, /*Cookie=*/nullptr);
  Builder.trackBinding(ResourceClass::UAV, /*Space=*/0, /*LowerBound=*/5,
                       /*UpperBound=*/5, /*Cookie=*/nullptr);
  Builder.trackBinding(ResourceClass::SRV, /*Space=*/0, /*LowerBound=*/0,
                       /*UpperBound=*/4, /*Cookie=*/nullptr);
  Builder.trackBinding(ResourceClass::UAV, /*Space=*/0, /*LowerBound=*/2,
                       /*UpperBound=*/3, /*Cookie=*/nullptr);
  Builder.trackBinding(ResourceClass::Sampler, /*Space=*/2, /*LowerBound=*/5,
                       /*UpperBound=*/5, /*Cookie=*/nullptr);
  Builder.trackBinding(ResourceClass::Sampler, /*Space=*/2, /*LowerBound=*/4,
                       /*UpperBound=*/4, /*Cookie=*/nullptr);
  bool HasOverlap;
  hlsl::BindingInfo Info = Builder.calculateBindingInfo(HasOverlap);

  EXPECT_FALSE(HasOverlap);

  hlsl::BindingInfo::BindingSpaces &SRVSpaces =
      Info.getBindingSpaces(ResourceClass::SRV);
  EXPECT_EQ(SRVSpaces.RC, ResourceClass::SRV);
  EXPECT_EQ(SRVSpaces.Spaces.size(), 1u);
  // verify that consecutive bindings are merged
  // (SRVSpaces has only one free space range {6, ~0u}).
  checkExpectedSpaceAndFreeRanges(SRVSpaces.Spaces[0], 0, {6u, ~0u});

  hlsl::BindingInfo::BindingSpaces &UAVSpaces =
      Info.getBindingSpaces(ResourceClass::UAV);
  EXPECT_EQ(UAVSpaces.RC, ResourceClass::UAV);
  EXPECT_EQ(UAVSpaces.Spaces.size(), 2u);
  checkExpectedSpaceAndFreeRanges(UAVSpaces.Spaces[0], 0,
                                  {0u, 1u, 4u, 4u, 6u, ~0u});
  checkExpectedSpaceAndFreeRanges(UAVSpaces.Spaces[1], 20, {0u, 9u, 15u, ~0u});

  hlsl::BindingInfo::BindingSpaces &CBufferSpaces =
      Info.getBindingSpaces(ResourceClass::CBuffer);
  EXPECT_EQ(CBufferSpaces.RC, ResourceClass::CBuffer);
  EXPECT_EQ(CBufferSpaces.Spaces.size(), 1u);
  checkExpectedSpaceAndFreeRanges(CBufferSpaces.Spaces[0], 0,
                                  {0u, 2u, 4u, ~0u});

  hlsl::BindingInfo::BindingSpaces &SamplerSpaces =
      Info.getBindingSpaces(ResourceClass::Sampler);
  EXPECT_EQ(SamplerSpaces.RC, ResourceClass::Sampler);
  EXPECT_EQ(SamplerSpaces.Spaces.size(), 1u);
  checkExpectedSpaceAndFreeRanges(SamplerSpaces.Spaces[0], 2,
                                  {0u, 3u, 6u, ~0u});
}

TEST(HLSLBindingTest, TestUnboundedAndOverlap) {
  hlsl::BindingInfoBuilder Builder;

  // StructuredBuffer<float> A[]  : register(t5);
  // StructuredBuffer<float> B[3] : register(t0);
  // StructuredBuffer<float> C[]  : register(t0, space2);
  // StructuredBuffer<float> D    : register(t4, space2); /* overlapping */
  Builder.trackBinding(ResourceClass::SRV, /*Space=*/0, /*LowerBound=*/5,
                       /*UpperBound=*/~0u, /*Cookie=*/nullptr);
  Builder.trackBinding(ResourceClass::SRV, /*Space=*/0, /*LowerBound=*/0,
                       /*UpperBound=*/2, /*Cookie=*/nullptr);
  Builder.trackBinding(ResourceClass::SRV, /*Space=*/2, /*LowerBound=*/0,
                       /*UpperBound=*/~0u, /*Cookie=*/nullptr);
  Builder.trackBinding(ResourceClass::SRV, /*Space=*/2, /*LowerBound=*/4,
                       /*UpperBound=*/4, /*Cookie=*/nullptr);
  bool HasOverlap;
  hlsl::BindingInfo Info = Builder.calculateBindingInfo(HasOverlap);

  EXPECT_TRUE(HasOverlap);

  hlsl::BindingInfo::BindingSpaces &SRVSpaces =
      Info.getBindingSpaces(ResourceClass::SRV);
  EXPECT_EQ(SRVSpaces.RC, ResourceClass::SRV);
  EXPECT_EQ(SRVSpaces.Spaces.size(), 2u);
  checkExpectedSpaceAndFreeRanges(SRVSpaces.Spaces[0], 0, {3, 4});
  checkExpectedSpaceAndFreeRanges(SRVSpaces.Spaces[1], 2, {});
}

TEST(HLSLBindingTest, TestExactOverlap) {
  hlsl::BindingInfoBuilder Builder;

  // Since the bindings overlap exactly we need sigil values to differentiate
  // them.
  // Note: We initialize these to 0 to suppress a -Wuninitialized-const-pointer,
  // but we really are just using the stack addresses here.
  char ID1 = 0;
  char ID2 = 0;

  // StructuredBuffer<float> A  : register(t5);
  // StructuredBuffer<float> B  : register(t5);
  Builder.trackBinding(ResourceClass::SRV, /*Space=*/0, /*LowerBound=*/5,
                       /*UpperBound=*/5, /*Cookie=*/&ID1);
  Builder.trackBinding(ResourceClass::SRV, /*Space=*/0, /*LowerBound=*/5,
                       /*UpperBound=*/5, /*Cookie=*/&ID2);
  bool HasOverlap;
  hlsl::BindingInfo Info = Builder.calculateBindingInfo(HasOverlap);

  EXPECT_TRUE(HasOverlap);

  hlsl::BindingInfo::BindingSpaces &SRVSpaces =
      Info.getBindingSpaces(ResourceClass::SRV);
  EXPECT_EQ(SRVSpaces.RC, ResourceClass::SRV);
  EXPECT_EQ(SRVSpaces.Spaces.size(), 1u);
  checkExpectedSpaceAndFreeRanges(SRVSpaces.Spaces[0], 0, {0u, 4u, 6u, ~0u});
}

TEST(HLSLBindingTest, TestEndOfRange) {
  hlsl::BindingInfoBuilder Builder;

  // RWBuffer<float> A     : register(u4294967295);  /* UINT32_MAX */
  // RWBuffer<float> B[10] : register(u4294967286, space1);
  //                         /* range (UINT32_MAX - 9, UINT32_MAX )*/
  // RWBuffer<float> C[10] : register(u2147483647, space2);
  //                         /* range (INT32_MAX, INT32_MAX + 9) */
  Builder.trackBinding(ResourceClass::UAV, /*Space=*/0, /*LowerBound=*/~0u,
                       /*UpperBound=*/~0u, /*Cookie=*/nullptr);
  Builder.trackBinding(ResourceClass::UAV, /*Space=*/1, /*LowerBound=*/~0u - 9u,
                       /*UpperBound=*/~0u, /*Cookie=*/nullptr);
  Builder.trackBinding(ResourceClass::UAV, /*Space=*/2,
                       /*LowerBound=*/2147483647u,
                       /*UpperBound=*/2147483647u + 9u, /*Cookie=*/nullptr);
  bool HasOverlap;
  hlsl::BindingInfo Info = Builder.calculateBindingInfo(HasOverlap);

  EXPECT_FALSE(HasOverlap);

  hlsl::BindingInfo::BindingSpaces &UAVSpaces =
      Info.getBindingSpaces(ResourceClass::UAV);
  EXPECT_EQ(UAVSpaces.RC, ResourceClass::UAV);
  EXPECT_EQ(UAVSpaces.Spaces.size(), 3u);
  checkExpectedSpaceAndFreeRanges(
      UAVSpaces.Spaces[0], 0, {0, std::numeric_limits<uint32_t>::max() - 1});
  checkExpectedSpaceAndFreeRanges(
      UAVSpaces.Spaces[1], 1, {0, std::numeric_limits<uint32_t>::max() - 10});
  checkExpectedSpaceAndFreeRanges(
      UAVSpaces.Spaces[2], 2,
      {0, static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) - 1u,
       static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) + 10u,
       std::numeric_limits<uint32_t>::max()});
}

TEST(HLSLBindingTest, TestFindAvailable) {
  hlsl::BindingInfoBuilder Builder;

  // RWBuffer<float> A : register(u5);
  // RWBuffer<float> B : register(u5, space1);
  // RWBuffer<float> C : register(u11, space1);
  // RWBuffer<float> D[] : register(u1, space2);
  Builder.trackBinding(ResourceClass::UAV, /*Space=*/0, /*LowerBound=*/5u,
                       /*UpperBound=*/5u, /*Cookie=*/nullptr);
  Builder.trackBinding(ResourceClass::UAV, /*Space=*/1, /*LowerBound=*/2u,
                       /*UpperBound=*/2u, /*Cookie=*/nullptr);
  Builder.trackBinding(ResourceClass::UAV, /*Space=*/1, /*LowerBound=*/6u,
                       /*UpperBound=*/6u, /*Cookie=*/nullptr);
  Builder.trackBinding(ResourceClass::UAV, /*Space=*/2, /*LowerBound=*/1u,
                       /*UpperBound=*/~0u, /*Cookie=*/nullptr);
  Builder.trackBinding(ResourceClass::UAV, /*Space=*/3, /*LowerBound=*/~0u - 1,
                       /*UpperBound=*/~0u - 1, /*Cookie=*/nullptr);
  bool HasOverlap;
  hlsl::BindingInfo Info = Builder.calculateBindingInfo(HasOverlap);

  EXPECT_FALSE(HasOverlap);

  // In space 0, we find room for a small binding at the beginning and
  // a large binding after `A`'s binding.
  std::optional<uint32_t> V =
      Info.findAvailableBinding(ResourceClass::UAV, /*Space=*/0, /*Size=*/1);
  EXPECT_THAT(V, HasSpecificValue(0u));
  V = Info.findAvailableBinding(ResourceClass::UAV, /*Space=*/0, /*Size=*/100);
  EXPECT_THAT(V, HasSpecificValue(6u));

  // In space 1, we try to fit larger bindings in the fill the gaps. Note that
  // we do this largest to smallest and observe that the gaps that are earlier
  // still exist.
  V = Info.findAvailableBinding(ResourceClass::UAV, /*Space=*/1, /*Size=*/4);
  EXPECT_THAT(V, HasSpecificValue(7u));
  V = Info.findAvailableBinding(ResourceClass::UAV, /*Space=*/1, /*Size=*/3);
  EXPECT_THAT(V, HasSpecificValue(3u));
  V = Info.findAvailableBinding(ResourceClass::UAV, /*Space=*/1, /*Size=*/2);
  EXPECT_THAT(V, HasSpecificValue(0u));
  // At this point, we've used all of the contiguous space up to 11u
  V = Info.findAvailableBinding(ResourceClass::UAV, /*Space=*/1, /*Size=*/1);
  EXPECT_THAT(V, HasSpecificValue(11u));

  // Space 2 is mostly full, we can only fit into the room at the beginning.
  V = Info.findAvailableBinding(ResourceClass::UAV, /*Space=*/2, /*Size=*/2);
  EXPECT_FALSE(V.has_value());
  V = Info.findAvailableBinding(ResourceClass::UAV, /*Space=*/2, /*Size=*/1);
  EXPECT_THAT(V, HasSpecificValue(0u));

  // Finding space for an unbounded array is a bit funnier. it always needs to
  // go a the end of the available space.
  V = Info.findAvailableBinding(ResourceClass::UAV, /*Space=*/3,
                                /*Size=*/~0u);
  // Note that we end up with a size 1 array here, starting at ~0u.
  EXPECT_THAT(V, HasSpecificValue(~0u));
  V = Info.findAvailableBinding(ResourceClass::UAV, /*Space=*/4,
                                /*Size=*/~0u);
  // In an empty space we find the slot at the beginning.
  EXPECT_THAT(V, HasSpecificValue(0u));
}

// Test that add duplicate bindings are correctly de-duplicated
TEST(HLSLBindingTest, TestNoOverlapWithDuplicates) {
  hlsl::BindingInfoBuilder Builder;

  // We add the same binding three times, and just use `nullptr` for the cookie
  // so that they should all be uniqued away.
  Builder.trackBinding(ResourceClass::SRV, /*Space=*/0, /*LowerBound=*/5,
                       /*UpperBound=*/5, /*Cookie=*/nullptr);
  Builder.trackBinding(ResourceClass::SRV, /*Space=*/0, /*LowerBound=*/5,
                       /*UpperBound=*/5, /*Cookie=*/nullptr);
  Builder.trackBinding(ResourceClass::SRV, /*Space=*/0, /*LowerBound=*/5,
                       /*UpperBound=*/5, /*Cookie=*/nullptr);
  bool HasOverlap;
  hlsl::BindingInfo Info = Builder.calculateBindingInfo(HasOverlap);

  EXPECT_FALSE(HasOverlap);
}
