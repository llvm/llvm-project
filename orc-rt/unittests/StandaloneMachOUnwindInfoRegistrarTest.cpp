//===- StandaloneMachOUnwindInfoRegistrarTest.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the storage layer underlying StandaloneMachOUnwindInfoRegistrar
// (its private UnwindInfoMap inner class). Exercises the
// register/deregister/lookup API without any libunwind interaction;
// libunwind-facing behavior is left to regression tests.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/StandaloneMachOUnwindInfoRegistrar.h"

#include "gtest/gtest.h"

using namespace orc_rt;

namespace {

// Helper: build a half-open code-address range from raw integer values.
ExecutorAddrRange range(uint64_t Start, uint64_t End) {
  return ExecutorAddrRange(ExecutorAddr(Start), ExecutorAddr(End));
}

} // namespace

namespace orc_rt {
// Fixture: befriended by StandaloneMachOUnwindInfoRegistrar so tests can name
// the otherwise-private UnwindInfoMap inner class and DynamicUnwindSections
// struct. Lives in orc_rt to match the unqualified friend declaration in the
// registrar header.
class UnwindInfoMapTest : public ::testing::Test {
protected:
  using UnwindInfoMap = StandaloneMachOUnwindInfoRegistrar::UnwindInfoMap;
  using DynamicUnwindSections =
      StandaloneMachOUnwindInfoRegistrar::DynamicUnwindSections;

  // An arbitrary, recognisable DynamicUnwindSections for use in tests that
  // don't care about the exact field values.
  static DynamicUnwindSections sampleInfo() {
    return {0x1000, 0x2000, 64, 0x3000, 32};
  }
};
} // namespace orc_rt

TEST_F(UnwindInfoMapTest, RegisterAndDeregisterSucceeds) {
  UnwindInfoMap Map;
  cantFail(Map.registerRanges({range(0x100, 0x200)}, sampleInfo()));
  cantFail(Map.deregisterRanges({range(0x100, 0x200)}));
}

TEST_F(UnwindInfoMapTest, DeregisterUnregisteredFails) {
  UnwindInfoMap Map;
  auto E = Map.deregisterRanges({range(0x100, 0x200)});
  ASSERT_TRUE(static_cast<bool>(E));
  EXPECT_EQ(toString(std::move(E)),
            "No unwind-info sections registered for range");
}

TEST_F(UnwindInfoMapTest, OverlappingRegistrationRejected) {
  // [0x100, 0x300) then [0x200, 0x400): second starts inside the first.
  UnwindInfoMap Map;
  cantFail(Map.registerRanges({range(0x100, 0x300)}, sampleInfo()));

  auto E = Map.registerRanges({range(0x200, 0x400)}, sampleInfo());
  ASSERT_TRUE(static_cast<bool>(E));
  EXPECT_EQ(toString(std::move(E)),
            "Code-range for unwind-info registration overlaps an existing "
            "range");
}

TEST_F(UnwindInfoMapTest, ContainedRegistrationRejected) {
  // [0x100, 0x400) then [0x200, 0x300): second sits entirely inside the first.
  UnwindInfoMap Map;
  cantFail(Map.registerRanges({range(0x100, 0x400)}, sampleInfo()));

  auto E = Map.registerRanges({range(0x200, 0x300)}, sampleInfo());
  ASSERT_TRUE(static_cast<bool>(E));
  EXPECT_EQ(toString(std::move(E)),
            "Code-range for unwind-info registration overlaps an existing "
            "range");
}

TEST_F(UnwindInfoMapTest, ExactDuplicateRegistrationRejected) {
  UnwindInfoMap Map;
  cantFail(Map.registerRanges({range(0x100, 0x200)}, sampleInfo()));

  auto E = Map.registerRanges({range(0x100, 0x200)}, sampleInfo());
  ASSERT_TRUE(static_cast<bool>(E));
  EXPECT_EQ(toString(std::move(E)),
            "Code-range for unwind-info registration overlaps an existing "
            "range");
}

TEST_F(UnwindInfoMapTest, EmptyRangeIgnored) {
  // Registering an empty range should succeed but produce no entry, so a
  // subsequent deregister of the same range fails.
  UnwindInfoMap Map;
  cantFail(Map.registerRanges({range(0x100, 0x100)}, sampleInfo()));

  auto E = Map.deregisterRanges({range(0x100, 0x100)});
  ASSERT_TRUE(static_cast<bool>(E));
  EXPECT_EQ(toString(std::move(E)),
            "No unwind-info sections registered for range");
}

TEST_F(UnwindInfoMapTest, AdjacentRangesAccepted) {
  // [0x100, 0x200) and [0x200, 0x300) touch at the boundary but don't
  // overlap.
  UnwindInfoMap Map;
  cantFail(Map.registerRanges({range(0x100, 0x200)}, sampleInfo()));
  cantFail(Map.registerRanges({range(0x200, 0x300)}, sampleInfo()));
}

TEST_F(UnwindInfoMapTest, PartialFailureLeavesEarlierRangesRegistered) {
  // Multi-range registerRanges call where the second range overlaps an
  // already-registered range. The first range in the failing call should
  // remain registered, and the pre-existing range is untouched.
  UnwindInfoMap Map;
  cantFail(Map.registerRanges({range(0x100, 0x200)}, sampleInfo()));

  auto E = Map.registerRanges({range(0x300, 0x400), range(0x150, 0x250)},
                              sampleInfo());
  ASSERT_TRUE(static_cast<bool>(E));
  consumeError(std::move(E));

  cantFail(Map.deregisterRanges({range(0x300, 0x400)}));
  cantFail(Map.deregisterRanges({range(0x100, 0x200)}));
}

TEST_F(UnwindInfoMapTest, LookupInsideRegisteredRange) {
  UnwindInfoMap Map;
  DynamicUnwindSections Info{0x1000, 0x2000, 64, 0x3000, 32};
  cantFail(Map.registerRanges({range(0x100, 0x200)}, Info));

  // Lookups at Start, midway, and just-below-End should all hit.
  for (uintptr_t Addr :
       {uintptr_t{0x100}, uintptr_t{0x180}, uintptr_t{0x1FF}}) {
    auto R = Map.lookup(Addr);
    ASSERT_TRUE(R.has_value()) << "Expected lookup to hit at " << Addr;
    EXPECT_EQ(R->DSOBase, 0x1000u);
    EXPECT_EQ(R->DWARFSection, 0x2000u);
    EXPECT_EQ(R->DWARFSectionLength, 64u);
    EXPECT_EQ(R->CompactUnwindSection, 0x3000u);
    EXPECT_EQ(R->CompactUnwindSectionLength, 32u);
  }
}

TEST_F(UnwindInfoMapTest, LookupOutsideRegisteredRangeReturnsNullopt) {
  UnwindInfoMap Map;
  cantFail(Map.registerRanges({range(0x100, 0x200)}, sampleInfo()));

  // Below any registered range.
  EXPECT_FALSE(Map.lookup(0x0).has_value());
  EXPECT_FALSE(Map.lookup(0xFF).has_value());

  // At End (half-open: End is not part of the range).
  EXPECT_FALSE(Map.lookup(0x200).has_value());

  // Above any registered range.
  EXPECT_FALSE(Map.lookup(0x1000).has_value());
}

TEST_F(UnwindInfoMapTest, LookupOnEmptyMapReturnsNullopt) {
  UnwindInfoMap Map;
  EXPECT_FALSE(Map.lookup(0).has_value());
  EXPECT_FALSE(Map.lookup(0x1000).has_value());
}

TEST_F(UnwindInfoMapTest, LookupBetweenRegisteredRangesReturnsNullopt) {
  // Two non-adjacent ranges; lookup in the gap must miss rather than return
  // the lower range (regression guard for the upper_bound - 1 logic).
  UnwindInfoMap Map;
  cantFail(Map.registerRanges({range(0x100, 0x200)}, sampleInfo()));
  cantFail(Map.registerRanges({range(0x300, 0x400)}, sampleInfo()));

  EXPECT_FALSE(Map.lookup(0x200).has_value());
  EXPECT_FALSE(Map.lookup(0x250).has_value());
  EXPECT_FALSE(Map.lookup(0x2FF).has_value());

  // And the second range is reachable.
  ASSERT_TRUE(Map.lookup(0x300).has_value());
  ASSERT_TRUE(Map.lookup(0x3FF).has_value());
}
