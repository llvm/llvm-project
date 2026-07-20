//===- SampleContextTrackerTest.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/SampleContextTracker.h"
#include "llvm/ProfileData/SampleProf.h"
#include "gtest/gtest.h"
#include <list>

using namespace llvm;
using namespace sampleprof;

namespace {

// SampleContextFrames are non-owning ArrayRefs into a name table, so the table
// backing every parsed context string must outlive the profile map and tracker.
static FunctionSamples &
addProfile(SampleProfileMap &Profiles,
           std::list<SampleContextFrameVector> &CSNameTable, StringRef CtxStr,
           uint64_t Samples) {
  FunctionSamples &FS = Profiles.create(SampleContext(CtxStr, CSNameTable));
  FS.addTotalSamples(Samples);
  FS.addBodySamples(1, 0, Samples);
  return FS;
}

// Regression test for a heap-use-after-free in
// SampleContextTracker::getBaseSamplesFor().
//
// When a function appears in nested (e.g. recursive) contexts, promoting the
// outer context relocates the inner same-function node into the base subtree
// (marking its FunctionSamples SyntheticContext) and frees the original subtree
// via clear(). getBaseSamplesFor() iterates a pre-collected list of all context
// profiles for the function; without skipping the already-relocated
// SyntheticContext profiles it re-promotes a freed node and reads freed memory.
//
// This builds exactly that shape (foo nested under foo) and checks the base
// profile is produced without crashing. Under an ASAN build this test fails on
// the unfixed code with a heap-use-after-free.
TEST(SampleContextTrackerTest, GetBaseSamplesForNestedRecursiveContext) {
  std::list<SampleContextFrameVector> CSNameTable;
  SampleProfileMap Profiles;

  addProfile(Profiles, CSNameTable, "[main:1 @ foo]", 100);
  addProfile(Profiles, CSNameTable, "[main:1 @ foo:2 @ foo]", 50);
  addProfile(Profiles, CSNameTable, "[main:1 @ foo:2 @ foo:2 @ foo]", 25);

  SampleContextTracker Tracker(Profiles, /*GUIDToFuncNameMap=*/nullptr);

  // Must not crash (heap-use-after-free on the unfixed code) and must return a
  // non-null merged base profile carrying samples.
  FunctionSamples *Base =
      Tracker.getBaseSamplesFor(FunctionId("foo"), /*MergeContext=*/true);
  ASSERT_NE(Base, nullptr);
  EXPECT_GE(Base->getTotalSamples(), 100u);
}

// A single (non-nested) context should also merge cleanly and is a sanity check
// that the SyntheticContext skip does not drop a legitimate first-time
// promotion.
TEST(SampleContextTrackerTest, GetBaseSamplesForSingleContext) {
  std::list<SampleContextFrameVector> CSNameTable;
  SampleProfileMap Profiles;

  addProfile(Profiles, CSNameTable, "[main:1 @ bar]", 42);

  SampleContextTracker Tracker(Profiles, /*GUIDToFuncNameMap=*/nullptr);
  FunctionSamples *Base =
      Tracker.getBaseSamplesFor(FunctionId("bar"), /*MergeContext=*/true);
  ASSERT_NE(Base, nullptr);
  EXPECT_EQ(Base->getTotalSamples(), 42u);
}

} // namespace
