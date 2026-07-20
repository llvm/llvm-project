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
// and later merges/frees it via clear()/removeChildContext().
// getBaseSamplesFor() iterates a pre-collected list of all context profiles for
// the function; the profile of a freed node otherwise still maps to that node
// in ProfileToNodeMap, so re-promoting it reads freed memory. The fix drops
// such stale map entries when a node is destroyed, so
// getContextNodeForProfile() returns null and the entry is skipped.
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
// that fixing the use-after-free does not drop a legitimate first-time
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

// A distinct callee nested under another function must still get its own base
// profile. Promoting foo relocates bar's node into foo's base subtree (marking
// bar SyntheticContext); the relocated node stays live, so getBaseSamplesFor
// for bar must still promote it. A too-broad "skip all SyntheticContext"
// use-after-free workaround would drop bar's samples here.
TEST(SampleContextTrackerTest, GetBaseSamplesForDistinctNestedCallee) {
  std::list<SampleContextFrameVector> CSNameTable;
  SampleProfileMap Profiles;

  addProfile(Profiles, CSNameTable, "[main:1 @ foo]", 100);
  addProfile(Profiles, CSNameTable, "[main:1 @ foo:3 @ bar]", 42);

  SampleContextTracker Tracker(Profiles, /*GUIDToFuncNameMap=*/nullptr);

  // Promote foo first; this relocates bar's node under foo's base.
  FunctionSamples *FooBase =
      Tracker.getBaseSamplesFor(FunctionId("foo"), /*MergeContext=*/true);
  ASSERT_NE(FooBase, nullptr);

  // bar must still be promotable to its own base with its samples intact.
  FunctionSamples *BarBase =
      Tracker.getBaseSamplesFor(FunctionId("bar"), /*MergeContext=*/true);
  ASSERT_NE(BarBase, nullptr);
  EXPECT_EQ(BarBase->getTotalSamples(), 42u);
}

} // namespace
