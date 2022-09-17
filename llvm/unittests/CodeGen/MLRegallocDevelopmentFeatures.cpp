//===- MLRegAllocDevelopmentFeatures.cpp - test dev MLRegalloc features ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../lib/CodeGen/MLRegallocEvictAdvisor.h"
#include "llvm/Analysis/NoInferenceModelRunner.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CodeGen.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <vector>

using testing::ContainerEq;
using testing::Test;

struct LRPosInfoIndexes {
  size_t StartIndex;
  size_t EndIndex;
  size_t PhysReg;
};

class RegallocDevelopmentFeaturesTest : public ::Test {
protected:
  SmallVector<LRStartEndInfo>
  setupOverlapProblem(const SmallVectorImpl<LRPosInfoIndexes> &Segments,
                      ilist<IndexListEntry> &IndexList) {
    SmallVector<LRStartEndInfo> PositionsToReturn;
    PositionsToReturn.reserve(Segments.size());
    for (auto CurrentPosIndexInfo : Segments) {
      LRStartEndInfo CurrentPosInfo = {};
      CurrentPosInfo.Pos = CurrentPosIndexInfo.PhysReg;
      PositionsToReturn.push_back(CurrentPosInfo);
    }
    size_t CurrentSegmentIndex = 0;
    size_t CurrentIndex = 0;
    while (CurrentSegmentIndex < Segments.size()) {
      auto *CurrentLEMem = static_cast<IndexListEntry *>(
          Allocator.Allocate(sizeof(IndexListEntry), alignof(IndexListEntry)));
      auto *CurrentListEntry =
          new (CurrentLEMem) IndexListEntry(nullptr, CurrentIndex);
      IndexList.push_back(CurrentListEntry);
      for (size_t CurrentPosInfoIndex = 0;
           CurrentPosInfoIndex < Segments.size(); ++CurrentPosInfoIndex) {
        if ((CurrentIndex / SlotIndex::InstrDist) ==
            Segments[CurrentPosInfoIndex].StartIndex) {
          PositionsToReturn[CurrentPosInfoIndex].Begin =
              SlotIndex(CurrentListEntry, 0);
        } else if ((CurrentIndex / SlotIndex::InstrDist) ==
                   Segments[CurrentPosInfoIndex].EndIndex) {
          PositionsToReturn[CurrentPosInfoIndex].End =
              SlotIndex(CurrentListEntry, 0);
          ++CurrentSegmentIndex;
        }
      }
      CurrentIndex += SlotIndex::InstrDist;
    }
    return PositionsToReturn;
  }

  NoInferenceModelRunner setupModelRunner() {
    const std::vector<TensorSpec> Inputs{
        TensorSpec::createSpec<int64_t>("instructions", InstructionsShape),
        TensorSpec::createSpec<int64_t>("instructions_mapping",
                                        InstructionsMappingShape)};
    LLVMContext Ctx;
    return NoInferenceModelRunner(Ctx, Inputs);
  }

  std::vector<int64_t>
  getExpectedMappingMatrix(SmallVectorImpl<LRPosInfoIndexes> &OverlapSetup) {
    std::vector<int64_t> ExpectedMappingMatrix(
        NumberOfInterferences * ModelMaxSupportedInstructionCount, 0);
    for (auto NewSegment : OverlapSetup) {
      for (size_t CurrentIndex = NewSegment.StartIndex;
           CurrentIndex <= NewSegment.EndIndex; ++CurrentIndex) {
        ExpectedMappingMatrix[NewSegment.PhysReg *
                                  ModelMaxSupportedInstructionCount +
                              CurrentIndex] = 1;
      }
    }
    return ExpectedMappingMatrix;
  }

  void runOverlapTest(SmallVectorImpl<LRPosInfoIndexes> &OverlapSetup) {
    ilist<IndexListEntry> IndexList;
    auto OverlapProblem = setupOverlapProblem(OverlapSetup, IndexList);
    NoInferenceModelRunner ModelRunner = setupModelRunner();
    size_t MaxIndex = 0;
    for (size_t CurrentOverlap = 0; CurrentOverlap < OverlapSetup.size();
         ++CurrentOverlap) {
      if (OverlapSetup[CurrentOverlap].EndIndex >
          OverlapSetup[MaxIndex].EndIndex) {
        MaxIndex = CurrentOverlap;
      }
    }
    SlotIndex LastIndex = OverlapProblem[MaxIndex].End;
    extractInstructionFeatures(
        OverlapProblem, &ModelRunner,
        [](SlotIndex InputSlot) -> int { return 0; }, 0, 1, LastIndex);
    std::vector<int64_t> MappingMatrix(
        ModelRunner.getTensor<int64_t>(1),
        ModelRunner.getTensor<int64_t>(1) +
            NumberOfInterferences * ModelMaxSupportedInstructionCount);
    ASSERT_THAT(MappingMatrix,
                ContainerEq(getExpectedMappingMatrix(OverlapSetup)));
    IndexList.clearAndLeakNodesUnsafely();
  }

  BumpPtrAllocator Allocator;
};

// meta tests to ensure that test setup works correctly

TEST_F(RegallocDevelopmentFeaturesTest,
       MetaOverlapInstructionDistancesAreCorrect) {
  SmallVector<LRPosInfoIndexes, 2> OverlapSetup;
  OverlapSetup.push_back({0, 5, 0});
  OverlapSetup.push_back({5, 10, 0});
  ilist<IndexListEntry> IndexList;
  auto OverlapProblem = setupOverlapProblem(OverlapSetup, IndexList);
  ASSERT_EQ(OverlapProblem[0].End.distance(OverlapProblem[1].End),
            5 * SlotIndex::InstrDist);
  ASSERT_EQ(OverlapProblem[0].End.distance(OverlapProblem[1].Begin), 0);
}

TEST_F(RegallocDevelopmentFeaturesTest, MetaSlotIndicesAreValid) {
  SmallVector<LRPosInfoIndexes, 1> OverlapSetup;
  OverlapSetup.push_back({0, 10, 0});
  ilist<IndexListEntry> IndexList;
  auto OverlapProblem = setupOverlapProblem(OverlapSetup, IndexList);
  ASSERT_TRUE(OverlapProblem[0].Begin.isValid());
  ASSERT_TRUE(OverlapProblem[0].End.isValid());
}

// Testing of feature extraction for per-instruction features

TEST_F(RegallocDevelopmentFeaturesTest, InstructionOpcodesAreCorrect) {
  SmallVector<LRPosInfoIndexes, 1> OverlapSetup;
  OverlapSetup.push_back({0, ModelMaxSupportedInstructionCount - 1, 0});
  ilist<IndexListEntry> IndexList;
  auto OverlapProblem = setupOverlapProblem(OverlapSetup, IndexList);
  NoInferenceModelRunner ModelRunner = setupModelRunner();
  SlotIndex LastIndex = OverlapProblem[0].End;
  SlotIndex FirstIndex = OverlapProblem[0].Begin;
  extractInstructionFeatures(
      OverlapProblem, &ModelRunner,
      [FirstIndex](SlotIndex InputSlot) -> int {
        return FirstIndex.distance(InputSlot) / SlotIndex::InstrDist;
      },
      0, 1, LastIndex);
  for (size_t CurrentInstructionIndex = 0;
       CurrentInstructionIndex < ModelMaxSupportedInstructionCount;
       ++CurrentInstructionIndex) {
    ASSERT_EQ(
        (size_t)ModelRunner.getTensor<int64_t>(0)[CurrentInstructionIndex],
        CurrentInstructionIndex);
  }
}

TEST_F(RegallocDevelopmentFeaturesTest, FullOverlap) {
  SmallVector<LRPosInfoIndexes, 2> OverlapSetup;
  OverlapSetup.push_back({0, ModelMaxSupportedInstructionCount - 1, 0});
  OverlapSetup.push_back({0, ModelMaxSupportedInstructionCount - 1, 1});
  runOverlapTest(OverlapSetup);
}

TEST_F(RegallocDevelopmentFeaturesTest, PartialOverlap) {
  SmallVector<LRPosInfoIndexes, 2> OverlapSetup;
  OverlapSetup.push_back({0, 20, 0});
  OverlapSetup.push_back({15, 30, 1});
  runOverlapTest(OverlapSetup);
}

TEST_F(RegallocDevelopmentFeaturesTest, PartialOverlapOpposite) {
  SmallVector<LRPosInfoIndexes, 2> OverlapSetup;
  OverlapSetup.push_back({15, 30, 1});
  OverlapSetup.push_back({0, 20, 0});
  runOverlapTest(OverlapSetup);
}

TEST_F(RegallocDevelopmentFeaturesTest, InternalOverlap) {
  SmallVector<LRPosInfoIndexes, 2> OverlapSetup;
  OverlapSetup.push_back({0, 30, 0});
  OverlapSetup.push_back({10, 20, 1});
  runOverlapTest(OverlapSetup);
}

TEST_F(RegallocDevelopmentFeaturesTest, TripleInternalOverlap) {
  SmallVector<LRPosInfoIndexes, 3> OverlapSetup;
  OverlapSetup.push_back({0, 30, 0});
  OverlapSetup.push_back({10, 25, 1});
  OverlapSetup.push_back({15, 20, 2});
  runOverlapTest(OverlapSetup);
}

TEST_F(RegallocDevelopmentFeaturesTest, InternalMultiOverlap) {
  SmallVector<LRPosInfoIndexes, 3> OverlapSetup;
  OverlapSetup.push_back({0, 45, 0});
  OverlapSetup.push_back({30, 40, 1});
  OverlapSetup.push_back({35, 60, 2});
  runOverlapTest(OverlapSetup);
}
