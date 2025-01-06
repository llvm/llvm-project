//===- MLRegAllocDevelopmentFeatures.cpp - test dev MLRegAlloc features ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../lib/CodeGen/MLRegAllocEvictAdvisor.h"
#include "llvm/Analysis/NoInferenceModelRunner.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <string>
#include <vector>

using namespace llvm;
using testing::ContainerEq;
using testing::Test;

namespace {

#include "MFCommon.inc"

struct LRPosInfoIndexes {
  size_t StartIndex;
  size_t EndIndex;
  size_t PhysReg;
};

class RegAllocDevelopmentFeaturesTest : public ::Test {
protected:
  SmallVector<LRStartEndInfo>
  setupOverlapProblem(const SmallVectorImpl<LRPosInfoIndexes> &Segments,
                      simple_ilist<IndexListEntry> &IndexList) {
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
      IndexList.push_back(*CurrentListEntry);
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
                                        InstructionsMappingShape),
        TensorSpec::createSpec<float>("mbb_frequencies", MBBFrequencyShape),
        TensorSpec::createSpec<int64_t>("mbb_mapping", InstructionsShape)};
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
    simple_ilist<IndexListEntry> IndexList;
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
        [](SlotIndex InputSlot) -> int { return 0; },
        [](SlotIndex InputSlot) -> float { return 0.0f; },
        [](SlotIndex InputSlot) -> MachineBasicBlock * { return nullptr; }, 0,
        1, 2, 3, LastIndex);
    std::vector<int64_t> MappingMatrix(
        ModelRunner.getTensor<int64_t>(1),
        ModelRunner.getTensor<int64_t>(1) +
            NumberOfInterferences * ModelMaxSupportedInstructionCount);
    ASSERT_THAT(MappingMatrix,
                ContainerEq(getExpectedMappingMatrix(OverlapSetup)));
    IndexList.clear();
  }

  BumpPtrAllocator Allocator;
};

// meta tests to ensure that test setup works correctly

TEST_F(RegAllocDevelopmentFeaturesTest,
       MetaOverlapInstructionDistancesAreCorrect) {
  SmallVector<LRPosInfoIndexes, 2> OverlapSetup;
  OverlapSetup.push_back({0, 5, 0});
  OverlapSetup.push_back({5, 10, 0});
  simple_ilist<IndexListEntry> IndexList;
  auto OverlapProblem = setupOverlapProblem(OverlapSetup, IndexList);
  ASSERT_EQ(OverlapProblem[0].End.distance(OverlapProblem[1].End),
            5 * SlotIndex::InstrDist);
  ASSERT_EQ(OverlapProblem[0].End.distance(OverlapProblem[1].Begin), 0);
}

TEST_F(RegAllocDevelopmentFeaturesTest, MetaSlotIndicesAreValid) {
  SmallVector<LRPosInfoIndexes, 1> OverlapSetup;
  OverlapSetup.push_back({0, 10, 0});
  simple_ilist<IndexListEntry> IndexList;
  auto OverlapProblem = setupOverlapProblem(OverlapSetup, IndexList);
  ASSERT_TRUE(OverlapProblem[0].Begin.isValid());
  ASSERT_TRUE(OverlapProblem[0].End.isValid());
}

// Testing of feature extraction for per-instruction features

TEST_F(RegAllocDevelopmentFeaturesTest, InstructionOpcodesAreCorrect) {
  SmallVector<LRPosInfoIndexes, 1> OverlapSetup;
  OverlapSetup.push_back({0, ModelMaxSupportedInstructionCount - 1, 0});
  simple_ilist<IndexListEntry> IndexList;
  auto OverlapProblem = setupOverlapProblem(OverlapSetup, IndexList);
  NoInferenceModelRunner ModelRunner = setupModelRunner();
  SlotIndex LastIndex = OverlapProblem[0].End;
  SlotIndex FirstIndex = OverlapProblem[0].Begin;
  extractInstructionFeatures(
      OverlapProblem, &ModelRunner,
      [FirstIndex](SlotIndex InputSlot) -> int {
        return FirstIndex.distance(InputSlot) / SlotIndex::InstrDist;
      },
      [](SlotIndex InputSlot) -> float { return 0.0f; },
      [](SlotIndex InputSlot) -> MachineBasicBlock * { return nullptr; }, 0, 1,
      2, 3, LastIndex);
  for (size_t CurrentInstructionIndex = 0;
       CurrentInstructionIndex < ModelMaxSupportedInstructionCount;
       ++CurrentInstructionIndex) {
    ASSERT_EQ(
        (size_t)ModelRunner.getTensor<int64_t>(0)[CurrentInstructionIndex],
        CurrentInstructionIndex);
  }
}

TEST_F(RegAllocDevelopmentFeaturesTest, FullOverlap) {
  SmallVector<LRPosInfoIndexes, 2> OverlapSetup;
  OverlapSetup.push_back({0, ModelMaxSupportedInstructionCount - 1, 0});
  OverlapSetup.push_back({0, ModelMaxSupportedInstructionCount - 1, 1});
  runOverlapTest(OverlapSetup);
}

TEST_F(RegAllocDevelopmentFeaturesTest, PartialOverlap) {
  SmallVector<LRPosInfoIndexes, 2> OverlapSetup;
  OverlapSetup.push_back({0, 20, 0});
  OverlapSetup.push_back({15, 30, 1});
  runOverlapTest(OverlapSetup);
}

TEST_F(RegAllocDevelopmentFeaturesTest, PartialOverlapOpposite) {
  SmallVector<LRPosInfoIndexes, 2> OverlapSetup;
  OverlapSetup.push_back({15, 30, 1});
  OverlapSetup.push_back({0, 20, 0});
  runOverlapTest(OverlapSetup);
}

TEST_F(RegAllocDevelopmentFeaturesTest, InternalOverlap) {
  SmallVector<LRPosInfoIndexes, 2> OverlapSetup;
  OverlapSetup.push_back({0, 30, 0});
  OverlapSetup.push_back({10, 20, 1});
  runOverlapTest(OverlapSetup);
}

TEST_F(RegAllocDevelopmentFeaturesTest, TripleInternalOverlap) {
  SmallVector<LRPosInfoIndexes, 3> OverlapSetup;
  OverlapSetup.push_back({0, 30, 0});
  OverlapSetup.push_back({10, 25, 1});
  OverlapSetup.push_back({15, 20, 2});
  runOverlapTest(OverlapSetup);
}

TEST_F(RegAllocDevelopmentFeaturesTest, InternalMultiOverlap) {
  SmallVector<LRPosInfoIndexes, 3> OverlapSetup;
  OverlapSetup.push_back({0, 45, 0});
  OverlapSetup.push_back({30, 40, 1});
  OverlapSetup.push_back({35, 60, 2});
  runOverlapTest(OverlapSetup);
}

TEST_F(RegAllocDevelopmentFeaturesTest, SingleMBBTest) {
  NoInferenceModelRunner ModelRunner = setupModelRunner();
  SlotIndex CurrentIndex;
  // set index to 1 so we can ensure that the mapping actually get set
  std::map<MachineBasicBlock *, size_t> VisitedMBBs = {{nullptr, 1}};
  extractMBBFrequency(
      CurrentIndex, 0, VisitedMBBs,
      [](SlotIndex InputSlot) -> float { return 1.0f; }, nullptr, &ModelRunner,
      2, 3);
  ASSERT_FLOAT_EQ(ModelRunner.getTensor<float>(2)[1], 1.0f);
  ASSERT_EQ(ModelRunner.getTensor<int64_t>(3)[0], 1);
}

TEST_F(RegAllocDevelopmentFeaturesTest, MBBFullTruncated) {
  SmallVector<LRPosInfoIndexes, 1> OverlapSetup;
  OverlapSetup.push_back({0, ModelMaxSupportedInstructionCount - 1, 0});
  simple_ilist<IndexListEntry> IndexList;
  auto OverlapProblem = setupOverlapProblem(OverlapSetup, IndexList);
  NoInferenceModelRunner ModelRunner = setupModelRunner();
  SlotIndex LastIndex = OverlapProblem[0].End;
  SlotIndex FirstIndex = OverlapProblem[0].Begin;

  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);
  std::array<MachineBasicBlock *, ModelMaxSupportedInstructionCount>
      MBBsForTest;
  for (size_t I = 0; I < ModelMaxSupportedInstructionCount; ++I) {
    MBBsForTest[I] = MF->CreateMachineBasicBlock();
  }

  extractInstructionFeatures(
      OverlapProblem, &ModelRunner,
      [](SlotIndex InputSlot) -> int { return 0; },
      [FirstIndex](SlotIndex InputSlot) -> float {
        return static_cast<float>(FirstIndex.distance(InputSlot) /
                                  SlotIndex::InstrDist);
      },
      [FirstIndex, MBBsForTest](SlotIndex InputSlot) -> MachineBasicBlock * {
        return MBBsForTest[FirstIndex.distance(InputSlot) /
                           SlotIndex::InstrDist];
      },
      0, 1, 2, 3, LastIndex);
  for (size_t MBBIndex = 0; MBBIndex < ModelMaxSupportedMBBCount; ++MBBIndex) {
    ASSERT_FLOAT_EQ(ModelRunner.getTensor<float>(2)[MBBIndex],
                    static_cast<float>(MBBIndex));
    ASSERT_EQ(ModelRunner.getTensor<int64_t>(3)[MBBIndex],
              static_cast<int64_t>(MBBIndex));
  }
  // the rest of the mapping values should be zero (truncated to 100 MBBs)
  for (size_t MBBIndex = ModelMaxSupportedMBBCount;
       MBBIndex < ModelMaxSupportedInstructionCount; ++MBBIndex) {
    ASSERT_EQ(ModelRunner.getTensor<int64_t>(3)[MBBIndex],
              static_cast<int64_t>(0));
  }
}

} // end namespace
