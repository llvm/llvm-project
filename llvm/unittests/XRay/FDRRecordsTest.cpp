//===- FDRRecords.cpp - Unit Tests for XRay FDR Record Loading ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "llvm/XRay/BlockIndexer.h"
#include "llvm/XRay/BlockPrinter.h"
#include "llvm/XRay/BlockVerifier.h"
#include "llvm/XRay/FDRLogBuilder.h"
#include "llvm/XRay/FDRRecords.h"
#include "llvm/XRay/RecordPrinter.h"

namespace llvm {
namespace xray {
namespace {

using ::testing::Eq;
using ::testing::Not;

TEST(XRayFDRTest, BuilderAndBlockIndexer) {
  // We recreate a single block of valid records, then ensure that we find all
  // of them belonging in the same index. We do this for three blocks, and
  // ensure we find the same records in the blocks we deduce.
  auto Block0 = LogBuilder()
                    .add<BufferExtents>(100)
                    .add<NewBufferRecord>(1)
                    .add<WallclockRecord>(1, 1)
                    .add<PIDRecord>(1)
                    .add<FunctionRecord>(RecordTypes::ENTER, 1, 1)
                    .add<FunctionRecord>(RecordTypes::EXIT, 1, 100)
                    .add<CustomEventRecordV5>(1, 4, "XRAY")
                    .add<TypedEventRecord>(1, 4, 2, "XRAY")
                    .consume();
  auto Block1 = LogBuilder()
                    .add<BufferExtents>(100)
                    .add<NewBufferRecord>(1)
                    .add<WallclockRecord>(1, 2)
                    .add<PIDRecord>(1)
                    .add<FunctionRecord>(RecordTypes::ENTER, 1, 1)
                    .add<FunctionRecord>(RecordTypes::EXIT, 1, 100)
                    .add<CustomEventRecordV5>(1, 4, "XRAY")
                    .add<TypedEventRecord>(1, 4, 2, "XRAY")
                    .consume();
  auto Block2 = LogBuilder()
                    .add<BufferExtents>(100)
                    .add<NewBufferRecord>(2)
                    .add<WallclockRecord>(1, 3)
                    .add<PIDRecord>(1)
                    .add<FunctionRecord>(RecordTypes::ENTER, 1, 1)
                    .add<FunctionRecord>(RecordTypes::EXIT, 1, 100)
                    .add<CustomEventRecordV5>(1, 4, "XRAY")
                    .add<TypedEventRecord>(1, 4, 2, "XRAY")
                    .consume();
  BlockIndexer::Index Index;
  BlockIndexer Indexer(Index);
  for (auto B : {std::ref(Block0), std::ref(Block1), std::ref(Block2)}) {
    for (auto &R : B.get())
      ASSERT_FALSE(errorToBool(R->apply(Indexer)));
    ASSERT_FALSE(errorToBool(Indexer.flush()));
  }

  // We have two threads worth of blocks.
  ASSERT_THAT(Index.size(), Eq(2u));
  auto T1Blocks = Index.find({1, 1});
  ASSERT_THAT(T1Blocks, Not(Eq(Index.end())));
  ASSERT_THAT(T1Blocks->second.size(), Eq(2u));
  auto T2Blocks = Index.find({1, 2});
  ASSERT_THAT(T2Blocks, Not(Eq(Index.end())));
  ASSERT_THAT(T2Blocks->second.size(), Eq(1u));
}

TEST(XRayFDRTest, BuilderAndBlockVerifier) {
  auto Block = LogBuilder()
                   .add<BufferExtents>(48)
                   .add<NewBufferRecord>(1)
                   .add<WallclockRecord>(1, 1)
                   .add<PIDRecord>(1)
                   .add<NewCPUIDRecord>(1, 2)
                   .consume();
  BlockVerifier Verifier;
  for (auto &R : Block)
    ASSERT_FALSE(errorToBool(R->apply(Verifier)));
  ASSERT_FALSE(errorToBool(Verifier.verify()));
}

TEST(XRayFDRTest, IndexAndVerifyBlocks) {
  auto Block0 = LogBuilder()
                    .add<BufferExtents>(64)
                    .add<NewBufferRecord>(1)
                    .add<WallclockRecord>(1, 1)
                    .add<PIDRecord>(1)
                    .add<NewCPUIDRecord>(1, 2)
                    .add<FunctionRecord>(RecordTypes::ENTER, 1, 1)
                    .add<FunctionRecord>(RecordTypes::EXIT, 1, 100)
                    .add<CustomEventRecordV5>(1, 4, "XRAY")
                    .add<TypedEventRecord>(1, 4, 2, "XRAY")
                    .consume();
  auto Block1 = LogBuilder()
                    .add<BufferExtents>(64)
                    .add<NewBufferRecord>(1)
                    .add<WallclockRecord>(1, 1)
                    .add<PIDRecord>(1)
                    .add<NewCPUIDRecord>(1, 2)
                    .add<FunctionRecord>(RecordTypes::ENTER, 1, 1)
                    .add<FunctionRecord>(RecordTypes::EXIT, 1, 100)
                    .add<CustomEventRecordV5>(1, 4, "XRAY")
                    .add<TypedEventRecord>(1, 4, 2, "XRAY")
                    .consume();
  auto Block2 = LogBuilder()
                    .add<BufferExtents>(64)
                    .add<NewBufferRecord>(1)
                    .add<WallclockRecord>(1, 1)
                    .add<PIDRecord>(1)
                    .add<NewCPUIDRecord>(1, 2)
                    .add<FunctionRecord>(RecordTypes::ENTER, 1, 1)
                    .add<FunctionRecord>(RecordTypes::EXIT, 1, 100)
                    .add<CustomEventRecordV5>(1, 4, "XRAY")
                    .add<TypedEventRecord>(1, 4, 2, "XRAY")
                    .consume();

  // First, index the records in different blocks.
  BlockIndexer::Index Index;
  BlockIndexer Indexer(Index);
  for (auto B : {std::ref(Block0), std::ref(Block1), std::ref(Block2)}) {
    for (auto &R : B.get())
      ASSERT_FALSE(errorToBool(R->apply(Indexer)));
    ASSERT_FALSE(errorToBool(Indexer.flush()));
  }

  // Next, verify that each block is consistently defined.
  BlockVerifier Verifier;
  for (auto &ProcessThreadBlocks : Index) {
    auto &Blocks = ProcessThreadBlocks.second;
    for (auto &B : Blocks) {
      for (auto *R : B.Records)
        ASSERT_FALSE(errorToBool(R->apply(Verifier)));
      ASSERT_FALSE(errorToBool(Verifier.verify()));
      Verifier.reset();
    }
  }

  // Then set up the printing mechanisms.
  std::string Output;
  raw_string_ostream OS(Output);
  RecordPrinter RP(OS);
  BlockPrinter BP(OS, RP);
  for (auto &ProcessThreadBlocks : Index) {
    auto &Blocks = ProcessThreadBlocks.second;
    for (auto &B : Blocks) {
      for (auto *R : B.Records)
        ASSERT_FALSE(errorToBool(R->apply(BP)));
      BP.reset();
    }
  }

  OS.flush();
  EXPECT_THAT(Output, Not(Eq("")));
}

} // namespace
} // namespace xray
} // namespace llvm
