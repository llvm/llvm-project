//===- llvm/unittest/XRay/FDRTraceWriterTest.cpp ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "llvm/XRay/BlockIndexer.h"
#include "llvm/XRay/FDRLogBuilder.h"
#include "llvm/XRay/FDRRecords.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace xray {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;
using ::testing::Not;
using ::testing::SizeIs;

// This test ensures that we can index blocks that follow version 3 of the log
// format.
TEST(FDRBlockIndexerTest, IndexBlocksV3) {
  auto Block0 = LogBuilder()
                    .add<BufferExtents>(80)
                    .add<NewBufferRecord>(1)
                    .add<WallclockRecord>(1, 2)
                    .add<PIDRecord>(1)
                    .add<NewCPUIDRecord>(1, 2)
                    .add<FunctionRecord>(RecordTypes::ENTER, 1, 1)
                    .add<FunctionRecord>(RecordTypes::EXIT, 1, 100)
                    .consume();
  auto Block1 = LogBuilder()
                    .add<BufferExtents>(80)
                    .add<NewBufferRecord>(1)
                    .add<WallclockRecord>(1, 2)
                    .add<PIDRecord>(1)
                    .add<NewCPUIDRecord>(1, 2)
                    .add<FunctionRecord>(RecordTypes::ENTER, 1, 1)
                    .add<FunctionRecord>(RecordTypes::EXIT, 1, 100)
                    .consume();
  auto Block2 = LogBuilder()
                    .add<BufferExtents>(80)
                    .add<NewBufferRecord>(2)
                    .add<WallclockRecord>(1, 2)
                    .add<PIDRecord>(1)
                    .add<NewCPUIDRecord>(2, 2)
                    .add<FunctionRecord>(RecordTypes::ENTER, 1, 1)
                    .add<FunctionRecord>(RecordTypes::EXIT, 1, 100)
                    .consume();
  BlockIndexer::Index Index;
  BlockIndexer Indexer(Index);
  // Iterate through the contrived blocks we have created above.
  for (auto B : {std::ref(Block0), std::ref(Block1), std::ref(Block2)}) {
    // For each record in the block, we apply the indexer.
    for (auto &R : B.get())
      ASSERT_FALSE(errorToBool(R->apply(Indexer)));
    ASSERT_FALSE(errorToBool(Indexer.flush()));
  }

  ASSERT_THAT(Index.size(), Eq(2u));
  auto T1Blocks = Index.find({1, 1});
  ASSERT_THAT(T1Blocks, Not(Eq(Index.end())));

  // Expect only six records, because we're ignoring the BufferExtents record.
  EXPECT_THAT(T1Blocks->second,
              ElementsAre(Field(&BlockIndexer::Block::Records, SizeIs(6u)),
                          Field(&BlockIndexer::Block::Records, SizeIs(6u))));
  auto T2Blocks = Index.find({1, 2});
  ASSERT_THAT(T2Blocks, Not(Eq(Index.end())));
  EXPECT_THAT(T2Blocks->second, ElementsAre(Field(&BlockIndexer::Block::Records,
                                                  SizeIs(Eq(6u)))));
}

// FIXME: Support indexing V2 and V1 blocks.

} // namespace
} // namespace xray
} // namespace llvm
