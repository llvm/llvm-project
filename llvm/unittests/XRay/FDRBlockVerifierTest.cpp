//===- llvm/unittest/XRay/FDRBlockVerifierTest.cpp --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/Testing/Support/Error.h"
#include "llvm/XRay/BlockIndexer.h"
#include "llvm/XRay/BlockVerifier.h"
#include "llvm/XRay/FDRLogBuilder.h"
#include "llvm/XRay/FDRRecords.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace xray {
namespace {

using ::testing::ElementsAre;
using ::testing::Not;
using ::testing::SizeIs;

TEST(FDRBlockVerifierTest, ValidBlocksV3) {
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
  for (auto B : {std::ref(Block0), std::ref(Block1), std::ref(Block2)}) {
    for (auto &R : B.get())
      ASSERT_FALSE(errorToBool(R->apply(Indexer)));
    ASSERT_FALSE(errorToBool(Indexer.flush()));
  }

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
}

TEST(FDRBlockVerifierTest, MissingPIDRecord) {
  auto Block = LogBuilder()
                   .add<BufferExtents>(20)
                   .add<NewBufferRecord>(1)
                   .add<WallclockRecord>(1, 2)
                   .add<NewCPUIDRecord>(1, 2)
                   .add<FunctionRecord>(RecordTypes::ENTER, 1, 1)
                   .add<FunctionRecord>(RecordTypes::EXIT, 1, 100)
                   .consume();
  BlockVerifier Verifier;
  for (auto &R : Block)
    ASSERT_FALSE(errorToBool(R->apply(Verifier)));
  ASSERT_FALSE(errorToBool(Verifier.verify()));
}

TEST(FDRBlockVerifierTest, MissingBufferExtents) {
  auto Block = LogBuilder()
                   .add<NewBufferRecord>(1)
                   .add<WallclockRecord>(1, 2)
                   .add<NewCPUIDRecord>(1, 2)
                   .add<FunctionRecord>(RecordTypes::ENTER, 1, 1)
                   .add<FunctionRecord>(RecordTypes::EXIT, 1, 100)
                   .consume();
  BlockVerifier Verifier;
  for (auto &R : Block)
    ASSERT_FALSE(errorToBool(R->apply(Verifier)));
  ASSERT_FALSE(errorToBool(Verifier.verify()));
}

TEST(FDRBlockVerifierTest, IgnoreRecordsAfterEOB) {
  auto Block = LogBuilder()
                   .add<NewBufferRecord>(1)
                   .add<WallclockRecord>(1, 2)
                   .add<NewCPUIDRecord>(1, 2)
                   .add<EndBufferRecord>()
                   .add<FunctionRecord>(RecordTypes::ENTER, 1, 1)
                   .add<FunctionRecord>(RecordTypes::EXIT, 1, 100)
                   .consume();
  BlockVerifier Verifier;
  for (auto &R : Block)
    ASSERT_FALSE(errorToBool(R->apply(Verifier)));
  ASSERT_FALSE(errorToBool(Verifier.verify()));
}

TEST(FDRBlockVerifierTest, MalformedV2) {
  auto Block = LogBuilder()
                   .add<NewBufferRecord>(1)
                   .add<WallclockRecord>(1, 2)
                   .add<NewCPUIDRecord>(1, 2)
                   .add<FunctionRecord>(RecordTypes::ENTER, 1, 1)
                   .add<FunctionRecord>(RecordTypes::EXIT, 1, 100)
                   .add<NewBufferRecord>(2)
                   .consume();
  BlockVerifier Verifier;

  ASSERT_THAT(Block, SizeIs(6u));
  EXPECT_THAT_ERROR(Block[0]->apply(Verifier), Succeeded());
  EXPECT_THAT_ERROR(Block[1]->apply(Verifier), Succeeded());
  EXPECT_THAT_ERROR(Block[2]->apply(Verifier), Succeeded());
  EXPECT_THAT_ERROR(Block[3]->apply(Verifier), Succeeded());
  EXPECT_THAT_ERROR(Block[4]->apply(Verifier), Succeeded());
  EXPECT_THAT_ERROR(Block[5]->apply(Verifier), Failed());
}

} // namespace
} // namespace xray
} // namespace llvm
