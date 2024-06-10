//===-------------- PGOCtxProfReadWriteTest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitstream/BitstreamReader.h"
#include "llvm/ProfileData/CtxInstrContextNode.h"
#include "llvm/ProfileData/PGOCtxProfReader.h"
#include "llvm/ProfileData/PGOCtxProfWriter.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::ctx_profile;

class PGOCtxProfRWTest : public ::testing::Test {
  std::vector<std::unique_ptr<char[]>> Nodes;
  std::map<GUID, const ContextNode *> Roots;

public:
  ContextNode *createNode(GUID Guid, uint32_t NrCounters, uint32_t NrCallsites,
                          ContextNode *Next = nullptr) {
    auto AllocSize = ContextNode::getAllocSize(NrCounters, NrCallsites);
    auto *Mem = Nodes.emplace_back(std::make_unique<char[]>(AllocSize)).get();
    std::memset(Mem, 0, AllocSize);
    auto *Ret = new (Mem) ContextNode(Guid, NrCounters, NrCallsites, Next);
    return Ret;
  }

  void SetUp() override {
    // Root (guid 1) has 2 callsites, one used for an indirect call to either
    // guid 2 or 4.
    // guid 2 calls guid 5
    // guid 5 calls guid 2
    // there's also a second root, guid3.
    auto *Root1 = createNode(1, 2, 2);
    Root1->counters()[0] = 10;
    Root1->counters()[1] = 11;
    Roots.insert({1, Root1});
    auto *L1 = createNode(2, 1, 1);
    L1->counters()[0] = 12;
    Root1->subContexts()[1] = createNode(4, 3, 1, L1);
    Root1->subContexts()[1]->counters()[0] = 13;
    Root1->subContexts()[1]->counters()[1] = 14;
    Root1->subContexts()[1]->counters()[2] = 15;

    auto *L3 = createNode(5, 6, 3);
    for (auto I = 0; I < 6; ++I)
      L3->counters()[I] = 16 + I;
    L1->subContexts()[0] = L3;
    L3->subContexts()[2] = createNode(2, 1, 1);
    L3->subContexts()[2]->counters()[0] = 30;
    auto *Root2 = createNode(3, 1, 0);
    Root2->counters()[0] = 40;
    Roots.insert({3, Root2});
  }

  const std::map<GUID, const ContextNode *> &roots() const { return Roots; }
};

void checkSame(const ContextNode &Raw, const PGOContextualProfile &Profile) {
  EXPECT_EQ(Raw.guid(), Profile.guid());
  ASSERT_EQ(Raw.counters_size(), Profile.counters().size());
  for (auto I = 0U; I < Raw.counters_size(); ++I)
    EXPECT_EQ(Raw.counters()[I], Profile.counters()[I]);

  for (auto I = 0U; I < Raw.callsites_size(); ++I) {
    if (Raw.subContexts()[I] == nullptr)
      continue;
    EXPECT_TRUE(Profile.hasCallsite(I));
    const auto &ProfileTargets = Profile.callsite(I);

    std::map<GUID, const ContextNode *> Targets;
    for (const auto *N = Raw.subContexts()[I]; N; N = N->next())
      EXPECT_TRUE(Targets.insert({N->guid(), N}).second);

    EXPECT_EQ(Targets.size(), ProfileTargets.size());
    for (auto It : Targets) {
      auto PIt = ProfileTargets.find(It.second->guid());
      EXPECT_NE(PIt, ProfileTargets.end());
      checkSame(*It.second, PIt->second);
    }
  }
}

TEST_F(PGOCtxProfRWTest, RoundTrip) {
  llvm::unittest::TempFile ProfileFile("ctx_profile", "", "", /*Unique*/ true);
  {
    std::error_code EC;
    raw_fd_stream Out(ProfileFile.path(), EC);
    ASSERT_FALSE(EC);
    {
      PGOCtxProfileWriter Writer(Out);
      for (auto &[_, R] : roots())
        Writer.write(*R);
    }
  }
  {
    ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
        MemoryBuffer::getFile(ProfileFile.path());
    ASSERT_TRUE(!!MB);
    ASSERT_NE(*MB, nullptr);
    BitstreamCursor Cursor((*MB)->getBuffer());
    PGOCtxProfileReader Reader(Cursor);
    auto Expected = Reader.loadContexts();
    ASSERT_TRUE(!!Expected);
    auto &Ctxes = *Expected;
    EXPECT_EQ(Ctxes.size(), roots().size());
    EXPECT_EQ(Ctxes.size(), 2U);
    for (auto &[G, R] : roots())
      checkSame(*R, Ctxes.find(G)->second);
  }
}

TEST_F(PGOCtxProfRWTest, InvalidCounters) {
  auto *R = createNode(1, 0, 1);
  llvm::unittest::TempFile ProfileFile("ctx_profile", "", "", /*Unique*/ true);
  {
    std::error_code EC;
    raw_fd_stream Out(ProfileFile.path(), EC);
    ASSERT_FALSE(EC);
    {
      PGOCtxProfileWriter Writer(Out);
      Writer.write(*R);
    }
  }
  {
    auto MB = MemoryBuffer::getFile(ProfileFile.path());
    ASSERT_TRUE(!!MB);
    ASSERT_NE(*MB, nullptr);
    BitstreamCursor Cursor((*MB)->getBuffer());
    PGOCtxProfileReader Reader(Cursor);
    auto Expected = Reader.loadContexts();
    EXPECT_FALSE(Expected);
    consumeError(Expected.takeError());
  }
}

TEST_F(PGOCtxProfRWTest, Empty) {
  BitstreamCursor Cursor("");
  PGOCtxProfileReader Reader(Cursor);
  auto Expected = Reader.loadContexts();
  EXPECT_FALSE(Expected);
  consumeError(Expected.takeError());
}

TEST_F(PGOCtxProfRWTest, Invalid) {
  BitstreamCursor Cursor("Surely this is not valid");
  PGOCtxProfileReader Reader(Cursor);
  auto Expected = Reader.loadContexts();
  EXPECT_FALSE(Expected);
  consumeError(Expected.takeError());
}

TEST_F(PGOCtxProfRWTest, ValidButEmpty) {
  llvm::unittest::TempFile ProfileFile("ctx_profile", "", "", /*Unique*/ true);
  {
    std::error_code EC;
    raw_fd_stream Out(ProfileFile.path(), EC);
    ASSERT_FALSE(EC);
    {
      PGOCtxProfileWriter Writer(Out);
      // don't write anything - this will just produce the metadata subblock.
    }
  }
  {
    auto MB = MemoryBuffer::getFile(ProfileFile.path());
    ASSERT_TRUE(!!MB);
    ASSERT_NE(*MB, nullptr);
    BitstreamCursor Cursor((*MB)->getBuffer());
    PGOCtxProfileReader Reader(Cursor);
    auto Expected = Reader.loadContexts();
    EXPECT_TRUE(!!Expected);
    EXPECT_TRUE(Expected->empty());
  }
}

TEST_F(PGOCtxProfRWTest, WrongVersion) {
  llvm::unittest::TempFile ProfileFile("ctx_profile", "", "", /*Unique*/ true);
  {
    std::error_code EC;
    raw_fd_stream Out(ProfileFile.path(), EC);
    ASSERT_FALSE(EC);
    {
      PGOCtxProfileWriter Writer(Out, PGOCtxProfileWriter::CurrentVersion + 1);
    }
  }
  {
    auto MB = MemoryBuffer::getFile(ProfileFile.path());
    ASSERT_TRUE(!!MB);
    ASSERT_NE(*MB, nullptr);
    BitstreamCursor Cursor((*MB)->getBuffer());
    PGOCtxProfileReader Reader(Cursor);
    auto Expected = Reader.loadContexts();
    EXPECT_FALSE(Expected);
    consumeError(Expected.takeError());
  }
}

TEST_F(PGOCtxProfRWTest, DuplicateRoots) {
  llvm::unittest::TempFile ProfileFile("ctx_profile", "", "", /*Unique*/ true);
  {
    std::error_code EC;
    raw_fd_stream Out(ProfileFile.path(), EC);
    ASSERT_FALSE(EC);
    {
      PGOCtxProfileWriter Writer(Out);
      Writer.write(*createNode(1, 1, 1));
      Writer.write(*createNode(1, 1, 1));
    }
  }
  {
    auto MB = MemoryBuffer::getFile(ProfileFile.path());
    ASSERT_TRUE(!!MB);
    ASSERT_NE(*MB, nullptr);
    BitstreamCursor Cursor((*MB)->getBuffer());
    PGOCtxProfileReader Reader(Cursor);
    auto Expected = Reader.loadContexts();
    EXPECT_FALSE(Expected);
    consumeError(Expected.takeError());
  }
}

TEST_F(PGOCtxProfRWTest, DuplicateTargets) {
  llvm::unittest::TempFile ProfileFile("ctx_profile", "", "", /*Unique*/ true);
  {
    std::error_code EC;
    raw_fd_stream Out(ProfileFile.path(), EC);
    ASSERT_FALSE(EC);
    {
      auto *R = createNode(1, 1, 1);
      auto *L1 = createNode(2, 1, 0);
      auto *L2 = createNode(2, 1, 0, L1);
      R->subContexts()[0] = L2;
      PGOCtxProfileWriter Writer(Out);
      Writer.write(*R);
    }
  }
  {
    auto MB = MemoryBuffer::getFile(ProfileFile.path());
    ASSERT_TRUE(!!MB);
    ASSERT_NE(*MB, nullptr);
    BitstreamCursor Cursor((*MB)->getBuffer());
    PGOCtxProfileReader Reader(Cursor);
    auto Expected = Reader.loadContexts();
    EXPECT_FALSE(Expected);
    consumeError(Expected.takeError());
  }
}
