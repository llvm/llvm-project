//===- OutlinedHashTreeRecordTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CGData/OutlinedHashTreeRecord.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(OutlinedHashTreeRecordTest, Empty) {
  OutlinedHashTreeRecord HashTreeRecord;
  ASSERT_TRUE(HashTreeRecord.empty());
}

TEST(OutlinedHashTreeRecordTest, Print) {
  OutlinedHashTreeRecord HashTreeRecord;
  HashTreeRecord.HashTree->insert({{1, 2}, 3});

  const char *ExpectedTreeStr = R"(---
0:
  Hash:            0x0
  Terminals:       0
  SuccessorIds:    [ 1 ]
1:
  Hash:            0x1
  Terminals:       0
  SuccessorIds:    [ 2 ]
2:
  Hash:            0x2
  Terminals:       3
  SuccessorIds:    [  ]
...
)";
  std::string TreeDump;
  raw_string_ostream OS(TreeDump);
  HashTreeRecord.print(OS);
  EXPECT_EQ(ExpectedTreeStr, TreeDump);
}

TEST(OutlinedHashTreeRecordTest, Stable) {
  OutlinedHashTreeRecord HashTreeRecord1;
  HashTreeRecord1.HashTree->insert({{1, 2}, 4});
  HashTreeRecord1.HashTree->insert({{1, 3}, 5});

  OutlinedHashTreeRecord HashTreeRecord2;
  HashTreeRecord2.HashTree->insert({{1, 3}, 5});
  HashTreeRecord2.HashTree->insert({{1, 2}, 4});

  // Output is stable regardless of insertion order.
  std::string TreeDump1;
  raw_string_ostream OS1(TreeDump1);
  HashTreeRecord1.print(OS1);
  std::string TreeDump2;
  raw_string_ostream OS2(TreeDump2);
  HashTreeRecord2.print(OS2);

  EXPECT_EQ(TreeDump1, TreeDump2);
}

TEST(OutlinedHashTreeRecordTest, Serialize) {
  OutlinedHashTreeRecord HashTreeRecord1;
  HashTreeRecord1.HashTree->insert({{1, 2}, 4});
  HashTreeRecord1.HashTree->insert({{1, 3}, 5});

  // Serialize and deserialize the tree.
  SmallVector<char> Out;
  raw_svector_ostream OS(Out);
  HashTreeRecord1.serialize(OS);

  OutlinedHashTreeRecord HashTreeRecord2;
  const uint8_t *Data = reinterpret_cast<const uint8_t *>(Out.data());
  HashTreeRecord2.deserialize(Data);

  // Two trees should be identical.
  std::string TreeDump1;
  raw_string_ostream OS1(TreeDump1);
  HashTreeRecord1.print(OS1);
  std::string TreeDump2;
  raw_string_ostream OS2(TreeDump2);
  HashTreeRecord2.print(OS2);

  EXPECT_EQ(TreeDump1, TreeDump2);
}

TEST(OutlinedHashTreeRecordTest, SerializeYAML) {
  OutlinedHashTreeRecord HashTreeRecord1;
  HashTreeRecord1.HashTree->insert({{1, 2}, 4});
  HashTreeRecord1.HashTree->insert({{1, 3}, 5});

  // Serialize and deserialize the tree in a YAML format.
  std::string Out;
  raw_string_ostream OS(Out);
  yaml::Output YOS(OS);
  HashTreeRecord1.serializeYAML(YOS);

  OutlinedHashTreeRecord HashTreeRecord2;
  yaml::Input YIS(StringRef(Out.data(), Out.size()));
  HashTreeRecord2.deserializeYAML(YIS);

  // Two trees should be identical.
  std::string TreeDump1;
  raw_string_ostream OS1(TreeDump1);
  HashTreeRecord1.print(OS1);
  std::string TreeDump2;
  raw_string_ostream OS2(TreeDump2);
  HashTreeRecord2.print(OS2);

  EXPECT_EQ(TreeDump1, TreeDump2);
}

TEST(OutlinedHashTreeRecordTest, InPlaceRead) {
  // A tree with a branch, so a node has two successors and the in-place
  // binary search is exercised.
  OutlinedHashTreeRecord Eager;
  Eager.HashTree->insert({{1, 2}, 4});
  Eager.HashTree->insert({{1, 3}, 5});

  // Serialize, then read the same bytes in place.
  SmallVector<char> Out;
  raw_svector_ostream OS(Out);
  Eager.serialize(OS);
  std::shared_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBufferCopy(StringRef(Out.data(), Out.size()));

  std::unique_ptr<OutlinedHashTree> LazyTree =
      OutlinedHashTreeRecord::createReadInPlace(Buffer, /*BlobOffset=*/0);
  ASSERT_TRUE(LazyTree->isReadInPlace());

  // empty() and find() read the buffer in place and agree with the eagerly
  // loaded tree. A read-only tree exposes only this consume path.
  EXPECT_FALSE(LazyTree->empty());
  EXPECT_EQ(LazyTree->find({1, 2}), Eager.HashTree->find({1, 2}));
  EXPECT_EQ(LazyTree->find({1, 3}), Eager.HashTree->find({1, 3}));
  EXPECT_EQ(*LazyTree->find({1, 2}), 4u);
  EXPECT_EQ(*LazyTree->find({1, 3}), 5u);
  // A missing successor and a missing first instruction both miss.
  EXPECT_EQ(LazyTree->find({1, 9}), 0u);
  EXPECT_EQ(LazyTree->find({9}), 0u);
  // A non-terminal prefix has no terminal count.
  EXPECT_FALSE(LazyTree->find({1}).has_value());
  EXPECT_TRUE(LazyTree->isReadInPlace());

  // Converting the read-in-place tree materializes it once; its whole-tree
  // counts match the eager tree.
  auto InMem = OutlinedHashTreeRecord::createInMemory(std::move(LazyTree));
  EXPECT_FALSE(InMem->isReadInPlace());
  EXPECT_EQ(InMem->size(), Eager.HashTree->size());
  EXPECT_EQ(InMem->size(/*GetTerminalCountOnly=*/true),
            Eager.HashTree->size(/*GetTerminalCountOnly=*/true));
  EXPECT_EQ(InMem->depth(), Eager.HashTree->depth());
  // find() still works on the materialized tree.
  EXPECT_EQ(*InMem->find({1, 3}), 5u);
}

} // end namespace
