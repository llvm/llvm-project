//===- OutlinedHashTreeRecordTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CGData/OutlinedHashTreeRecord.h"
#include "llvm/Support/Endian.h"
#include "llvm/Testing/Support/Error.h"
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
  ASSERT_THAT_ERROR(HashTreeRecord2.deserialize(Data, Data + Out.size()),
                    Succeeded());

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

  auto LazyTreeOrErr =
      OutlinedHashTreeRecord::createReadInPlace(Buffer, /*BlobOffset=*/0);
  ASSERT_THAT_EXPECTED(LazyTreeOrErr, Succeeded());
  std::unique_ptr<OutlinedHashTree> LazyTree = std::move(*LazyTreeOrErr);
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
  auto InMemOrErr = OutlinedHashTreeRecord::createInMemory(std::move(LazyTree));
  ASSERT_THAT_EXPECTED(InMemOrErr, Succeeded());
  auto &InMem = *InMemOrErr;
  EXPECT_FALSE(InMem->isReadInPlace());
  EXPECT_EQ(InMem->size(), Eager.HashTree->size());
  EXPECT_EQ(InMem->size(/*GetTerminalCountOnly=*/true),
            Eager.HashTree->size(/*GetTerminalCountOnly=*/true));
  EXPECT_EQ(InMem->depth(), Eager.HashTree->depth());
  // find() still works on the materialized tree.
  EXPECT_EQ(*InMem->find({1, 3}), 5u);
}

// The tree below has a root with two leaf successors, hashes 1 and 2. The root
// block comes first in the block region, right after the 16-byte blob header.
// Its successor entries, sorted by hash, are 16 bytes each: the child hash,
// then the child's block offset.
constexpr unsigned NumNodesAt = 0;
constexpr unsigned BlockSizeAt = 8;
constexpr unsigned FirstChildHashAt = 24;
constexpr unsigned FirstChildOffsetAt = 32;
constexpr unsigned SecondChildHashAt = 40;

SmallVector<char> serializeTwoLeafTree() {
  OutlinedHashTreeRecord Record;
  Record.HashTree->insert({{1}, 1});
  Record.HashTree->insert({{2}, 1});
  SmallVector<char> Out;
  raw_svector_ostream OS(Out);
  Record.serialize(OS);
  EXPECT_EQ(support::endian::read64le(Out.data() + FirstChildHashAt), 1u);
  EXPECT_EQ(support::endian::read64le(Out.data() + SecondChildHashAt), 2u);
  return Out;
}

Error deserialize(ArrayRef<char> Blob) {
  OutlinedHashTreeRecord Record;
  const auto *Data = reinterpret_cast<const unsigned char *>(Blob.data());
  return Record.deserialize(Data, Data + Blob.size());
}

Expected<std::unique_ptr<OutlinedHashTree>> readInPlace(ArrayRef<char> Blob) {
  return OutlinedHashTreeRecord::createReadInPlace(
      MemoryBuffer::getMemBufferCopy(StringRef(Blob.data(), Blob.size())),
      /*BlobOffset=*/0);
}

TEST(OutlinedHashTreeRecordTest, DeserializeMalformed) {
  SmallVector<char> Blob = serializeTwoLeafTree();
  ASSERT_THAT_ERROR(deserialize(Blob), Succeeded());

  EXPECT_THAT_ERROR(
      deserialize(ArrayRef(Blob).drop_back()),
      FailedWithMessage(
          "malformed codegen data: truncated outlined hash tree"));

  SmallVector<char> OutOfBounds = Blob;
  support::endian::write64le(OutOfBounds.data() + FirstChildOffsetAt,
                             1ULL << 40);
  EXPECT_THAT_ERROR(deserialize(OutOfBounds),
                    FailedWithMessage("malformed codegen data: outlined hash "
                                      "tree block out of bounds"));

  SmallVector<char> Misaligned = Blob;
  support::endian::write64le(
      Misaligned.data() + FirstChildOffsetAt,
      support::endian::read64le(Blob.data() + FirstChildOffsetAt) + 1);
  EXPECT_THAT_ERROR(deserialize(Misaligned),
                    FailedWithMessage("malformed codegen data: outlined hash "
                                      "tree block misaligned"));

  // A self edge: the root's first child is the root itself.
  SmallVector<char> SelfEdge = Blob;
  support::endian::write64le(SelfEdge.data() + FirstChildOffsetAt, 0);
  EXPECT_THAT_ERROR(deserialize(SelfEdge),
                    FailedWithMessage("malformed codegen data: outlined hash "
                                      "tree block reached more than once"));

  SmallVector<char> Duplicate = Blob;
  support::endian::write64le(Duplicate.data() + SecondChildHashAt, 1);
  EXPECT_THAT_ERROR(
      deserialize(Duplicate),
      FailedWithMessage("malformed codegen data: outlined hash tree "
                        "successors not sorted by unique hash"));

  SmallVector<char> WrongNodeCount = Blob;
  support::endian::write32le(WrongNodeCount.data() + NumNodesAt, 4);
  EXPECT_THAT_ERROR(deserialize(WrongNodeCount),
                    FailedWithMessage("malformed codegen data: outlined hash "
                                      "tree blocks do not match the blob "
                                      "header"));

  SmallVector<char> MisalignedBlockSize = Blob;
  support::endian::write64le(
      MisalignedBlockSize.data() + BlockSizeAt,
      support::endian::read64le(Blob.data() + BlockSizeAt) - 1);
  EXPECT_THAT_ERROR(deserialize(MisalignedBlockSize),
                    FailedWithMessage("malformed codegen data: outlined hash "
                                      "tree block region size misaligned"));
}

TEST(OutlinedHashTreeRecordTest, InPlaceReadMalformed) {
  SmallVector<char> Blob = serializeTwoLeafTree();

  EXPECT_THAT_EXPECTED(
      readInPlace(ArrayRef(Blob).drop_back()),
      FailedWithMessage(
          "malformed codegen data: truncated outlined hash tree"));

  support::endian::write64le(Blob.data() + FirstChildOffsetAt, 1ULL << 40);
  auto TreeOrErr = readInPlace(Blob);
  ASSERT_THAT_EXPECTED(TreeOrErr, Succeeded());
  std::unique_ptr<OutlinedHashTree> Tree = std::move(*TreeOrErr);
  EXPECT_FALSE(Tree->find({1}).has_value());
  EXPECT_EQ(Tree->find({1, 2}), 0u);
  EXPECT_EQ(Tree->find({2}), 1u);

  EXPECT_THAT_EXPECTED(
      OutlinedHashTreeRecord::createInMemory(std::move(Tree)),
      FailedWithMessage(
          "malformed codegen data: outlined hash tree block out of bounds"));
}

} // end namespace
