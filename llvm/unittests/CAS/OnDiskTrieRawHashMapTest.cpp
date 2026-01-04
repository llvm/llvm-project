//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/OnDiskTrieRawHashMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

#if LLVM_ENABLE_ONDISK_CAS
using namespace llvm;
using namespace llvm::cas;

namespace {

struct OnDiskTrieRawHashMapTestFixture
    : public ::testing::TestWithParam<size_t> {
  static constexpr size_t MB = 1024u * 1024u;
  static constexpr size_t DataSize = 8; // Multiple of 8B.

  std::optional<unittest::TempDir> Temp;
  size_t NumHashBytes;

  void SetUp() override {
    Temp.emplace("trie-raw-hash-map", /*Unique=*/true);
    NumHashBytes = GetParam();
  }
  void TearDown() override { Temp.reset(); }

  Expected<OnDiskTrieRawHashMap> createTrie() {
    size_t NumHashBits = NumHashBytes * 8;
    return OnDiskTrieRawHashMap::create(
        Temp->path((Twine(NumHashBytes) + "B").str()), "index",
        /*NumHashBits=*/NumHashBits, DataSize, /*MaxFileSize=*/MB,
        /*NewInitialFileSize=*/std::nullopt);
  }
};

// Create tries with various sizes of hash and with data.
TEST_P(OnDiskTrieRawHashMapTestFixture, General) {
  std::optional<OnDiskTrieRawHashMap> Trie1;
  ASSERT_THAT_ERROR(createTrie().moveInto(Trie1), Succeeded());
  std::optional<OnDiskTrieRawHashMap> Trie2;
  ASSERT_THAT_ERROR(createTrie().moveInto(Trie2), Succeeded());

  uint8_t Hash0Bytes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  uint8_t Hash1Bytes[8] = {1, 0, 0, 0, 0, 0, 0, 0};
  auto Hash0 = ArrayRef(Hash0Bytes).take_front(NumHashBytes);
  auto Hash1 = ArrayRef(Hash1Bytes).take_front(NumHashBytes);
  constexpr StringLiteral Data0v1Bytes = "data0.v1";
  constexpr StringLiteral Data0v2Bytes = "data0.v2";
  constexpr StringLiteral Data1Bytes = "data1...";
  static_assert(Data0v1Bytes.size() == DataSize, "math error");
  static_assert(Data0v2Bytes.size() == DataSize, "math error");
  static_assert(Data1Bytes.size() == DataSize, "math error");
  ArrayRef<char> Data0v1 = ArrayRef(Data0v1Bytes.data(), Data0v1Bytes.size());
  ArrayRef<char> Data0v2 = ArrayRef(Data0v2Bytes.data(), Data0v2Bytes.size());
  ArrayRef<char> Data1 = ArrayRef(Data1Bytes.data(), Data1Bytes.size());

  // Lookup when trie is empty.
  EXPECT_FALSE(Trie1->find(Hash0));

  // Insert.
  std::optional<FileOffset> Offset;
  std::optional<MutableArrayRef<char>> Data;
  {
    std::optional<OnDiskTrieRawHashMap::OnDiskPtr> Insertion;
    ASSERT_THAT_ERROR(Trie1->insert({Hash0, Data0v1}).moveInto(Insertion),
                      Succeeded());
    EXPECT_EQ(Hash0, (*Insertion)->Hash);
    EXPECT_EQ(Data0v1, (*Insertion)->Data);
    EXPECT_TRUE(isAddrAligned(Align(8), (*Insertion)->Data.data()));

    Offset = Insertion->getOffset();
    Data = (*Insertion)->Data;
  }

  // Find.
  {
    auto Lookup = Trie1->find(Hash0);
    ASSERT_TRUE(Lookup);
    EXPECT_EQ(Hash0, Lookup->Hash);
    EXPECT_EQ(Data0v1, Lookup->Data);
    EXPECT_EQ(Offset->get(), Lookup.getOffset().get());
  }

  // Find in a different instance of the same on-disk trie that existed
  // before the insertion.
  {
    auto Lookup = Trie2->find(Hash0);
    ASSERT_TRUE(Lookup);
    EXPECT_EQ(Hash0, Lookup->Hash);
    EXPECT_EQ(Data0v1, Lookup->Data);
    EXPECT_EQ(Offset->get(), Lookup.getOffset().get());
  }

  // Create a new instance and check that too.
  Trie2.reset();
  ASSERT_THAT_ERROR(createTrie().moveInto(Trie2), Succeeded());
  {
    auto Lookup = Trie2->find(Hash0);
    ASSERT_TRUE(Lookup);
    EXPECT_EQ(Hash0, Lookup->Hash);
    EXPECT_EQ(Data0v1, Lookup->Data);
    EXPECT_EQ(Offset->get(), Lookup.getOffset().get());
  }

  // Change the data.
  llvm::copy(Data0v2, Data->data());
  {
    auto Lookup = Trie2->find(Hash0);
    ASSERT_TRUE(Lookup);
    EXPECT_EQ(Hash0, Lookup->Hash);
    EXPECT_EQ(Data0v2, Lookup->Data);
    EXPECT_EQ(Offset->get(), Lookup.getOffset().get());
  }

  // Find different hash.
  EXPECT_FALSE(Trie1->find(Hash1));
  EXPECT_FALSE(Trie2->find(Hash1));

  // Recover from an offset.
  {
    OnDiskTrieRawHashMap::ConstOnDiskPtr Recovered;
    ASSERT_THAT_ERROR(Trie1->recoverFromFileOffset(*Offset).moveInto(Recovered),
                      Succeeded());
    ASSERT_TRUE(Recovered);
    EXPECT_EQ(Offset->get(), Recovered.getOffset().get());
    EXPECT_EQ(Hash0, Recovered->Hash);
    EXPECT_EQ(Data0v2, Recovered->Data);
  }

  // Recover from a bad offset.
  {
    FileOffset BadOffset(1);
    OnDiskTrieRawHashMap::ConstOnDiskPtr Recovered;
    ASSERT_THAT_ERROR(
        Trie1->recoverFromFileOffset(BadOffset).moveInto(Recovered), Failed());
  }

  // Insert another thing.
  {
    std::optional<OnDiskTrieRawHashMap::OnDiskPtr> Insertion;
    ASSERT_THAT_ERROR(Trie1->insert({Hash1, Data1}).moveInto(Insertion),
                      Succeeded());
    EXPECT_EQ(Hash1, (*Insertion)->Hash);
    EXPECT_EQ(Data1, (*Insertion)->Data);
    EXPECT_TRUE(isAddrAligned(Align(8), (*Insertion)->Data.data()));

    EXPECT_NE(Offset->get(), Insertion->getOffset().get());
  }

  // Validate.
  {
    auto RecordVerify =
        [&](FileOffset Offset,
            OnDiskTrieRawHashMap::ConstValueProxy Proxy) -> Error {
      if (Proxy.Hash.size() != NumHashBytes)
        return createStringError("wrong hash size");
      if (Proxy.Data.size() != DataSize)
        return createStringError("wrong data size");

      return Error::success();
    };
    ASSERT_THAT_ERROR(Trie1->validate(RecordVerify), Succeeded());
    ASSERT_THAT_ERROR(Trie2->validate(RecordVerify), Succeeded());
  }

  // Size and capacity.
  {
    EXPECT_EQ(Trie1->capacity(), MB);
    EXPECT_EQ(Trie2->capacity(), MB);
    EXPECT_LE(Trie1->size(), MB);
    EXPECT_LE(Trie2->size(), MB);
  }
}

INSTANTIATE_TEST_SUITE_P(OnDiskTrieRawHashMapTest,
                         OnDiskTrieRawHashMapTestFixture,
                         ::testing::Values(1, 2, 4, 8));

TEST(OnDiskTrieRawHashMapTest, OutOfSpace) {
  unittest::TempDir Temp("trie-raw-hash-map", /*Unique=*/true);
  std::optional<OnDiskTrieRawHashMap> Trie;

  // Too small to create header.
  ASSERT_THAT_ERROR(OnDiskTrieRawHashMap::create(
                        Temp.path("NoSpace1").str(), "index",
                        /*NumHashBits=*/8, /*DataSize=*/8, /*MaxFileSize=*/8,
                        /*NewInitialFileSize=*/std::nullopt)
                        .moveInto(Trie),
                    Failed());

  // Just enough for root node but not enough for any insertion.
  ASSERT_THAT_ERROR(OnDiskTrieRawHashMap::create(
                        Temp.path("NoSpace2").str(), "index",
                        /*NumHashBits=*/8, /*DataSize=*/8, /*MaxFileSize=*/118,
                        /*NewInitialFileSize=*/std::nullopt,
                        /*NewTableNumRootBits=*/1, /*NewTableNumSubtrieBits=*/1)
                        .moveInto(Trie),
                    Succeeded());
  uint8_t Hash0Bytes[1] = {0};
  auto Hash0 = ArrayRef(Hash0Bytes);
  constexpr StringLiteral Data0v1Bytes = "data0.v1";
  ArrayRef<char> Data0v1 = ArrayRef(Data0v1Bytes.data(), Data0v1Bytes.size());
  std::optional<OnDiskTrieRawHashMap::OnDiskPtr> Insertion;
  ASSERT_THAT_ERROR(Trie->insert({Hash0, Data0v1}).moveInto(Insertion),
                    Failed());
}

} // namespace

#endif // LLVM_ENABLE_ONDISK_CAS
