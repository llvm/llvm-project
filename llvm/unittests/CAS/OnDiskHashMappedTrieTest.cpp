//===- OnDiskHashMappedTrieTest.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/OnDiskHashMappedTrie.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

#if LLVM_ENABLE_ONDISK_CAS
using namespace llvm;
using namespace llvm::cas;

namespace {

TEST(OnDiskHashMappedTrieTest, Insertion) {
  unittest::TempDir Temp("on-disk-hash-mapped-trie", /*Unique=*/true);

  // Create tries with various sizes of hash and with data.
  //
  // NOTE: The check related to \a recoverFromFileOffset() catches a potential
  // off-by-one bounds-checking bug when the trie record size (data + hash) add
  // up to a multiple of 8B. Iterate through a few different hash sizes to
  // check it both ways.
  constexpr size_t MB = 1024u * 1024u;
  constexpr size_t DataSize = 8; // Multiple of 8B.
  for (size_t NumHashBytes : {1, 2, 4, 8}) {
    size_t NumHashBits = NumHashBytes * 8;

    auto createTrie = [&]() {
      return OnDiskHashMappedTrie::create(
          Temp.path((Twine(NumHashBytes) + "B").str()), "index",
          /*NumHashBits=*/NumHashBits, DataSize, /*MaxFileSize=*/MB,
          /*NewInitialFileSize=*/std::nullopt);
    };

    std::optional<OnDiskHashMappedTrie> Trie1;
    ASSERT_THAT_ERROR(createTrie().moveInto(Trie1), Succeeded());
    std::optional<OnDiskHashMappedTrie> Trie2;
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
    ArrayRef<char> Data0v1 =
        ArrayRef(Data0v1Bytes.data(), Data0v1Bytes.size());
    ArrayRef<char> Data0v2 =
        ArrayRef(Data0v2Bytes.data(), Data0v2Bytes.size());
    ArrayRef<char> Data1 = ArrayRef(Data1Bytes.data(), Data1Bytes.size());

    // Lookup when trie is empty.
    EXPECT_FALSE(Trie1->find(Hash0));

    // Insert.
    std::optional<FileOffset> Offset;
    std::optional<MutableArrayRef<char>> Data;
    {
      auto Insertion = Trie1->insert({Hash0, Data0v1});
      ASSERT_TRUE(Insertion);
      EXPECT_EQ(Hash0, Insertion->Hash);
      EXPECT_EQ(Data0v1, Insertion->Data);
      EXPECT_TRUE(isAddrAligned(Align(8), Insertion->Data.data()));

      Offset = Insertion.getOffset();
      Data = Insertion->Data;
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
      auto Recovered = Trie1->recoverFromFileOffset(*Offset);
      ASSERT_TRUE(Recovered);
      EXPECT_EQ(Offset->get(), Recovered.getOffset().get());
      EXPECT_EQ(Hash0, Recovered->Hash);
      EXPECT_EQ(Data0v2, Recovered->Data);
    }

    // Insert another thing.
    {
      auto Insertion = Trie1->insert({Hash1, Data1});
      ASSERT_TRUE(Insertion);
      EXPECT_EQ(Hash1, Insertion->Hash);
      EXPECT_EQ(Data1, Insertion->Data);
      EXPECT_TRUE(isAddrAligned(Align(8), Insertion->Data.data()));
      EXPECT_NE(Offset->get(), Insertion.getOffset().get());
    }
  }
}

} // namespace

#endif
