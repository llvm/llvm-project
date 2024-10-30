//===- TrieRawHashMapTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TrieRawHashMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/SHA1.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace llvm {
class TrieRawHashMapTestHelper {
public:
  TrieRawHashMapTestHelper() = default;

  void setTrie(ThreadSafeTrieRawHashMapBase *T) { Trie = T; }

  ThreadSafeTrieRawHashMapBase::PointerBase getRoot() const {
    return Trie->getRoot();
  }
  unsigned getStartBit(ThreadSafeTrieRawHashMapBase::PointerBase P) const {
    return Trie->getStartBit(P);
  }
  unsigned getNumBits(ThreadSafeTrieRawHashMapBase::PointerBase P) const {
    return Trie->getNumBits(P);
  }
  unsigned getNumSlotUsed(ThreadSafeTrieRawHashMapBase::PointerBase P) const {
    return Trie->getNumSlotUsed(P);
  }
  unsigned getNumTries() const { return Trie->getNumTries(); }
  std::string
  getTriePrefixAsString(ThreadSafeTrieRawHashMapBase::PointerBase P) const {
    return Trie->getTriePrefixAsString(P);
  }
  ThreadSafeTrieRawHashMapBase::PointerBase
  getNextTrie(ThreadSafeTrieRawHashMapBase::PointerBase P) const {
    return Trie->getNextTrie(P);
  }

private:
  ThreadSafeTrieRawHashMapBase *Trie = nullptr;
};
} // namespace llvm

namespace {
template <typename DataType, size_t HashSize = sizeof(uint64_t)>
class SimpleTrieHashMapTest : public TrieRawHashMapTestHelper,
                              public ::testing::Test {
public:
  using NumType = DataType;
  using HashType = std::array<uint8_t, HashSize>;
  using TrieType = ThreadSafeTrieRawHashMap<DataType, sizeof(HashType)>;

  TrieType &createTrie(size_t RootBits, size_t SubtrieBits) {
    auto &Ret = Trie.emplace(RootBits, SubtrieBits);
    TrieRawHashMapTestHelper::setTrie(&Ret);
    return Ret;
  }

  void destroyTrie() { Trie.reset(); }
  ~SimpleTrieHashMapTest() { destroyTrie(); }

  // Use the number itself as hash to test the pathological case.
  static HashType hash(uint64_t Num) {
    uint64_t HashN =
        llvm::support::endian::byte_swap(Num, llvm::endianness::big);
    HashType Hash;
    memcpy(&Hash[0], &HashN, sizeof(HashType));
    return Hash;
  };

private:
  std::optional<TrieType> Trie;
};

using SmallNodeTrieTest = SimpleTrieHashMapTest<uint64_t>;

TEST_F(SmallNodeTrieTest, TrieAllocation) {
  NumType Numbers[] = {
      0x0, std::numeric_limits<NumType>::max(),      0x1, 0x2,
      0x3, std::numeric_limits<NumType>::max() - 1u,
  };

  unsigned ExpectedTries[] = {
      1,       // Allocate Root.
      1,       // Both on the root.
      64,      // 0 and 1 sinks all the way down.
      64,      // no new allocation needed.
      65,      // need a new node between 2 and 3.
      65 + 63, // 63 new allocation to sink two big numbers all the way.
  };

  const char *ExpectedPrefix[] = {
      "", // Root.
      "", // Root.
      "00000000000000[0000000]",
      "00000000000000[0000000]",
      "00000000000000[0000001]",
      "ffffffffffffff[1111111]",
  };

  // Use root and subtrie sizes of 1 so this gets sunk quite deep.
  auto &Trie = createTrie(/*RootBits=*/1, /*SubtrieBits=*/1);

  for (unsigned I = 0; I < 6; ++I) {
    // Lookup first to exercise hint code for deep tries.
    TrieType::pointer Lookup = Trie.find(hash(Numbers[I]));
    EXPECT_FALSE(Lookup);

    Trie.insert(Lookup, TrieType::value_type(hash(Numbers[I]), Numbers[I]));
    EXPECT_EQ(getNumTries(), ExpectedTries[I]);
    EXPECT_EQ(getTriePrefixAsString(getNextTrie(getRoot())), ExpectedPrefix[I]);
  }
}

TEST_F(SmallNodeTrieTest, TrieStructure) {
  NumType Numbers[] = {
      // Three numbers that will nest deeply to test (1) sinking subtries and
      // (2) deep, non-trivial hints.
      std::numeric_limits<NumType>::max(),
      std::numeric_limits<NumType>::max() - 2u,
      std::numeric_limits<NumType>::max() - 3u,
      // One number to stay at the top-level.
      0x37,
  };

  // Use root and subtrie sizes of 1 so this gets sunk quite deep.
  auto &Trie = createTrie(/*RootBits=*/1, /*SubtrieBits=*/1);

  for (NumType N : Numbers) {
    // Lookup first to exercise hint code for deep tries.
    TrieType::pointer Lookup = Trie.find(hash(N));
    EXPECT_FALSE(Lookup);

    Trie.insert(Lookup, TrieType::value_type(hash(N), N));
  }
  for (NumType N : Numbers) {
    TrieType::pointer Lookup = Trie.find(hash(N));
    EXPECT_TRUE(Lookup);
    if (!Lookup)
      continue;
    EXPECT_EQ(hash(N), Lookup->Hash);
    EXPECT_EQ(N, Lookup->Data);

    // Confirm a subsequent insertion fails to overwrite by trying to insert a
    // bad value.
    auto Result = Trie.insert(Lookup, TrieType::value_type(hash(N), N - 1));
    EXPECT_EQ(N, Result->Data);
  }

  // Check the trie so we can confirm the structure is correct. Each subtrie
  // should have 2 slots. The root's index=0 should have the content for
  // 0x37 directly, and index=1 should be a linked-list of subtries, finally
  // ending with content for (max-2) and (max-3).
  //
  // Note: This structure is not exhaustive (too expensive to update tests),
  // but it does test that the dump format is somewhat readable and that the
  // basic structure is correct.
  //
  // Note: This test requires that the trie reads bytes starting from index 0
  // of the array of uint8_t, and then reads each byte's bits from high to low.

  // Check the Trie.
  // We should allocated a total of 64 SubTries for 64 bit hash.
  ASSERT_EQ(getNumTries(), 64u);
  // Check the root trie. Two slots and both are used.
  ASSERT_EQ(getNumSlotUsed(getRoot()), 2u);
  // Check last subtrie.
  // Last allocated trie is the next node in the allocation chain.
  auto LastAlloctedSubTrie = getNextTrie(getRoot());
  ASSERT_EQ(getTriePrefixAsString(LastAlloctedSubTrie),
            "ffffffffffffff[1111110]");
  ASSERT_EQ(getStartBit(LastAlloctedSubTrie), 63u);
  ASSERT_EQ(getNumBits(LastAlloctedSubTrie), 1u);
  ASSERT_EQ(getNumSlotUsed(LastAlloctedSubTrie), 2u);
}

TEST_F(SmallNodeTrieTest, TrieStructureSmallFinalSubtrie) {
  NumType Numbers[] = {
      // Three numbers that will nest deeply to test (1) sinking subtries and
      // (2) deep, non-trivial hints.
      std::numeric_limits<NumType>::max(),
      std::numeric_limits<NumType>::max() - 2u,
      std::numeric_limits<NumType>::max() - 3u,
      // One number to stay at the top-level.
      0x37,
  };

  // Use subtrie size of 5 to avoid hitting 64 evenly, making the final subtrie
  // small.
  auto &Trie = createTrie(/*RootBits=*/8, /*SubtrieBits=*/5);

  for (NumType N : Numbers) {
    // Lookup first to exercise hint code for deep tries.
    TrieType::pointer Lookup = Trie.find(hash(N));
    EXPECT_FALSE(Lookup);

    Trie.insert(Lookup, TrieType::value_type(hash(N), N));
  }
  for (NumType N : Numbers) {
    TrieType::pointer Lookup = Trie.find(hash(N));
    ASSERT_TRUE(Lookup);
    EXPECT_EQ(hash(N), Lookup->Hash);
    EXPECT_EQ(N, Lookup->Data);

    // Confirm a subsequent insertion fails to overwrite by trying to insert a
    // bad value.
    auto Result = Trie.insert(Lookup, TrieType::value_type(hash(N), N - 1));
    EXPECT_EQ(N, Result->Data);
  }

  // Check the trie so we can confirm the structure is correct. The root
  // should have 2^8=256 slots, most subtries should have 2^5=32 slots, and the
  // deepest subtrie should have 2^1=2 slots (since (64-8)mod(5)=1).
  // should have 2 slots. The root's index=0 should have the content for
  // 0x37 directly, and index=1 should be a linked-list of subtries, finally
  // ending with content for (max-2) and (max-3).
  //
  // Note: This structure is not exhaustive (too expensive to update tests),
  // but it does test that the dump format is somewhat readable and that the
  // basic structure is correct.
  //
  // Note: This test requires that the trie reads bytes starting from index 0
  // of the array of uint8_t, and then reads each byte's bits from high to low.

  // Check the Trie.
  // 64 bit hash = 8 + 5 * 11 + 1, so 1 root, 11 8bit subtrie and 1 last level
  // subtrie, 13 total.
  ASSERT_EQ(getNumTries(), 13u);
  // Check the root trie. Two slots and both are used.
  ASSERT_EQ(getNumSlotUsed(getRoot()), 2u);
  // Check last subtrie.
  // Last allocated trie is the next node in the allocation chain.
  auto LastAlloctedSubTrie = getNextTrie(getRoot());
  ASSERT_EQ(getTriePrefixAsString(LastAlloctedSubTrie),
            "ffffffffffffff[1111110]");
  ASSERT_EQ(getStartBit(LastAlloctedSubTrie), 63u);
  ASSERT_EQ(getNumBits(LastAlloctedSubTrie), 1u);
  ASSERT_EQ(getNumSlotUsed(LastAlloctedSubTrie), 2u);
}

TEST_F(SmallNodeTrieTest, TrieDestructionLoop) {
  // Test destroying large Trie. Make sure there is no recursion that can
  // overflow the stack.

  // Limit the tries to 2 slots (1 bit) to generate subtries at a higher rate.
  auto &Trie = createTrie(/*NumRootBits=*/1, /*NumSubtrieBits=*/1);

  // Fill them up. Pick a MaxN high enough to cause a stack overflow in debug
  // builds.
  static constexpr uint64_t MaxN = 100000;
  for (uint64_t N = 0; N != MaxN; ++N) {
    HashType Hash = hash(N);
    Trie.insert(TrieType::pointer(), TrieType::value_type(Hash, NumType{N}));
  }

  // Destroy tries. If destruction is recursive and MaxN is high enough, these
  // will both fail.
  destroyTrie();
}

struct NumWithDestructorT {
  uint64_t Num;
  llvm::function_ref<void()> DestructorCallback;
  ~NumWithDestructorT() { DestructorCallback(); }
};

using NodeWithDestructorTrieTest = SimpleTrieHashMapTest<NumWithDestructorT>;

TEST_F(NodeWithDestructorTrieTest, TrieDestructionLoop) {
  // Test destroying large Trie. Make sure there is no recursion that can
  // overflow the stack.

  // Limit the tries to 2 slots (1 bit) to generate subtries at a higher rate.
  auto &Trie = createTrie(/*NumRootBits=*/1, /*NumSubtrieBits=*/1);

  // Fill them up. Pick a MaxN high enough to cause a stack overflow in debug
  // builds.
  static constexpr uint64_t MaxN = 100000;

  uint64_t DestructorCalled = 0;
  auto DtorCallback = [&DestructorCalled]() { ++DestructorCalled; };
  for (uint64_t N = 0; N != MaxN; ++N) {
    HashType Hash = hash(N);
    Trie.insert(TrieType::pointer(),
                TrieType::value_type(Hash, NumType{N, DtorCallback}));
  }
  // Reset the count after all the temporaries get destroyed.
  DestructorCalled = 0;

  // Destroy tries. If destruction is recursive and MaxN is high enough, these
  // will both fail.
  destroyTrie();

  // Count the number of destructor calls during `destroyTrie()`.
  ASSERT_EQ(DestructorCalled, MaxN);
}

using NumStrNodeTrieTest = SimpleTrieHashMapTest<std::string>;

TEST_F(NumStrNodeTrieTest, TrieInsertLazy) {
  for (unsigned RootBits : {2, 3, 6, 10}) {
    for (unsigned SubtrieBits : {2, 3, 4}) {
      auto &Trie = createTrie(RootBits, SubtrieBits);
      for (int I = 0, E = 1000; I != E; ++I) {
        TrieType::pointer Lookup;
        HashType H = hash(I);
        if (I & 1)
          Lookup = Trie.find(H);

        auto insertNum = [&](uint64_t Num) {
          std::string S = Twine(I).str();
          auto Hash = hash(Num);
          return Trie.insertLazy(
              Hash, [&](TrieType::LazyValueConstructor C) { C(std::move(S)); });
        };
        auto S1 = insertNum(I);
        // The address of the Data should be the same.
        EXPECT_EQ(&S1->Data, &insertNum(I)->Data);

        auto insertStr = [&](std::string S) {
          int Num = std::stoi(S);
          return insertNum(Num);
        };
        std::string S2 = S1->Data;
        // The address of the Data should be the same.
        EXPECT_EQ(&S1->Data, &insertStr(S2)->Data);
      }
      for (int I = 0, E = 1000; I != E; ++I) {
        std::string S = Twine(I).str();
        TrieType::pointer Lookup = Trie.find(hash(I));
        EXPECT_TRUE(Lookup);
        if (!Lookup)
          continue;
        EXPECT_EQ(S, Lookup->Data);
      }
    }
  }
}
} // end anonymous namespace
