//===- HashMappedTrieTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/HashMappedTrie.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/SHA1.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;

static StringRef takeNextLine(StringRef &Lines) {
  size_t Newline = Lines.find('\n');
  StringRef Line = Lines.take_front(Newline);
  Lines = Lines.drop_front(Newline + 1);
  return Line;
}

namespace {

TEST(HashMappedTrieTest, TrieStructure) {
  using NumType = uint64_t;
  using HashType = std::array<uint8_t, sizeof(NumType)>;
  using TrieType = ThreadSafeHashMappedTrie<NumType, sizeof(HashType)>;
  NumType Numbers[] = {
      // Three numbers that will nest deeply to test (1) sinking subtries and
      // (2) deep, non-trivial hints.
      std::numeric_limits<NumType>::max(),
      std::numeric_limits<NumType>::max() - 2u,
      std::numeric_limits<NumType>::max() - 3u,
      // One number to stay at the top-level.
      0x37,
  };

  // Use the number itself as hash to test the pathological case.
  auto hash = [](NumType Num) {
    NumType HashN = llvm::support::endian::byte_swap(Num, llvm::endianness::big);
    HashType Hash;
    memcpy(&Hash[0], &HashN, sizeof(HashType));
    return Hash;
  };

  // Use root and subtrie sizes of 1 so this gets sunk quite deep.
  TrieType Trie(1, 1);
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
    EXPECT_EQ(N,
              Trie.insert(Lookup, TrieType::value_type(hash(N), N - 1))->Data);
  }

  // Dump out the trie so we can confirm the structure is correct. Each subtrie
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
  SmallString<128> Dump;
  {
    raw_svector_ostream OS(Dump);
    Trie.print(OS);
  }

  // Check the header.
  StringRef DumpRef = Dump;
  ASSERT_EQ("root-bits=1 subtrie-bits=1", takeNextLine(DumpRef));

  // Check the root trie.
  ASSERT_EQ("root num-slots=2", takeNextLine(DumpRef));
  ASSERT_EQ("- index=0 content=[0000]000000000000037", takeNextLine(DumpRef));
  ASSERT_EQ("- index=1 subtrie=[1]", takeNextLine(DumpRef));
  ASSERT_EQ("subtrie=[1] num-slots=2", takeNextLine(DumpRef));

  // Check the last subtrie.
  size_t LastSubtrie = DumpRef.rfind("\nsubtrie=");
  ASSERT_NE(StringRef::npos, LastSubtrie);
  DumpRef = DumpRef.substr(LastSubtrie + 1);
  ASSERT_EQ("subtrie=fffffffffffffff[110] num-slots=2", takeNextLine(DumpRef));
  ASSERT_EQ("- index=0 content=fffffffffffffff[1100]", takeNextLine(DumpRef));
  ASSERT_EQ("- index=1 content=fffffffffffffff[1101]", takeNextLine(DumpRef));
  ASSERT_TRUE(DumpRef.empty());
}

TEST(HashMappedTrieTest, TrieStructureSmallFinalSubtrie) {
  using NumType = uint64_t;
  using HashType = std::array<uint8_t, sizeof(NumType)>;
  using TrieType = ThreadSafeHashMappedTrie<NumType, sizeof(HashType)>;
  NumType Numbers[] = {
      // Three numbers that will nest deeply to test (1) sinking subtries and
      // (2) deep, non-trivial hints.
      std::numeric_limits<NumType>::max(),
      std::numeric_limits<NumType>::max() - 2u,
      std::numeric_limits<NumType>::max() - 3u,
      // One number to stay at the top-level.
      0x37,
  };

  // Use the number itself as hash to test the pathological case.
  auto hash = [](NumType Num) {
    NumType HashN = llvm::support::endian::byte_swap(Num, llvm::endianness::big);
    HashType Hash;
    memcpy(&Hash[0], &HashN, sizeof(HashType));
    return Hash;
  };

  // Use subtrie size of 7 to avoid hitting 64 evenly, making the final subtrie
  // small.
  TrieType Trie(8, 5);
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
    EXPECT_EQ(N,
              Trie.insert(Lookup, TrieType::value_type(hash(N), N - 1))->Data);
  }

  // Dump out the trie so we can confirm the structure is correct. The root
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
  SmallString<128> Dump;
  {
    raw_svector_ostream OS(Dump);
    Trie.print(OS);
  }

  // Check the header.
  StringRef DumpRef = Dump;
  ASSERT_EQ("root-bits=8 subtrie-bits=5", takeNextLine(DumpRef));

  // Check the root trie.
  ASSERT_EQ("root num-slots=256", takeNextLine(DumpRef));
  ASSERT_EQ("- index=0 content=[00000000]00000000000037",
            takeNextLine(DumpRef));
  ASSERT_EQ("- index=255 subtrie=ff", takeNextLine(DumpRef));
  ASSERT_EQ("subtrie=ff num-slots=32", takeNextLine(DumpRef));

  // Check the last subtrie.
  size_t LastSubtrie = DumpRef.rfind("\nsubtrie=");
  ASSERT_NE(StringRef::npos, LastSubtrie);
  DumpRef = DumpRef.substr(LastSubtrie + 1);
  ASSERT_EQ("subtrie=fffffffffffffff[110] num-slots=2", takeNextLine(DumpRef));
  ASSERT_EQ("- index=0 content=fffffffffffffff[1100]", takeNextLine(DumpRef));
  ASSERT_EQ("- index=1 content=fffffffffffffff[1101]", takeNextLine(DumpRef));
  ASSERT_TRUE(DumpRef.empty());
}

TEST(HashMappedTrieTest, TrieDestructionLoop) {
  using NumT = uint64_t;
  struct NumWithDestructorT {
    NumT Num;
    operator NumT() const { return Num; }
    ~NumWithDestructorT() {}
  };

  using HashT = std::array<uint8_t, sizeof(NumT)>;
  using TrieT = ThreadSafeHashMappedTrie<NumT, sizeof(HashT)>;
  using TrieWithDestructorT =
      ThreadSafeHashMappedTrie<NumWithDestructorT, sizeof(HashT)>;

  // Use the number itself in big-endian order as the hash.
  auto hash = [](NumT Num) {
    NumT HashN = llvm::support::endian::byte_swap(Num, llvm::endianness::big);
    HashT Hash;
    memcpy(&Hash[0], &HashN, sizeof(HashT));
    return Hash;
  };

  // Use optionals to control when destructors are called.
  std::optional<TrieT> Trie;
  std::optional<TrieWithDestructorT> TrieWithDestructor;

  // Limit the tries to 2 slots (1 bit) to generate subtries at a higher rate.
  Trie.emplace(/*NumRootBits=*/1, /*NumSubtrieBits=*/1);
  TrieWithDestructor.emplace(/*NumRootBits=*/1, /*NumSubtrieBits=*/1);

  // Fill them up. Pick a MaxN high enough to cause a stack overflow in debug
  // builds.
  static constexpr uint64_t MaxN = 100000;
  for (uint64_t N = 0; N != MaxN; ++N) {
    HashT Hash = hash(N);
    Trie->insert(TrieT::pointer(), TrieT::value_type(Hash, N));
    TrieWithDestructor->insert(
        TrieWithDestructorT::pointer(),
        TrieWithDestructorT::value_type(Hash, NumWithDestructorT{N}));
  }

  // Destroy tries. If destruction is recursive and MaxN is high enough, these
  // will both fail.
  Trie.reset();
  TrieWithDestructor.reset();
}

namespace {
using HasherT = SHA1;
using HashType = decltype(HasherT::hash(std::declval<ArrayRef<uint8_t> &>()));
template <class T>
class ThreadSafeHashMappedTrieSet
    : ThreadSafeHashMappedTrie<T, sizeof(HashType)> {
public:
  using TrieType =
      typename ThreadSafeHashMappedTrieSet::ThreadSafeHashMappedTrie;
  using LazyValueConstructor = typename ThreadSafeHashMappedTrieSet::
      ThreadSafeHashMappedTrie::LazyValueConstructor;

  class pointer : public TrieType::const_pointer {
    using BaseType = typename TrieType::const_pointer;

  public:
    const T &operator*() const {
      return TrieType::const_pointer::operator*().Data;
    }
    const T *operator->() const { return &operator*(); }

    pointer() = default;
    pointer(pointer &&) = default;
    pointer(const pointer &) = default;
    pointer &operator=(pointer &&) = default;
    pointer &operator=(const pointer &) = default;

  private:
    pointer(BaseType Result) : BaseType(Result) {}
    friend class ThreadSafeHashMappedTrieSet;
  };

  ThreadSafeHashMappedTrieSet(
      std::optional<size_t> NumRootBits = std::nullopt,
      std::optional<size_t> NumSubtrieBits = std::nullopt)
      : TrieType(NumRootBits, NumSubtrieBits) {}

  static HashType hash(const T &V) {
    return HasherT::hash(ArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t *>(V.data()), V.size()));
  }
  pointer find(const T &Value) const {
    return pointer(TrieType::find(hash(Value)));
  }
  pointer insert(pointer Hint, T &&Value) {
    return pointer(TrieType::insertLazy(
        typename pointer::BaseType(Hint),
        [&](LazyValueConstructor C) { C(std::move(Value)); }));
  }
  pointer insert(pointer Hint, const T &Value) {
    return pointer(
        TrieType::insertLazy(typename pointer::BaseType(Hint), hash(Value),
                             [&](LazyValueConstructor C) { C(Value); }));
  }
  pointer insert(T &&Value) { return insert(pointer(), Value); }
  pointer insert(const T &Value) { return insert(pointer(), Value); }
};
} // end anonymous namespace

TEST(HashMappedTrieTest, Strings) {
  for (unsigned RootBits : {2, 3, 6, 10}) {
    for (unsigned SubtrieBits : {2, 3, 4}) {
      ThreadSafeHashMappedTrieSet<std::string> Strings(RootBits, SubtrieBits);
      const std::string &A1 = *Strings.insert("A");
      EXPECT_EQ(&A1, &*Strings.insert("A"));
      std::string A2 = A1;
      EXPECT_EQ(&A1, &*Strings.insert(A2));

      const std::string &B1 = *Strings.insert("B");
      EXPECT_EQ(&B1, &*Strings.insert(B1));
      std::string B2 = B1;
      EXPECT_EQ(&B1, &*Strings.insert(B2));

      for (int I = 0, E = 1000; I != E; ++I) {
        ThreadSafeHashMappedTrieSet<std::string>::pointer Lookup;
        std::string S = Twine(I).str();
        if (I & 1)
          Lookup = Strings.find(S);
        const std::string &S1 = *Strings.insert(Lookup, S);
        EXPECT_EQ(&S1, &*Strings.insert(S1));
        std::string S2 = S1;
        EXPECT_EQ(&S1, &*Strings.insert(S2));
      }
      for (int I = 0, E = 1000; I != E; ++I) {
        std::string S = Twine(I).str();
        ThreadSafeHashMappedTrieSet<std::string>::pointer Lookup =
            Strings.find(S);
        EXPECT_TRUE(Lookup);
        if (!Lookup)
          continue;
        EXPECT_EQ(S, *Lookup);
      }
    }
  }
}

} // namespace
