//===- unittests/Support/OnDiskHashTableTest.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/OnDiskHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <type_traits>

using namespace llvm;

namespace {

// Common types used in the tests.
using KeyType = std::string;
using ValueType = std::string;

// Shared reader trait for verifying the written tables.
struct TestLookupTrait {
  using data_type = ValueType;
  using internal_key_type = KeyType;
  using external_key_type = KeyType;
  using hash_value_type = uint32_t;
  using offset_type = uint32_t;

  static bool EqualKey(const internal_key_type &A, const internal_key_type &B) {
    return A == B;
  }
  static const internal_key_type &GetInternalKey(const external_key_type &K) {
    return K;
  }
  static const external_key_type &GetExternalKey(const internal_key_type &K) {
    return K;
  }

  static hash_value_type ComputeHash(const internal_key_type &K) {
    // Simple hash function for testing.
    hash_value_type Hash = 0;
    for (char C : K)
      Hash = Hash * 33 + C;
    return Hash;
  }

  static std::pair<offset_type, offset_type>
  ReadKeyDataLength(const unsigned char *&D) {
    using namespace llvm::support;
    offset_type KeyLen =
        endian::readNext<offset_type, llvm::endianness::little, unaligned>(D);
    offset_type DataLen =
        endian::readNext<offset_type, llvm::endianness::little, unaligned>(D);
    return {KeyLen, DataLen};
  }

  internal_key_type ReadKey(const unsigned char *D, offset_type KeyLen) {
    return std::string(reinterpret_cast<const char *>(D), KeyLen);
  }

  data_type ReadData(const internal_key_type &, const unsigned char *D,
                     offset_type DataLen) {
    return std::string(reinterpret_cast<const char *>(D), DataLen);
  }
};

// Unified/Split Templated Writer Trait.
template <bool IsUnified> struct TestWriterTrait : public TestLookupTrait {
  using key_type = KeyType;
  using key_type_ref = const KeyType &;
  using data_type = ValueType;
  using data_type_ref = const ValueType &;

  // Split (3-step) emission. Enabled only when IsUnified is false.
  template <bool B = IsUnified, std::enable_if_t<!B, int> = 0>
  static std::pair<offset_type, offset_type>
  EmitKeyDataLength(raw_ostream &Out, key_type_ref K, data_type_ref V) {
    using namespace llvm::support;
    endian::Writer LE(Out, llvm::endianness::little);
    offset_type KeyLen = K.size();
    offset_type DataLen = V.size();
    LE.write<offset_type>(KeyLen);
    LE.write<offset_type>(DataLen);
    return {KeyLen, DataLen};
  }

  template <bool B = IsUnified, std::enable_if_t<!B, int> = 0>
  static void EmitKey(raw_ostream &Out, key_type_ref K, offset_type) {
    Out << K;
  }

  template <bool B = IsUnified, std::enable_if_t<!B, int> = 0>
  static void EmitData(raw_ostream &Out, key_type_ref, data_type_ref V,
                       offset_type) {
    Out << V;
  }

  // Unified (KeyValuePair) emission. Enabled only when IsUnified is true.
  template <bool B = IsUnified, std::enable_if_t<B, int> = 0>
  static void EmitKeyValuePair(raw_ostream &Out, key_type_ref K,
                               data_type_ref V) {
    using namespace llvm::support;
    endian::Writer LE(Out, llvm::endianness::little);
    offset_type KeyLen = K.size();
    offset_type DataLen = V.size();
    LE.write<offset_type>(KeyLen);
    LE.write<offset_type>(DataLen);
    Out << K;
    Out << V;
  }
};

template <typename Trait>
static SmallVector<char, 1024>
WriteTable(const std::vector<std::pair<KeyType, ValueType>> &Data) {
  SmallVector<char, 1024> Buf;
  raw_svector_ostream OS(Buf);
  OS << "PAD"; // Pad with some bytes because Emit requires non-zero offset.
  OnDiskChainedHashTableGenerator<Trait> Generator;
  for (const auto &[Key, Value] : Data)
    Generator.insert(Key, Value);
  uint32_t TableOffset = Generator.Emit(OS);

  using namespace llvm::support;
  endian::Writer LE(OS, llvm::endianness::little);
  LE.write<uint32_t>(TableOffset);
  return Buf;
}

TEST(OnDiskHashTableTest, SplitVsUnified) {
  // Mock data to write.
  std::vector<std::pair<KeyType, ValueType>> Data = {{"apple", "red"},
                                                     {"banana", "yellow"},
                                                     {"grape", "purple"},
                                                     {"orange", "orange"}};

  // Write using Split Trait (IsUnified = false).
  SmallVector<char, 1024> SplitBuf = WriteTable<TestWriterTrait<false>>(Data);

  // Write using Unified Trait (IsUnified = true).
  SmallVector<char, 1024> UnifiedBuf = WriteTable<TestWriterTrait<true>>(Data);

  // Assert that the generated binary data is identical.
  ASSERT_EQ(SplitBuf, UnifiedBuf);

  // Verify the table layout (verifying SplitBuf is sufficient since they are
  // identical).
  const unsigned char *Base =
      reinterpret_cast<const unsigned char *>(SplitBuf.data());
  const unsigned char *Ptr = Base + SplitBuf.size() - sizeof(uint32_t);
  using namespace llvm::support;
  uint32_t TableOffset =
      endian::readNext<uint32_t, llvm::endianness::little, unaligned>(Ptr);

  const unsigned char *Buckets = Base + TableOffset;

  std::unique_ptr<OnDiskChainedHashTable<TestLookupTrait>> Table(
      OnDiskChainedHashTable<TestLookupTrait>::Create(Buckets, Base));

  // Verify lookups.
  {
    auto It = Table->find("apple");
    ASSERT_NE(It, Table->end());
    EXPECT_EQ(*It, "red");
  }
  {
    auto It = Table->find("banana");
    ASSERT_NE(It, Table->end());
    EXPECT_EQ(*It, "yellow");
  }
  {
    auto It = Table->find("grape");
    ASSERT_NE(It, Table->end());
    EXPECT_EQ(*It, "purple");
  }
  {
    auto It = Table->find("orange");
    ASSERT_NE(It, Table->end());
    EXPECT_EQ(*It, "orange");
  }

  EXPECT_EQ(Table->find("pear"), Table->end());
}

} // namespace
