//===- llvm/unittest/DebugInfo/PDB/HashTableTest.cpp ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ErrorChecking.h"
#include "gtest/gtest.h"

#include "llvm/DebugInfo/MSF/BinaryByteStream.h"
#include "llvm/DebugInfo/MSF/BinaryStreamReader.h"
#include "llvm/DebugInfo/MSF/BinaryStreamWriter.h"
#include "llvm/DebugInfo/PDB/Native/HashTable.h"

#include <vector>

using namespace llvm;
using namespace llvm::pdb;

namespace {
class HashTableInternals : public HashTable {
public:
  using HashTable::Buckets;
  using HashTable::Present;
  using HashTable::Deleted;
};
}

TEST(HashTableTest, TestSimple) {
  HashTable Table;
  EXPECT_EQ(0u, Table.size());
  EXPECT_GT(Table.capacity(), 0u);

  Table.set(3, 7);
  EXPECT_EQ(1u, Table.size());
  ASSERT_NE(Table.end(), Table.find(3));
  EXPECT_EQ(7u, Table.get(3));
}

TEST(HashTableTest, TestCollision) {
  HashTable Table;
  EXPECT_EQ(0u, Table.size());
  EXPECT_GT(Table.capacity(), 0u);

  // We use knowledge of the hash table's implementation details to make sure
  // to add another value that is the equivalent to the first value modulo the
  // hash table's capacity.
  uint32_t N1 = Table.capacity() + 1;
  uint32_t N2 = 2 * N1;

  Table.set(N1, 7);
  Table.set(N2, 12);
  EXPECT_EQ(2u, Table.size());
  ASSERT_NE(Table.end(), Table.find(N1));
  ASSERT_NE(Table.end(), Table.find(N2));

  EXPECT_EQ(7u, Table.get(N1));
  EXPECT_EQ(12u, Table.get(N2));
}

TEST(HashTableTest, TestRemove) {
  HashTable Table;
  EXPECT_EQ(0u, Table.size());
  EXPECT_GT(Table.capacity(), 0u);

  Table.set(1, 2);
  Table.set(3, 4);
  EXPECT_EQ(2u, Table.size());
  ASSERT_NE(Table.end(), Table.find(1));
  ASSERT_NE(Table.end(), Table.find(3));

  EXPECT_EQ(2u, Table.get(1));
  EXPECT_EQ(4u, Table.get(3));

  Table.remove(1u);
  EXPECT_EQ(1u, Table.size());
  EXPECT_EQ(Table.end(), Table.find(1));
  ASSERT_NE(Table.end(), Table.find(3));
  EXPECT_EQ(4u, Table.get(3));
}

TEST(HashTableTest, TestCollisionAfterMultipleProbes) {
  HashTable Table;
  EXPECT_EQ(0u, Table.size());
  EXPECT_GT(Table.capacity(), 0u);

  // Probing looks for the first available slot.  A slot may already be filled
  // as a result of an item with a *different* hash value already being there.
  // Test that when this happens, the probe still finds the value.
  uint32_t N1 = Table.capacity() + 1;
  uint32_t N2 = N1 + 1;
  uint32_t N3 = 2 * N1;

  Table.set(N1, 7);
  Table.set(N2, 11);
  Table.set(N3, 13);
  EXPECT_EQ(3u, Table.size());
  ASSERT_NE(Table.end(), Table.find(N1));
  ASSERT_NE(Table.end(), Table.find(N2));
  ASSERT_NE(Table.end(), Table.find(N3));

  EXPECT_EQ(7u, Table.get(N1));
  EXPECT_EQ(11u, Table.get(N2));
  EXPECT_EQ(13u, Table.get(N3));

  // Remove the one that had been filled in the middle, then insert another one
  // with a collision.  It should fill the newly emptied slot.
  Table.remove(N2);
  uint32_t N4 = N1 * 3;
  Table.set(N4, 17);
  EXPECT_EQ(3u, Table.size());
  ASSERT_NE(Table.end(), Table.find(N1));
  ASSERT_NE(Table.end(), Table.find(N3));
  ASSERT_NE(Table.end(), Table.find(N4));

  EXPECT_EQ(7u, Table.get(N1));
  EXPECT_EQ(13u, Table.get(N3));
  EXPECT_EQ(17u, Table.get(N4));
}

TEST(HashTableTest, Grow) {
  // So that we are independent of the load factor, `capacity` items, which is
  // guaranteed to trigger a grow.  Then verify that the size is the same, the
  // capacity is larger, and all the original items are still in the table.

  HashTable Table;
  uint32_t OldCapacity = Table.capacity();
  for (uint32_t I = 0; I < OldCapacity; ++I) {
    Table.set(OldCapacity + I * 2 + 1, I * 2 + 3);
  }
  EXPECT_EQ(OldCapacity, Table.size());
  EXPECT_GT(Table.capacity(), OldCapacity);
  for (uint32_t I = 0; I < OldCapacity; ++I) {
    ASSERT_NE(Table.end(), Table.find(OldCapacity + I * 2 + 1));
    EXPECT_EQ(I * 2 + 3, Table.get(OldCapacity + I * 2 + 1));
  }
}

TEST(HashTableTest, Serialization) {
  HashTableInternals Table;
  uint32_t Cap = Table.capacity();
  for (uint32_t I = 0; I < Cap; ++I) {
    Table.set(Cap + I * 2 + 1, I * 2 + 3);
  }

  std::vector<uint8_t> Buffer(Table.calculateSerializedLength());
  msf::MutableByteStream Stream(Buffer);
  msf::StreamWriter Writer(Stream);
  EXPECT_NO_ERROR(Table.commit(Writer));
  // We should have written precisely the number of bytes we calculated earlier.
  EXPECT_EQ(Buffer.size(), Writer.getOffset());

  HashTableInternals Table2;
  msf::StreamReader Reader(Stream);
  EXPECT_NO_ERROR(Table2.load(Reader));
  // We should have read precisely the number of bytes we calculated earlier.
  EXPECT_EQ(Buffer.size(), Reader.getOffset());

  EXPECT_EQ(Table.size(), Table2.size());
  EXPECT_EQ(Table.capacity(), Table2.capacity());
  EXPECT_EQ(Table.Buckets, Table2.Buckets);
  EXPECT_EQ(Table.Present, Table2.Present);
  EXPECT_EQ(Table.Deleted, Table2.Deleted);
}
