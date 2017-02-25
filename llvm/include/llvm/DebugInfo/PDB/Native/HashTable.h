//===- HashTable.h - PDB Hash Table -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_HASHTABLE_H
#define LLVM_DEBUGINFO_PDB_RAW_HASHTABLE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/DebugInfo/MSF/BinaryStreamArray.h"
#include "llvm/DebugInfo/MSF/BinaryStreamReader.h"
#include "llvm/DebugInfo/MSF/BinaryStreamWriter.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <utility>

namespace llvm {
namespace pdb {

class HashTableIterator;

class HashTable {
  friend class HashTableIterator;
  struct Header {
    support::ulittle32_t Size;
    support::ulittle32_t Capacity;
  };

  typedef std::vector<std::pair<uint32_t, uint32_t>> BucketList;

public:
  HashTable();
  explicit HashTable(uint32_t Capacity);

  Error load(msf::StreamReader &Stream);

  uint32_t calculateSerializedLength() const;
  Error commit(msf::StreamWriter &Writer) const;

  void clear();

  uint32_t capacity() const;
  uint32_t size() const;

  HashTableIterator begin() const;
  HashTableIterator end() const;
  HashTableIterator find(uint32_t K);

  void set(uint32_t K, uint32_t V);
  void remove(uint32_t K);
  uint32_t get(uint32_t K);

protected:
  bool isPresent(uint32_t K) const { return Present.test(K); }
  bool isDeleted(uint32_t K) const { return Deleted.test(K); }
  BucketList Buckets;
  mutable SparseBitVector<> Present;
  mutable SparseBitVector<> Deleted;

private:
  static uint32_t maxLoad(uint32_t capacity);
  void grow();

  static Error readSparseBitVector(msf::StreamReader &Stream,
                                   SparseBitVector<> &V);
  static Error writeSparseBitVector(msf::StreamWriter &Writer,
                                    SparseBitVector<> &Vec);
};

class HashTableIterator
    : public iterator_facade_base<HashTableIterator, std::forward_iterator_tag,
                                  std::pair<uint32_t, uint32_t>> {
  friend class HashTable;
  HashTableIterator(const HashTable &Map, uint32_t Index, bool IsEnd);

public:
  HashTableIterator(const HashTable &Map);

  HashTableIterator &operator=(const HashTableIterator &R);
  bool operator==(const HashTableIterator &R) const;
  const std::pair<uint32_t, uint32_t> &operator*() const;
  HashTableIterator &operator++();

private:
  bool isEnd() const { return IsEnd; }
  uint32_t index() const { return Index; }

  const HashTable *Map;
  uint32_t Index;
  bool IsEnd;
};

} // end namespace pdb
} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_RAW_HASHTABLE_H
