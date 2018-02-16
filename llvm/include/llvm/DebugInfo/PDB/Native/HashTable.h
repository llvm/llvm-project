//===- HashTable.h - PDB Hash Table -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_NATIVE_HASHTABLE_H
#define LLVM_DEBUGINFO_PDB_NATIVE_HASHTABLE_H

#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

namespace llvm {

class BinaryStreamReader;
class BinaryStreamWriter;

namespace pdb {

class HashTable;

class HashTableIterator
    : public iterator_facade_base<HashTableIterator, std::forward_iterator_tag,
                                  std::pair<uint32_t, uint32_t>> {
  friend HashTable;

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

class HashTable {
  friend class HashTableIterator;

  struct Header {
    support::ulittle32_t Size;
    support::ulittle32_t Capacity;
  };

  using BucketList = std::vector<std::pair<uint32_t, uint32_t>>;

public:
  HashTable();
  explicit HashTable(uint32_t Capacity);

  Error load(BinaryStreamReader &Stream);

  uint32_t calculateSerializedLength() const;
  Error commit(BinaryStreamWriter &Writer) const;

  void clear();

  uint32_t capacity() const;
  uint32_t size() const;

  HashTableIterator begin() const;
  HashTableIterator end() const;

  /// Find the entry with the specified key value.
  HashTableIterator find(uint32_t K) const;

  /// Find the entry whose key has the specified hash value, using the specified
  /// traits defining hash function and equality.
  template <typename Traits, typename Key, typename Context>
  HashTableIterator find_as(const Key &K, const Context &Ctx) const {
    uint32_t H = Traits::hash(K, Ctx) % capacity();
    uint32_t I = H;
    Optional<uint32_t> FirstUnused;
    do {
      if (isPresent(I)) {
        if (Traits::realKey(Buckets[I].first, Ctx) == K)
          return HashTableIterator(*this, I, false);
      } else {
        if (!FirstUnused)
          FirstUnused = I;
        // Insertion occurs via linear probing from the slot hint, and will be
        // inserted at the first empty / deleted location.  Therefore, if we are
        // probing and find a location that is neither present nor deleted, then
        // nothing must have EVER been inserted at this location, and thus it is
        // not possible for a matching value to occur later.
        if (!isDeleted(I))
          break;
      }
      I = (I + 1) % capacity();
    } while (I != H);

    // The only way FirstUnused would not be set is if every single entry in the
    // table were Present.  But this would violate the load factor constraints
    // that we impose, so it should never happen.
    assert(FirstUnused);
    return HashTableIterator(*this, *FirstUnused, true);
  }

  /// Set the entry with the specified key to the specified value.
  void set(uint32_t K, uint32_t V);

  /// Set the entry using a key type that the specified Traits can convert
  /// from a real key to an internal key.
  template <typename Traits, typename Key, typename Context>
  bool set_as(const Key &K, uint32_t V, Context &Ctx) {
    return set_as_internal<Traits, Key, Context>(K, V, None, Ctx);
  }

  void remove(uint32_t K);

  template <typename Traits, typename Key, typename Context>
  void remove_as(const Key &K, Context &Ctx) {
    auto Iter = find_as<Traits, Key, Context>(K, Ctx);
    // It wasn't here to begin with, just exit.
    if (Iter == end())
      return;

    assert(Present.test(Iter.index()));
    assert(!Deleted.test(Iter.index()));
    Deleted.set(Iter.index());
    Present.reset(Iter.index());
  }

  uint32_t get(uint32_t K);

protected:
  bool isPresent(uint32_t K) const { return Present.test(K); }
  bool isDeleted(uint32_t K) const { return Deleted.test(K); }

  BucketList Buckets;
  mutable SparseBitVector<> Present;
  mutable SparseBitVector<> Deleted;

private:
  /// Set the entry using a key type that the specified Traits can convert
  /// from a real key to an internal key.
  template <typename Traits, typename Key, typename Context>
  bool set_as_internal(const Key &K, uint32_t V, Optional<uint32_t> InternalKey,
                       Context &Ctx) {
    auto Entry = find_as<Traits, Key, Context>(K, Ctx);
    if (Entry != end()) {
      assert(isPresent(Entry.index()));
      assert(Traits::realKey(Buckets[Entry.index()].first, Ctx) == K);
      // We're updating, no need to do anything special.
      Buckets[Entry.index()].second = V;
      return false;
    }

    auto &B = Buckets[Entry.index()];
    assert(!isPresent(Entry.index()));
    assert(Entry.isEnd());
    B.first = InternalKey ? *InternalKey : Traits::lowerKey(K, Ctx);
    B.second = V;
    Present.set(Entry.index());
    Deleted.reset(Entry.index());

    grow<Traits, Key, Context>(Ctx);

    assert((find_as<Traits, Key, Context>(K, Ctx)) != end());
    return true;
  }

  static uint32_t maxLoad(uint32_t capacity);

  template <typename Traits, typename Key, typename Context>
  void grow(Context &Ctx) {
    uint32_t S = size();
    if (S < maxLoad(capacity()))
      return;
    assert(capacity() != UINT32_MAX && "Can't grow Hash table!");

    uint32_t NewCapacity =
        (capacity() <= INT32_MAX) ? capacity() * 2 : UINT32_MAX;

    // Growing requires rebuilding the table and re-hashing every item.  Make a
    // copy with a larger capacity, insert everything into the copy, then swap
    // it in.
    HashTable NewMap(NewCapacity);
    for (auto I : Present) {
      auto RealKey = Traits::realKey(Buckets[I].first, Ctx);
      NewMap.set_as_internal<Traits, Key, Context>(RealKey, Buckets[I].second,
                                                   Buckets[I].first, Ctx);
    }

    Buckets.swap(NewMap.Buckets);
    std::swap(Present, NewMap.Present);
    std::swap(Deleted, NewMap.Deleted);
    assert(capacity() == NewCapacity);
    assert(size() == S);
  }

  static Error readSparseBitVector(BinaryStreamReader &Stream,
                                   SparseBitVector<> &V);
  static Error writeSparseBitVector(BinaryStreamWriter &Writer,
                                    SparseBitVector<> &Vec);
};

} // end namespace pdb

} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_NATIVE_HASHTABLE_H
