//===- HashTable.cpp - PDB Hash Table ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/HashTable.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"

#include <assert.h>

using namespace llvm;
using namespace llvm::pdb;

HashTable::HashTable() : HashTable(8) {}

HashTable::HashTable(uint32_t Capacity) { Buckets.resize(Capacity); }

Error HashTable::load(msf::StreamReader &Stream) {
  const Header *H;
  if (auto EC = Stream.readObject(H))
    return EC;
  if (H->Capacity == 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Invalid Hash Table Capacity");
  if (H->Size > maxLoad(H->Capacity))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Invalid Hash Table Size");

  Buckets.resize(H->Capacity);

  if (auto EC = readSparseBitVector(Stream, Present))
    return EC;
  if (Present.count() != H->Size)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Present bit vector does not match size!");

  if (auto EC = readSparseBitVector(Stream, Deleted))
    return EC;
  if (Present.intersects(Deleted))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Present bit vector interesects deleted!");

  for (uint32_t P : Present) {
    if (auto EC = Stream.readInteger(Buckets[P].first))
      return EC;
    if (auto EC = Stream.readInteger(Buckets[P].second))
      return EC;
  }

  return Error::success();
}

uint32_t HashTable::calculateSerializedLength() const {
  uint32_t Size = sizeof(Header);

  int NumBitsP = Present.find_last() + 1;
  int NumBitsD = Deleted.find_last() + 1;

  // Present bit set number of words, followed by that many actual words.
  Size += sizeof(uint32_t);
  Size += alignTo(NumBitsP, sizeof(uint32_t));

  // Deleted bit set number of words, followed by that many actual words.
  Size += sizeof(uint32_t);
  Size += alignTo(NumBitsD, sizeof(uint32_t));

  // One (Key, Value) pair for each entry Present.
  Size += 2 * sizeof(uint32_t) * size();

  return Size;
}

Error HashTable::commit(msf::StreamWriter &Writer) const {
  Header H;
  H.Size = size();
  H.Capacity = capacity();
  if (auto EC = Writer.writeObject(H))
    return EC;

  if (auto EC = writeSparseBitVector(Writer, Present))
    return EC;

  if (auto EC = writeSparseBitVector(Writer, Deleted))
    return EC;

  for (const auto &Entry : *this) {
    if (auto EC = Writer.writeInteger(Entry.first))
      return EC;
    if (auto EC = Writer.writeInteger(Entry.second))
      return EC;
  }
  return Error::success();
}

void HashTable::clear() {
  Buckets.resize(8);
  Present.clear();
  Deleted.clear();
}

uint32_t HashTable::capacity() const { return Buckets.size(); }
uint32_t HashTable::size() const { return Present.count(); }

HashTableIterator HashTable::begin() const { return HashTableIterator(*this); }
HashTableIterator HashTable::end() const {
  return HashTableIterator(*this, 0, true);
}

HashTableIterator HashTable::find(uint32_t K) {
  uint32_t H = K % capacity();
  uint32_t I = H;
  Optional<uint32_t> FirstUnused;
  do {
    if (isPresent(I)) {
      if (Buckets[I].first == K)
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

void HashTable::set(uint32_t K, uint32_t V) {
  auto Entry = find(K);
  if (Entry != end()) {
    assert(isPresent(Entry.index()));
    assert(Buckets[Entry.index()].first == K);
    // We're updating, no need to do anything special.
    Buckets[Entry.index()].second = V;
    return;
  }

  auto &B = Buckets[Entry.index()];
  assert(!isPresent(Entry.index()));
  assert(Entry.isEnd());
  B.first = K;
  B.second = V;
  Present.set(Entry.index());
  Deleted.reset(Entry.index());

  grow();

  assert(find(K) != end());
}

void HashTable::remove(uint32_t K) {
  auto Iter = find(K);
  // It wasn't here to begin with, just exit.
  if (Iter == end())
    return;

  assert(Present.test(Iter.index()));
  assert(!Deleted.test(Iter.index()));
  Deleted.set(Iter.index());
  Present.reset(Iter.index());
}

uint32_t HashTable::get(uint32_t K) {
  auto I = find(K);
  assert(I != end());
  return (*I).second;
}

uint32_t HashTable::maxLoad(uint32_t capacity) { return capacity * 2 / 3 + 1; }

void HashTable::grow() {
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
    NewMap.set(Buckets[I].first, Buckets[I].second);
  }

  Buckets.swap(NewMap.Buckets);
  std::swap(Present, NewMap.Present);
  std::swap(Deleted, NewMap.Deleted);
  assert(capacity() == NewCapacity);
  assert(size() == S);
}

Error HashTable::readSparseBitVector(msf::StreamReader &Stream,
                                     SparseBitVector<> &V) {
  uint32_t NumWords;
  if (auto EC = Stream.readInteger(NumWords))
    return joinErrors(
        std::move(EC),
        make_error<RawError>(raw_error_code::corrupt_file,
                             "Expected hash table number of words"));

  for (uint32_t I = 0; I != NumWords; ++I) {
    uint32_t Word;
    if (auto EC = Stream.readInteger(Word))
      return joinErrors(std::move(EC),
                        make_error<RawError>(raw_error_code::corrupt_file,
                                             "Expected hash table word"));
    for (unsigned Idx = 0; Idx < 32; ++Idx)
      if (Word & (1U << Idx))
        V.set((I * 32) + Idx);
  }
  return Error::success();
}

Error HashTable::writeSparseBitVector(msf::StreamWriter &Writer,
                                      SparseBitVector<> &Vec) {
  int ReqBits = Vec.find_last() + 1;
  uint32_t NumWords = alignTo(ReqBits, sizeof(uint32_t)) / sizeof(uint32_t);
  if (auto EC = Writer.writeInteger(NumWords))
    return joinErrors(
        std::move(EC),
        make_error<RawError>(raw_error_code::corrupt_file,
                             "Could not write linear map number of words"));

  uint32_t Idx = 0;
  for (uint32_t I = 0; I != NumWords; ++I) {
    uint32_t Word = 0;
    for (uint32_t WordIdx = 0; WordIdx < 32; ++WordIdx, ++Idx) {
      if (Vec.test(Idx))
        Word |= (1 << WordIdx);
    }
    if (auto EC = Writer.writeInteger(Word))
      return joinErrors(std::move(EC), make_error<RawError>(
                                           raw_error_code::corrupt_file,
                                           "Could not write linear map word"));
  }
  return Error::success();
}

HashTableIterator::HashTableIterator(const HashTable &Map, uint32_t Index,
                                     bool IsEnd)
    : Map(&Map), Index(Index), IsEnd(IsEnd) {}

HashTableIterator::HashTableIterator(const HashTable &Map) : Map(&Map) {
  int I = Map.Present.find_first();
  if (I == -1) {
    Index = 0;
    IsEnd = true;
  } else {
    Index = static_cast<uint32_t>(I);
    IsEnd = false;
  }
}

HashTableIterator &HashTableIterator::operator=(const HashTableIterator &R) {
  Map = R.Map;
  return *this;
}

bool HashTableIterator::operator==(const HashTableIterator &R) const {
  if (IsEnd && R.IsEnd)
    return true;
  if (IsEnd != R.IsEnd)
    return false;

  return (Map == R.Map) && (Index == R.Index);
}

const std::pair<uint32_t, uint32_t> &HashTableIterator::operator*() const {
  assert(Map->Present.test(Index));
  return Map->Buckets[Index];
}

HashTableIterator &HashTableIterator::operator++() {
  while (Index < Map->Buckets.size()) {
    ++Index;
    if (Map->Present.test(Index))
      return *this;
  }

  IsEnd = true;
  return *this;
}
