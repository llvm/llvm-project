//===- HashTable.cpp - PDB Hash Table -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/HashTable.h"
#include "llvm/ADT/Optional.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>

using namespace llvm;
using namespace llvm::pdb;

namespace {
struct IdentityTraits {
  static uint32_t hash(uint32_t K, const HashTable &Ctx) { return K; }
  static uint32_t realKey(uint32_t K, const HashTable &Ctx) { return K; }
  static uint32_t lowerKey(uint32_t K, const HashTable &Ctx) { return K; }
};
} // namespace

HashTable::HashTable() : HashTable(8) {}

HashTable::HashTable(uint32_t Capacity) { Buckets.resize(Capacity); }

Error HashTable::load(BinaryStreamReader &Stream) {
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

Error HashTable::commit(BinaryStreamWriter &Writer) const {
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

HashTableIterator HashTable::find(uint32_t K) const {
  return find_as<IdentityTraits>(K, *this);
}

void HashTable::set(uint32_t K, uint32_t V) {
  set_as<IdentityTraits, uint32_t>(K, V, *this);
}

void HashTable::remove(uint32_t K) { remove_as<IdentityTraits>(K, *this); }

uint32_t HashTable::get(uint32_t K) {
  auto I = find(K);
  assert(I != end());
  return (*I).second;
}

uint32_t HashTable::maxLoad(uint32_t capacity) { return capacity * 2 / 3 + 1; }

Error HashTable::readSparseBitVector(BinaryStreamReader &Stream,
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

Error HashTable::writeSparseBitVector(BinaryStreamWriter &Writer,
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
