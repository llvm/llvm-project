//===- TypeDatabase.cpp --------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/TypeDatabase.h"

using namespace llvm;
using namespace llvm::codeview;

TypeDatabase::TypeDatabase(uint32_t Capacity) : TypeNameStorage(Allocator) {
  CVUDTNames.resize(Capacity);
  TypeRecords.resize(Capacity);
  ValidRecords.resize(Capacity);
}

TypeIndex TypeDatabase::appendType(StringRef Name, const CVType &Data) {
  LargestTypeIndex = getAppendIndex();
  if (LargestTypeIndex.toArrayIndex() >= capacity())
    grow();
  recordType(Name, LargestTypeIndex, Data);
  return LargestTypeIndex;
}

void TypeDatabase::recordType(StringRef Name, TypeIndex Index,
                              const CVType &Data) {
  LargestTypeIndex = empty() ? Index : std::max(Index, LargestTypeIndex);

  if (LargestTypeIndex.toArrayIndex() >= capacity())
    grow(Index);

  uint32_t AI = Index.toArrayIndex();

  assert(!contains(Index));
  assert(AI < capacity());

  CVUDTNames[AI] = Name;
  TypeRecords[AI] = Data;
  ValidRecords.set(AI);
  ++Count;
}

/// Saves the name in a StringSet and creates a stable StringRef.
StringRef TypeDatabase::saveTypeName(StringRef TypeName) {
  return TypeNameStorage.save(TypeName);
}

StringRef TypeDatabase::getTypeName(TypeIndex Index) const {
  if (Index.isNoneType() || Index.isSimple())
    return TypeIndex::simpleTypeName(Index);

  if (contains(Index))
    return CVUDTNames[Index.toArrayIndex()];

  return "<unknown UDT>";
}

const CVType &TypeDatabase::getTypeRecord(TypeIndex Index) const {
  assert(contains(Index));
  return TypeRecords[Index.toArrayIndex()];
}

CVType &TypeDatabase::getTypeRecord(TypeIndex Index) {
  assert(contains(Index));
  return TypeRecords[Index.toArrayIndex()];
}

bool TypeDatabase::contains(TypeIndex Index) const {
  uint32_t AI = Index.toArrayIndex();
  if (AI >= capacity())
    return false;

  return ValidRecords.test(AI);
}

uint32_t TypeDatabase::size() const { return Count; }

uint32_t TypeDatabase::capacity() const { return TypeRecords.size(); }

CVType TypeDatabase::getType(TypeIndex Index) { return getTypeRecord(Index); }

StringRef TypeDatabase::getTypeName(TypeIndex Index) {
  return static_cast<const TypeDatabase *>(this)->getTypeName(Index);
}

bool TypeDatabase::contains(TypeIndex Index) {
  return static_cast<const TypeDatabase *>(this)->contains(Index);
}

uint32_t TypeDatabase::size() {
  return static_cast<const TypeDatabase *>(this)->size();
}

uint32_t TypeDatabase::capacity() {
  return static_cast<const TypeDatabase *>(this)->capacity();
}

void TypeDatabase::grow() { grow(LargestTypeIndex + 1); }

void TypeDatabase::grow(TypeIndex NewIndex) {
  uint32_t NewSize = NewIndex.toArrayIndex() + 1;

  if (NewSize <= capacity())
    return;

  uint32_t NewCapacity = NewSize * 3 / 2;

  TypeRecords.resize(NewCapacity);
  CVUDTNames.resize(NewCapacity);
  ValidRecords.resize(NewCapacity);
}

bool TypeDatabase::empty() const { return size() == 0; }

Optional<TypeIndex> TypeDatabase::largestTypeIndexLessThan(TypeIndex TI) const {
  uint32_t AI = TI.toArrayIndex();
  int N = ValidRecords.find_prev(AI);
  if (N == -1)
    return None;
  return TypeIndex::fromArrayIndex(N);
}

TypeIndex TypeDatabase::getAppendIndex() const {
  if (empty())
    return TypeIndex::fromArrayIndex(0);

  return LargestTypeIndex + 1;
}

Optional<TypeIndex> TypeDatabase::getFirst() {
  int N = ValidRecords.find_first();
  if (N == -1)
    return None;
  return TypeIndex::fromArrayIndex(N);
}

Optional<TypeIndex> TypeDatabase::getNext(TypeIndex Prev) {
  int N = ValidRecords.find_next(Prev.toArrayIndex());
  if (N == -1)
    return None;
  return TypeIndex::fromArrayIndex(N);
}
