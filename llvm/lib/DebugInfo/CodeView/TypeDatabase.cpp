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

namespace {
struct SimpleTypeEntry {
  StringRef Name;
  SimpleTypeKind Kind;
};
}

/// The names here all end in "*". If the simple type is a pointer type, we
/// return the whole name. Otherwise we lop off the last character in our
/// StringRef.
static const SimpleTypeEntry SimpleTypeNames[] = {
    {"void*", SimpleTypeKind::Void},
    {"<not translated>*", SimpleTypeKind::NotTranslated},
    {"HRESULT*", SimpleTypeKind::HResult},
    {"signed char*", SimpleTypeKind::SignedCharacter},
    {"unsigned char*", SimpleTypeKind::UnsignedCharacter},
    {"char*", SimpleTypeKind::NarrowCharacter},
    {"wchar_t*", SimpleTypeKind::WideCharacter},
    {"char16_t*", SimpleTypeKind::Character16},
    {"char32_t*", SimpleTypeKind::Character32},
    {"__int8*", SimpleTypeKind::SByte},
    {"unsigned __int8*", SimpleTypeKind::Byte},
    {"short*", SimpleTypeKind::Int16Short},
    {"unsigned short*", SimpleTypeKind::UInt16Short},
    {"__int16*", SimpleTypeKind::Int16},
    {"unsigned __int16*", SimpleTypeKind::UInt16},
    {"long*", SimpleTypeKind::Int32Long},
    {"unsigned long*", SimpleTypeKind::UInt32Long},
    {"int*", SimpleTypeKind::Int32},
    {"unsigned*", SimpleTypeKind::UInt32},
    {"__int64*", SimpleTypeKind::Int64Quad},
    {"unsigned __int64*", SimpleTypeKind::UInt64Quad},
    {"__int64*", SimpleTypeKind::Int64},
    {"unsigned __int64*", SimpleTypeKind::UInt64},
    {"__int128*", SimpleTypeKind::Int128},
    {"unsigned __int128*", SimpleTypeKind::UInt128},
    {"__half*", SimpleTypeKind::Float16},
    {"float*", SimpleTypeKind::Float32},
    {"float*", SimpleTypeKind::Float32PartialPrecision},
    {"__float48*", SimpleTypeKind::Float48},
    {"double*", SimpleTypeKind::Float64},
    {"long double*", SimpleTypeKind::Float80},
    {"__float128*", SimpleTypeKind::Float128},
    {"_Complex float*", SimpleTypeKind::Complex32},
    {"_Complex double*", SimpleTypeKind::Complex64},
    {"_Complex long double*", SimpleTypeKind::Complex80},
    {"_Complex __float128*", SimpleTypeKind::Complex128},
    {"bool*", SimpleTypeKind::Boolean8},
    {"__bool16*", SimpleTypeKind::Boolean16},
    {"__bool32*", SimpleTypeKind::Boolean32},
    {"__bool64*", SimpleTypeKind::Boolean64},
};

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
  if (Index.isNoneType())
    return "<no type>";

  if (Index.isSimple()) {
    // This is a simple type.
    for (const auto &SimpleTypeName : SimpleTypeNames) {
      if (SimpleTypeName.Kind == Index.getSimpleKind()) {
        if (Index.getSimpleMode() == SimpleTypeMode::Direct)
          return SimpleTypeName.Name.drop_back(1);
        // Otherwise, this is a pointer type. We gloss over the distinction
        // between near, far, 64, 32, etc, and just give a pointer type.
        return SimpleTypeName.Name;
      }
    }
    return "<unknown simple type>";
  }

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
