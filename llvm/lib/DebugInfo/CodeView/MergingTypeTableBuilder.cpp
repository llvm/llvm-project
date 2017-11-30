//===- MergingTypeTableBuilder.cpp ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/MergingTypeTableBuilder.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/ContinuationRecordBuilder.h"
#include "llvm/DebugInfo/CodeView/RecordSerialization.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>

using namespace llvm;
using namespace llvm::codeview;

namespace {

struct HashedType {
  hash_code Hash;
  ArrayRef<uint8_t> Data;
  TypeIndex Index;
};

/// Wrapper around a poitner to a HashedType. Hash and equality operations are
/// based on data in the pointee.
struct HashedTypePtr {
  HashedTypePtr() = default;
  HashedTypePtr(HashedType *Ptr) : Ptr(Ptr) {}

  HashedType *Ptr = nullptr;
};

} // end anonymous namespace

namespace llvm {

template <> struct DenseMapInfo<HashedTypePtr> {
  static inline HashedTypePtr getEmptyKey() { return HashedTypePtr(nullptr); }

  static inline HashedTypePtr getTombstoneKey() {
    return HashedTypePtr(reinterpret_cast<HashedType *>(1));
  }

  static unsigned getHashValue(HashedTypePtr Val) {
    assert(Val.Ptr != getEmptyKey().Ptr && Val.Ptr != getTombstoneKey().Ptr);
    return Val.Ptr->Hash;
  }

  static bool isEqual(HashedTypePtr LHSP, HashedTypePtr RHSP) {
    HashedType *LHS = LHSP.Ptr;
    HashedType *RHS = RHSP.Ptr;
    if (RHS == getEmptyKey().Ptr || RHS == getTombstoneKey().Ptr)
      return LHS == RHS;
    if (LHS->Hash != RHS->Hash)
      return false;
    return LHS->Data == RHS->Data;
  }
};

} // end namespace llvm

/// Private implementation so that we don't leak our DenseMap instantiations to
/// users.
class llvm::codeview::TypeHasher {
private:
  /// Storage for type record provided by the caller. Records will outlive the
  /// hasher object, so they should be allocated here.
  BumpPtrAllocator &RecordStorage;

  /// Storage for hash keys. These only need to live as long as the hashing
  /// operation.
  BumpPtrAllocator KeyStorage;

  /// Hash table. We really want a DenseMap<ArrayRef<uint8_t>, TypeIndex> here,
  /// but DenseMap is inefficient when the keys are long (like type records)
  /// because it recomputes the hash value of every key when it grows. This
  /// value type stores the hash out of line in KeyStorage, so that table
  /// entries are small and easy to rehash.
  DenseSet<HashedTypePtr> HashedRecords;

public:
  TypeHasher(BumpPtrAllocator &RecordStorage) : RecordStorage(RecordStorage) {}

  void reset() { HashedRecords.clear(); }

  /// Takes the bytes of type record, inserts them into the hash table, saves
  /// them, and returns a pointer to an identical stable type record along with
  /// its type index in the destination stream.
  TypeIndex getOrCreateRecord(ArrayRef<uint8_t> &Record, TypeIndex TI);
};

TypeIndex TypeHasher::getOrCreateRecord(ArrayRef<uint8_t> &Record,
                                        TypeIndex TI) {
  assert(Record.size() < UINT32_MAX && "Record too big");
  assert(Record.size() % 4 == 0 && "Record is not aligned to 4 bytes!");

  // Compute the hash up front so we can store it in the key.
  HashedType TempHashedType = {hash_value(Record), Record, TI};
  auto Result = HashedRecords.insert(HashedTypePtr(&TempHashedType));
  HashedType *&Hashed = Result.first->Ptr;

  if (Result.second) {
    // This was a new type record. We need stable storage for both the key and
    // the record. The record should outlive the hashing operation.
    Hashed = KeyStorage.Allocate<HashedType>();
    *Hashed = TempHashedType;

    uint8_t *Stable = RecordStorage.Allocate<uint8_t>(Record.size());
    memcpy(Stable, Record.data(), Record.size());
    Hashed->Data = makeArrayRef(Stable, Record.size());
  }

  // Update the caller's copy of Record to point a stable copy.
  Record = Hashed->Data;
  return Hashed->Index;
}

TypeIndex MergingTypeTableBuilder::nextTypeIndex() const {
  return TypeIndex::fromArrayIndex(SeenRecords.size());
}

MergingTypeTableBuilder::MergingTypeTableBuilder(BumpPtrAllocator &Storage)
    : RecordStorage(Storage) {
  Hasher = llvm::make_unique<TypeHasher>(Storage);
}

MergingTypeTableBuilder::~MergingTypeTableBuilder() = default;

Optional<TypeIndex> MergingTypeTableBuilder::getFirst() {
  if (empty())
    return None;

  return TypeIndex(TypeIndex::FirstNonSimpleIndex);
}

Optional<TypeIndex> MergingTypeTableBuilder::getNext(TypeIndex Prev) {
  if (++Prev == nextTypeIndex())
    return None;
  return Prev;
}

CVType MergingTypeTableBuilder::getType(TypeIndex Index) {
  CVType Type;
  Type.RecordData = SeenRecords[Index.toArrayIndex()];
  const RecordPrefix *P =
      reinterpret_cast<const RecordPrefix *>(Type.RecordData.data());
  Type.Type = static_cast<TypeLeafKind>(uint16_t(P->RecordKind));
  return Type;
}

StringRef MergingTypeTableBuilder::getTypeName(TypeIndex Index) {
  llvm_unreachable("Method not implemented");
}

bool MergingTypeTableBuilder::contains(TypeIndex Index) {
  if (Index.isSimple() || Index.isNoneType())
    return false;

  return Index.toArrayIndex() < SeenRecords.size();
}

uint32_t MergingTypeTableBuilder::size() { return SeenRecords.size(); }

uint32_t MergingTypeTableBuilder::capacity() { return SeenRecords.size(); }

ArrayRef<ArrayRef<uint8_t>> MergingTypeTableBuilder::records() const {
  return SeenRecords;
}

void MergingTypeTableBuilder::reset() {
  Hasher->reset();
  SeenRecords.clear();
}

TypeIndex
MergingTypeTableBuilder::insertRecordBytes(ArrayRef<uint8_t> &Record) {
  TypeIndex ActualTI = Hasher->getOrCreateRecord(Record, nextTypeIndex());
  if (nextTypeIndex() == ActualTI)
    SeenRecords.push_back(Record);
  return ActualTI;
}

TypeIndex
MergingTypeTableBuilder::insertRecord(ContinuationRecordBuilder &Builder) {
  TypeIndex TI;
  auto Fragments = Builder.end(nextTypeIndex());
  assert(!Fragments.empty());
  for (auto C : Fragments)
    TI = insertRecordBytes(C.RecordData);
  return TI;
}
