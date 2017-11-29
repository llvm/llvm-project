//===- TypeTableBuilder.h ----------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPETABLEBUILDER_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPETABLEBUILDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/RecordSerialization.h"
#include "llvm/DebugInfo/CodeView/SimpleTypeSerializer.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeRecordMapping.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/Error.h"
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

namespace llvm {
namespace codeview {

class ContinuationRecordBuilder;
class TypeHasher;

class TypeTableBuilder {

  BumpPtrAllocator &RecordStorage;
  SimpleTypeSerializer SimpleSerializer;

  /// Private type record hashing implementation details are handled here.
  std::unique_ptr<TypeHasher> Hasher;

  /// Contains a list of all records indexed by TypeIndex.toArrayIndex().
  SmallVector<ArrayRef<uint8_t>, 2> SeenRecords;

  /// Temporary storage that we use to copy a record's data while re-writing
  /// its type indices.
  SmallVector<uint8_t, 256> RemapStorage;

public:
  explicit TypeTableBuilder(BumpPtrAllocator &Storage, bool Hash = true);
  ~TypeTableBuilder();

  void reset();

  bool empty() const { return SeenRecords.empty(); }

  TypeIndex nextTypeIndex() const;

  BumpPtrAllocator &getAllocator() { return RecordStorage; }

  ArrayRef<ArrayRef<uint8_t>> records() const;
  TypeIndex insertRecordBytes(ArrayRef<uint8_t> &Record);
  TypeIndex insertRecord(const RemappedType &Record);
  TypeIndex insertRecord(ContinuationRecordBuilder &Builder);

  template <typename T> TypeIndex writeLeafType(T &Record) {
    ArrayRef<uint8_t> Data = SimpleSerializer.serialize(Record);
    return insertRecordBytes(Data);
  }

  template <typename TFunc> void ForEachRecord(TFunc Func) {
    uint32_t Index = TypeIndex::FirstNonSimpleIndex;

    for (auto Record : SeenRecords) {
      Func(TypeIndex(Index), Record);
      ++Index;
    }
  }
};

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_TYPETABLEBUILDER_H
