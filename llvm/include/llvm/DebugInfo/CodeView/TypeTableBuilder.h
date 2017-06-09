//===- TypeTableBuilder.h ---------------------------------------*- C++ -*-===//
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
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeSerializer.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <type_traits>

namespace llvm {
namespace codeview {

class TypeTableBuilder {
private:
  TypeIndex handleError(Error EC) const {
    assert(false && "Couldn't write Type!");
    consumeError(std::move(EC));
    return TypeIndex();
  }

  BumpPtrAllocator &Allocator;
  TypeSerializer Serializer;

public:
  explicit TypeTableBuilder(BumpPtrAllocator &Allocator,
                            bool WriteUnique = true)
      : Allocator(Allocator), Serializer(Allocator, WriteUnique) {}
  TypeTableBuilder(const TypeTableBuilder &) = delete;
  TypeTableBuilder &operator=(const TypeTableBuilder &) = delete;

  bool empty() const { return Serializer.records().empty(); }

  BumpPtrAllocator &getAllocator() const { return Allocator; }

  template <typename T> TypeIndex writeKnownType(T &Record) {
    static_assert(!std::is_same<T, FieldListRecord>::value,
                  "Can't serialize FieldList!");

    CVType Type;
    Type.Type = static_cast<TypeLeafKind>(Record.getKind());
    if (auto EC = Serializer.visitTypeBegin(Type))
      return handleError(std::move(EC));
    if (auto EC = Serializer.visitKnownRecord(Type, Record))
      return handleError(std::move(EC));

    auto ExpectedIndex = Serializer.visitTypeEndGetIndex(Type);
    if (!ExpectedIndex)
      return handleError(ExpectedIndex.takeError());

    return *ExpectedIndex;
  }

  TypeIndex writeSerializedRecord(ArrayRef<uint8_t> Record) {
    return Serializer.insertRecordBytes(Record);
  }

  TypeIndex writeSerializedRecord(const RemappedType &Record) {
    return Serializer.insertRecord(Record);
  }

  template <typename TFunc> void ForEachRecord(TFunc Func) {
    uint32_t Index = TypeIndex::FirstNonSimpleIndex;

    for (auto Record : Serializer.records()) {
      Func(TypeIndex(Index), Record);
      ++Index;
    }
  }

  ArrayRef<ArrayRef<uint8_t>> records() const { return Serializer.records(); }
};

class FieldListRecordBuilder {
  TypeTableBuilder &TypeTable;
  BumpPtrAllocator Allocator;
  TypeSerializer TempSerializer;
  CVType Type;

public:
  explicit FieldListRecordBuilder(TypeTableBuilder &TypeTable)
      : TypeTable(TypeTable), TempSerializer(Allocator, false) {
    Type.Type = TypeLeafKind::LF_FIELDLIST;
  }

  void begin() {
    TempSerializer.reset();

    if (auto EC = TempSerializer.visitTypeBegin(Type))
      consumeError(std::move(EC));
  }

  template <typename T> void writeMemberType(T &Record) {
    CVMemberRecord CVMR;
    CVMR.Kind = static_cast<TypeLeafKind>(Record.getKind());
    if (auto EC = TempSerializer.visitMemberBegin(CVMR))
      consumeError(std::move(EC));
    if (auto EC = TempSerializer.visitKnownMember(CVMR, Record))
      consumeError(std::move(EC));
    if (auto EC = TempSerializer.visitMemberEnd(CVMR))
      consumeError(std::move(EC));
  }

  TypeIndex end(bool Write) {
    TypeIndex Index;
    if (auto EC = TempSerializer.visitTypeEnd(Type)) {
      consumeError(std::move(EC));
      return TypeIndex();
    }

    if (Write) {
      for (auto Record : TempSerializer.records())
        Index = TypeTable.writeSerializedRecord(Record);
    }

    return Index;
  }
};

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_TYPETABLEBUILDER_H
