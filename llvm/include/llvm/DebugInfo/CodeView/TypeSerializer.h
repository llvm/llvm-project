//===- TypeSerializer.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPESERIALIZER_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPESERIALIZER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/RecordSerialization.h"
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

class TypeHasher;

class TypeSerializer : public TypeVisitorCallbacks {
  struct SubRecord {
    SubRecord(TypeLeafKind K, uint32_t S) : Kind(K), Size(S) {}

    TypeLeafKind Kind;
    uint32_t Size = 0;
  };
  struct RecordSegment {
    SmallVector<SubRecord, 16> SubRecords;

    uint32_t length() const {
      uint32_t L = sizeof(RecordPrefix);
      for (const auto &R : SubRecords) {
        L += R.Size;
      }
      return L;
    }
  };

  using MutableRecordList = SmallVector<MutableArrayRef<uint8_t>, 2>;

  static constexpr uint8_t ContinuationLength = 8;
  BumpPtrAllocator &RecordStorage;
  RecordSegment CurrentSegment;
  MutableRecordList FieldListSegments;

  Optional<TypeLeafKind> TypeKind;
  Optional<TypeLeafKind> MemberKind;
  std::vector<uint8_t> RecordBuffer;
  MutableBinaryByteStream Stream;
  BinaryStreamWriter Writer;
  TypeRecordMapping Mapping;

  /// Private type record hashing implementation details are handled here.
  std::unique_ptr<TypeHasher> Hasher;

  /// Contains a list of all records indexed by TypeIndex.toArrayIndex().
  SmallVector<ArrayRef<uint8_t>, 2> SeenRecords;

  /// Temporary storage that we use to copy a record's data while re-writing
  /// its type indices.
  SmallVector<uint8_t, 256> RemapStorage;

  TypeIndex nextTypeIndex() const;

  bool isInFieldList() const;
  MutableArrayRef<uint8_t> getCurrentSubRecordData();
  MutableArrayRef<uint8_t> getCurrentRecordData();
  Error writeRecordPrefix(TypeLeafKind Kind);

  Expected<MutableArrayRef<uint8_t>>
  addPadding(MutableArrayRef<uint8_t> Record);

public:
  explicit TypeSerializer(BumpPtrAllocator &Storage, bool Hash = true);
  ~TypeSerializer() override;

  void reset();

  BumpPtrAllocator &getAllocator() { return RecordStorage; }

  ArrayRef<ArrayRef<uint8_t>> records() const;
  TypeIndex insertRecordBytes(ArrayRef<uint8_t> &Record);
  TypeIndex insertRecord(const RemappedType &Record);
  Expected<TypeIndex> visitTypeEndGetIndex(CVType &Record);

  using TypeVisitorCallbacks::visitTypeBegin;
  Error visitTypeBegin(CVType &Record) override;
  Error visitTypeEnd(CVType &Record) override;
  Error visitMemberBegin(CVMemberRecord &Record) override;
  Error visitMemberEnd(CVMemberRecord &Record) override;

#define TYPE_RECORD(EnumName, EnumVal, Name)                                   \
  virtual Error visitKnownRecord(CVType &CVR, Name##Record &Record) override { \
    return visitKnownRecordImpl(CVR, Record);                                  \
  }
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  Error visitKnownMember(CVMemberRecord &CVR, Name##Record &Record) override { \
    return visitKnownMemberImpl<Name##Record>(CVR, Record);                    \
  }
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/CodeViewTypes.def"

private:
  template <typename RecordKind>
  Error visitKnownRecordImpl(CVType &CVR, RecordKind &Record) {
    return Mapping.visitKnownRecord(CVR, Record);
  }

  template <typename RecordType>
  Error visitKnownMemberImpl(CVMemberRecord &CVR, RecordType &Record) {
    assert(CVR.Kind == static_cast<TypeLeafKind>(Record.getKind()));

    if (auto EC = Writer.writeEnum(CVR.Kind))
      return EC;

    if (auto EC = Mapping.visitKnownMember(CVR, Record))
      return EC;

    // Get all the data that was just written and is yet to be committed to
    // the current segment.  Then pad it to 4 bytes.
    MutableArrayRef<uint8_t> ThisRecord = getCurrentSubRecordData();
    auto ExpectedRecord = addPadding(ThisRecord);
    if (!ExpectedRecord)
      return ExpectedRecord.takeError();
    ThisRecord = *ExpectedRecord;

    CurrentSegment.SubRecords.emplace_back(CVR.Kind, ThisRecord.size());
    CVR.Data = ThisRecord;

    // Both the last subrecord and the total length of this segment should be
    // multiples of 4.
    assert(ThisRecord.size() % 4 == 0);
    assert(CurrentSegment.length() % 4 == 0);

    return Error::success();
  }
};

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_TYPESERIALIZER_H
