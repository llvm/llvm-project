//===- TypeSerialzier.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/TypeSerializer.h"

#include "llvm/Support/BinaryStreamWriter.h"

#include <string.h>

using namespace llvm;
using namespace llvm::codeview;

bool TypeSerializer::isInFieldList() const {
  return TypeKind.hasValue() && *TypeKind == TypeLeafKind::LF_FIELDLIST;
}

TypeIndex TypeSerializer::calcNextTypeIndex() const {
  if (LastTypeIndex.isNoneType())
    return TypeIndex(TypeIndex::FirstNonSimpleIndex);
  else
    return TypeIndex(LastTypeIndex.getIndex() + 1);
}

TypeIndex TypeSerializer::incrementTypeIndex() {
  TypeIndex Previous = LastTypeIndex;
  LastTypeIndex = calcNextTypeIndex();
  return Previous;
}

MutableArrayRef<uint8_t> TypeSerializer::getCurrentSubRecordData() {
  assert(isInFieldList());
  return getCurrentRecordData().drop_front(CurrentSegment.length());
}

MutableArrayRef<uint8_t> TypeSerializer::getCurrentRecordData() {
  return MutableArrayRef<uint8_t>(RecordBuffer).take_front(Writer.getOffset());
}

Error TypeSerializer::writeRecordPrefix(TypeLeafKind Kind) {
  RecordPrefix Prefix;
  Prefix.RecordKind = Kind;
  Prefix.RecordLen = 0;
  if (auto EC = Writer.writeObject(Prefix))
    return EC;
  return Error::success();
}

TypeIndex
TypeSerializer::insertRecordBytesPrivate(ArrayRef<uint8_t> &Record) {
  assert(Record.size() % 4 == 0 && "Record is not aligned to 4 bytes!");

  StringRef S(reinterpret_cast<const char *>(Record.data()), Record.size());

  TypeIndex NextTypeIndex = calcNextTypeIndex();
  auto Result = HashedRecords.try_emplace(S, NextTypeIndex);

  StringRef NewData = Result.first->getKey();
  Record = ArrayRef<uint8_t>(NewData.bytes_begin(), NewData.bytes_end());

  if (Result.second) {
    // If this triggered an insert into the map, store the bytes.
    LastTypeIndex = NextTypeIndex;
    SeenRecords.push_back(Record);
  }

  return Result.first->getValue();
}

Expected<MutableArrayRef<uint8_t>>
TypeSerializer::addPadding(MutableArrayRef<uint8_t> Record) {
  uint32_t Align = Record.size() % 4;
  if (Align == 0)
    return Record;

  int PaddingBytes = 4 - Align;
  int N = PaddingBytes;
  while (PaddingBytes > 0) {
    uint8_t Pad = static_cast<uint8_t>(LF_PAD0 + PaddingBytes);
    if (auto EC = Writer.writeInteger(Pad))
      return std::move(EC);
    --PaddingBytes;
  }
  return MutableArrayRef<uint8_t>(Record.data(), Record.size() + N);
}

TypeSerializer::TypeSerializer(BumpPtrAllocator &Storage)
    : RecordStorage(Storage), LastTypeIndex(),
      RecordBuffer(MaxRecordLength * 2),
      Stream(RecordBuffer, llvm::support::little), Writer(Stream),
      Mapping(Writer), HashedRecords(Storage) {
  // RecordBuffer needs to be able to hold enough data so that if we are 1
  // byte short of MaxRecordLen, and then we try to write MaxRecordLen bytes,
  // we won't overflow.
}

ArrayRef<ArrayRef<uint8_t>> TypeSerializer::records() const {
  return SeenRecords;
}

TypeIndex TypeSerializer::getLastTypeIndex() const { return LastTypeIndex; }

TypeIndex TypeSerializer::insertRecordBytes(ArrayRef<uint8_t> Record) {
  assert(!TypeKind.hasValue() && "Already in a type mapping!");
  assert(Writer.getOffset() == 0 && "Stream has data already!");

  return insertRecordBytesPrivate(Record);
}

Error TypeSerializer::visitTypeBegin(CVType &Record) {
  assert(!TypeKind.hasValue() && "Already in a type mapping!");
  assert(Writer.getOffset() == 0 && "Stream has data already!");

  if (auto EC = writeRecordPrefix(Record.kind()))
    return EC;

  TypeKind = Record.kind();
  if (auto EC = Mapping.visitTypeBegin(Record))
    return EC;

  return Error::success();
}

Expected<TypeIndex> TypeSerializer::visitTypeEndGetIndex(CVType &Record) {
  assert(TypeKind.hasValue() && "Not in a type mapping!");
  if (auto EC = Mapping.visitTypeEnd(Record))
    return std::move(EC);

  // Update the record's length and fill out the CVType members to point to
  // the stable memory holding the record's data.
  auto ThisRecordData = getCurrentRecordData();
  auto ExpectedData = addPadding(ThisRecordData);
  if (!ExpectedData)
    return ExpectedData.takeError();
  ThisRecordData = *ExpectedData;

  RecordPrefix *Prefix =
      reinterpret_cast<RecordPrefix *>(ThisRecordData.data());
  Prefix->RecordLen = ThisRecordData.size() - sizeof(uint16_t);

  Record.Type = *TypeKind;
  Record.RecordData = ThisRecordData;
  TypeIndex InsertedTypeIndex = insertRecordBytesPrivate(Record.RecordData);

  // Write out each additional segment in reverse order, and update each
  // record's continuation index to point to the previous one.
  for (auto X : reverse(FieldListSegments)) {
    auto CIBytes = X.take_back(sizeof(uint32_t));
    support::ulittle32_t *CI =
        reinterpret_cast<support::ulittle32_t *>(CIBytes.data());
    assert(*CI == 0xB0C0B0C0 && "Invalid TypeIndex placeholder");
    *CI = InsertedTypeIndex.getIndex();
    InsertedTypeIndex = insertRecordBytesPrivate(X);
  }

  TypeKind.reset();
  Writer.setOffset(0);
  FieldListSegments.clear();
  CurrentSegment.SubRecords.clear();

  return InsertedTypeIndex;
}

Error TypeSerializer::visitTypeEnd(CVType &Record) {
  auto ExpectedIndex = visitTypeEndGetIndex(Record);
  if (!ExpectedIndex)
    return ExpectedIndex.takeError();
  return Error::success();
}

Error TypeSerializer::visitMemberBegin(CVMemberRecord &Record) {
  assert(isInFieldList() && "Not in a field list!");
  assert(!MemberKind.hasValue() && "Already in a member record!");
  MemberKind = Record.Kind;

  if (auto EC = Mapping.visitMemberBegin(Record))
    return EC;

  return Error::success();
}

Error TypeSerializer::visitMemberEnd(CVMemberRecord &Record) {
  if (auto EC = Mapping.visitMemberEnd(Record))
    return EC;

  // Check if this subrecord makes the current segment not fit in 64K minus
  // the space for a continuation record (8 bytes). If the segment does not
  // fit, insert a continuation record.
  if (Writer.getOffset() > MaxRecordLength - ContinuationLength) {
    MutableArrayRef<uint8_t> Data = getCurrentRecordData();
    SubRecord LastSubRecord = CurrentSegment.SubRecords.back();
    uint32_t CopySize = CurrentSegment.length() - LastSubRecord.Size;
    auto CopyData = Data.take_front(CopySize);
    auto LeftOverData = Data.drop_front(CopySize);
    assert(LastSubRecord.Size == LeftOverData.size());

    // Allocate stable storage for the record and copy the old record plus
    // continuation over.
    uint16_t LengthWithSize = CopySize + ContinuationLength;
    assert(LengthWithSize <= MaxRecordLength);
    RecordPrefix *Prefix = reinterpret_cast<RecordPrefix *>(CopyData.data());
    Prefix->RecordLen = LengthWithSize - sizeof(uint16_t);

    uint8_t *SegmentBytes = RecordStorage.Allocate<uint8_t>(LengthWithSize);
    auto SavedSegment = MutableArrayRef<uint8_t>(SegmentBytes, LengthWithSize);
    MutableBinaryByteStream CS(SavedSegment, llvm::support::little);
    BinaryStreamWriter CW(CS);
    if (auto EC = CW.writeBytes(CopyData))
      return EC;
    if (auto EC = CW.writeEnum(TypeLeafKind::LF_INDEX))
      return EC;
    if (auto EC = CW.writeInteger<uint16_t>(0))
      return EC;
    if (auto EC = CW.writeInteger<uint32_t>(0xB0C0B0C0))
      return EC;
    FieldListSegments.push_back(SavedSegment);

    // Write a new placeholder record prefix to mark the start of this new
    // top-level record.
    Writer.setOffset(0);
    if (auto EC = writeRecordPrefix(TypeLeafKind::LF_FIELDLIST))
      return EC;

    // Then move over the subrecord that overflowed the old segment to the
    // beginning of this segment.  Note that we have to use memmove here
    // instead of Writer.writeBytes(), because the new and old locations
    // could overlap.
    ::memmove(Stream.data().data() + sizeof(RecordPrefix), LeftOverData.data(),
              LeftOverData.size());
    // And point the segment writer at the end of that subrecord.
    Writer.setOffset(LeftOverData.size() + sizeof(RecordPrefix));

    CurrentSegment.SubRecords.clear();
    CurrentSegment.SubRecords.push_back(LastSubRecord);
  }

  // Update the CVMemberRecord since we may have shifted around or gotten
  // padded.
  Record.Data = getCurrentSubRecordData();

  MemberKind.reset();
  return Error::success();
}
