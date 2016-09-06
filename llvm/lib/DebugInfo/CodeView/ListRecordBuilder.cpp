//===-- ListRecordBuilder.cpp ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/DebugInfo/CodeView/ListRecordBuilder.h"
#include "llvm/DebugInfo/CodeView/TypeTableBuilder.h"

using namespace llvm;
using namespace codeview;

ListRecordBuilder::ListRecordBuilder(TypeRecordKind Kind)
    : Kind(Kind), Builder(Kind) {}

void ListRecordBuilder::writeMemberType(const ListContinuationRecord &R) {
  TypeRecordBuilder &Builder = getBuilder();

  assert(getLastContinuationSize() < 65535 - 8 && "continuation won't fit");

  Builder.writeTypeRecordKind(TypeRecordKind::ListContinuation);
  Builder.writeUInt16(0);
  Builder.writeTypeIndex(R.getContinuationIndex());

  // End the current segment manually so that nothing comes after the
  // continuation.
  ContinuationOffsets.push_back(Builder.size());
  SubrecordStart = Builder.size();
}

void ListRecordBuilder::finishSubRecord() {
  // The type table inserts a 16 bit size field before each list, so factor that
  // into our alignment padding.
  uint32_t Remainder =
      (Builder.size() + 2 * (ContinuationOffsets.size() + 1)) % 4;
  if (Remainder != 0) {
    for (int32_t PaddingBytesLeft = 4 - Remainder; PaddingBytesLeft > 0;
         --PaddingBytesLeft) {
      Builder.writeUInt8(LF_PAD0 + PaddingBytesLeft);
    }
  }

  // Check if this subrecord makes the current segment not fit in 64K minus the
  // space for a continuation record (8 bytes). If the segment does not fit,
  // back up and insert a continuation record, sliding the current subrecord
  // down.
  if (getLastContinuationSize() > 65535 - 8) {
    assert(SubrecordStart != 0 && "can't slide from the start!");
    SmallString<128> SubrecordCopy(
        Builder.str().slice(SubrecordStart, Builder.size()));
    assert(SubrecordCopy.size() < 65530 && "subrecord is too large to slide!");
    Builder.truncate(SubrecordStart);

    // Write a placeholder continuation record.
    Builder.writeTypeRecordKind(TypeRecordKind::ListContinuation);
    Builder.writeUInt16(0);
    Builder.writeUInt32(0);
    ContinuationOffsets.push_back(Builder.size());
    assert(Builder.size() == SubrecordStart + 8 && "wrong continuation size");
    assert(getLastContinuationSize() < 65535 && "segment too big");

    // Start a new list record of the appropriate kind, and copy the previous
    // subrecord into place.
    Builder.writeTypeRecordKind(Kind);
    Builder.writeBytes(SubrecordCopy);
  }

  SubrecordStart = Builder.size();
}

TypeIndex ListRecordBuilder::writeListRecord(TypeTableBuilder &Table) {
  // Get the continuation segments as a reversed vector of StringRefs for
  // convenience.
  SmallVector<StringRef, 1> Segments;
  StringRef Data = str();
  size_t LastEnd = 0;
  for (size_t SegEnd : ContinuationOffsets) {
    Segments.push_back(Data.slice(LastEnd, SegEnd));
    LastEnd = SegEnd;
  }
  Segments.push_back(Data.slice(LastEnd, Builder.size()));

  // Pop the last record off and emit it directly.
  StringRef LastRec = Segments.pop_back_val();
  TypeIndex ContinuationIndex = Table.writeRecord(LastRec);

  // Emit each record with a continuation in reverse order, so that each one
  // references the previous record.
  for (StringRef Rec : reverse(Segments)) {
    assert(*reinterpret_cast<const ulittle16_t *>(Rec.data()) ==
           unsigned(Kind));
    ulittle32_t *ContinuationPtr =
        reinterpret_cast<ulittle32_t *>(const_cast<char *>(Rec.end())) - 1;
    *ContinuationPtr = ContinuationIndex.getIndex();
    ContinuationIndex = Table.writeRecord(Rec);
  }
  return ContinuationIndex;
}
