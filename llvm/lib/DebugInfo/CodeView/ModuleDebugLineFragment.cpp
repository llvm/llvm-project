//===- ModuleDebugLineFragment.cpp -------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/ModuleDebugLineFragment.h"

#include "llvm/DebugInfo/CodeView/CodeViewError.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugFileChecksumFragment.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugFragmentRecord.h"
#include "llvm/DebugInfo/CodeView/StringTable.h"

using namespace llvm;
using namespace llvm::codeview;

Error LineColumnExtractor::extract(BinaryStreamRef Stream, uint32_t &Len,
                                   LineColumnEntry &Item,
                                   const LineFragmentHeader *Header) {
  using namespace codeview;
  const LineBlockFragmentHeader *BlockHeader;
  BinaryStreamReader Reader(Stream);
  if (auto EC = Reader.readObject(BlockHeader))
    return EC;
  bool HasColumn = Header->Flags & uint16_t(LF_HaveColumns);
  uint32_t LineInfoSize =
      BlockHeader->NumLines *
      (sizeof(LineNumberEntry) + (HasColumn ? sizeof(ColumnNumberEntry) : 0));
  if (BlockHeader->BlockSize < sizeof(LineBlockFragmentHeader))
    return make_error<CodeViewError>(cv_error_code::corrupt_record,
                                     "Invalid line block record size");
  uint32_t Size = BlockHeader->BlockSize - sizeof(LineBlockFragmentHeader);
  if (LineInfoSize > Size)
    return make_error<CodeViewError>(cv_error_code::corrupt_record,
                                     "Invalid line block record size");
  // The value recorded in BlockHeader->BlockSize includes the size of
  // LineBlockFragmentHeader.
  Len = BlockHeader->BlockSize;
  Item.NameIndex = BlockHeader->NameIndex;
  if (auto EC = Reader.readArray(Item.LineNumbers, BlockHeader->NumLines))
    return EC;
  if (HasColumn) {
    if (auto EC = Reader.readArray(Item.Columns, BlockHeader->NumLines))
      return EC;
  }
  return Error::success();
}

ModuleDebugLineFragmentRef::ModuleDebugLineFragmentRef()
    : ModuleDebugFragmentRef(ModuleDebugFragmentKind::Lines) {}

Error ModuleDebugLineFragmentRef::initialize(BinaryStreamReader Reader) {
  if (auto EC = Reader.readObject(Header))
    return EC;

  if (auto EC =
          Reader.readArray(LinesAndColumns, Reader.bytesRemaining(), Header))
    return EC;

  return Error::success();
}

bool ModuleDebugLineFragmentRef::hasColumnInfo() const {
  return !!(Header->Flags & LF_HaveColumns);
}

ModuleDebugLineFragment::ModuleDebugLineFragment(
    ModuleDebugFileChecksumFragment &Checksums, StringTable &Strings)
    : ModuleDebugFragment(ModuleDebugFragmentKind::Lines),
      Checksums(Checksums) {}

void ModuleDebugLineFragment::createBlock(StringRef FileName) {
  uint32_t Offset = Checksums.mapChecksumOffset(FileName);

  Blocks.emplace_back(Offset);
}

void ModuleDebugLineFragment::addLineInfo(uint32_t Offset,
                                          const LineInfo &Line) {
  Block &B = Blocks.back();
  LineNumberEntry LNE;
  LNE.Flags = Line.getRawData();
  LNE.Offset = Offset;
  B.Lines.push_back(LNE);
}

void ModuleDebugLineFragment::addLineAndColumnInfo(uint32_t Offset,
                                                   const LineInfo &Line,
                                                   uint32_t ColStart,
                                                   uint32_t ColEnd) {
  Block &B = Blocks.back();
  assert(B.Lines.size() == B.Columns.size());

  addLineInfo(Offset, Line);
  ColumnNumberEntry CNE;
  CNE.StartColumn = ColStart;
  CNE.EndColumn = ColEnd;
  B.Columns.push_back(CNE);
}

Error ModuleDebugLineFragment::commit(BinaryStreamWriter &Writer) {
  LineFragmentHeader Header;
  Header.CodeSize = CodeSize;
  Header.Flags = hasColumnInfo() ? LF_HaveColumns : 0;
  Header.RelocOffset = RelocOffset;
  Header.RelocSegment = RelocSegment;

  if (auto EC = Writer.writeObject(Header))
    return EC;

  for (const auto &B : Blocks) {
    LineBlockFragmentHeader BlockHeader;
    assert(B.Lines.size() == B.Columns.size() || B.Columns.empty());

    BlockHeader.NumLines = B.Lines.size();
    BlockHeader.BlockSize = sizeof(LineBlockFragmentHeader);
    BlockHeader.BlockSize += BlockHeader.NumLines * sizeof(LineNumberEntry);
    if (hasColumnInfo())
      BlockHeader.BlockSize += BlockHeader.NumLines * sizeof(ColumnNumberEntry);
    BlockHeader.NameIndex = B.ChecksumBufferOffset;
    if (auto EC = Writer.writeObject(BlockHeader))
      return EC;

    if (auto EC = Writer.writeArray(makeArrayRef(B.Lines)))
      return EC;

    if (hasColumnInfo()) {
      if (auto EC = Writer.writeArray(makeArrayRef(B.Columns)))
        return EC;
    }
  }
  return Error::success();
}

uint32_t ModuleDebugLineFragment::calculateSerializedLength() {
  uint32_t Size = sizeof(LineFragmentHeader);
  for (const auto &B : Blocks) {
    Size += sizeof(LineBlockFragmentHeader);
    Size += B.Lines.size() * sizeof(LineNumberEntry);
    if (hasColumnInfo())
      Size += B.Columns.size() * sizeof(ColumnNumberEntry);
  }
  return Size;
}

void ModuleDebugLineFragment::setRelocationAddress(uint16_t Segment,
                                                   uint16_t Offset) {
  RelocOffset = Offset;
  RelocSegment = Segment;
}

void ModuleDebugLineFragment::setCodeSize(uint32_t Size) { CodeSize = Size; }

void ModuleDebugLineFragment::setFlags(LineFlags Flags) { this->Flags = Flags; }

bool ModuleDebugLineFragment::hasColumnInfo() const {
  return Flags & LF_HaveColumns;
}
