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
#include "llvm/DebugInfo/CodeView/ModuleDebugFragmentRecord.h"

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
