//===- DebugStringTableSubsection.cpp - CodeView String Table ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/DebugStringTableSubsection.h"

#include "llvm/Support/BinaryStream.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/BinaryStreamWriter.h"

using namespace llvm;
using namespace llvm::codeview;

DebugStringTableSubsectionRef::DebugStringTableSubsectionRef()
    : DebugSubsectionRef(DebugSubsectionKind::StringTable) {}

Error DebugStringTableSubsectionRef::initialize(BinaryStreamRef Contents) {
  Stream = Contents;
  return Error::success();
}

Expected<StringRef>
DebugStringTableSubsectionRef::getString(uint32_t Offset) const {
  BinaryStreamReader Reader(Stream);
  Reader.setOffset(Offset);
  StringRef Result;
  if (auto EC = Reader.readCString(Result))
    return std::move(EC);
  return Result;
}

DebugStringTableSubsection::DebugStringTableSubsection()
    : DebugSubsection(DebugSubsectionKind::StringTable) {}

uint32_t DebugStringTableSubsection::insert(StringRef S) {
  auto P = Strings.insert({S, StringSize});

  // If a given string didn't exist in the string table, we want to increment
  // the string table size.
  if (P.second)
    StringSize += S.size() + 1; // +1 for '\0'
  return P.first->second;
}

uint32_t DebugStringTableSubsection::calculateSerializedSize() const {
  return StringSize;
}

Error DebugStringTableSubsection::commit(BinaryStreamWriter &Writer) const {
  assert(Writer.bytesRemaining() == StringSize);
  uint32_t MaxOffset = 1;

  for (auto &Pair : Strings) {
    StringRef S = Pair.getKey();
    uint32_t Offset = Pair.getValue();
    Writer.setOffset(Offset);
    if (auto EC = Writer.writeCString(S))
      return EC;
    MaxOffset = std::max<uint32_t>(MaxOffset, Offset + S.size() + 1);
  }

  Writer.setOffset(MaxOffset);
  assert(Writer.bytesRemaining() == 0);
  return Error::success();
}

uint32_t DebugStringTableSubsection::size() const { return Strings.size(); }

uint32_t DebugStringTableSubsection::getStringId(StringRef S) const {
  auto P = Strings.find(S);
  assert(P != Strings.end());
  return P->second;
}
