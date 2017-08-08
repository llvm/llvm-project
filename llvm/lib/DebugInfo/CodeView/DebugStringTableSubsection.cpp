//===- DebugStringTableSubsection.cpp - CodeView String Table -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/DebugStringTableSubsection.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <cassert>
#include <cstdint>

using namespace llvm;
using namespace llvm::codeview;

DebugStringTableSubsectionRef::DebugStringTableSubsectionRef()
    : DebugSubsectionRef(DebugSubsectionKind::StringTable) {}

Error DebugStringTableSubsectionRef::initialize(BinaryStreamRef Contents) {
  Stream = Contents;
  return Error::success();
}

Error DebugStringTableSubsectionRef::initialize(BinaryStreamReader &Reader) {
  return Reader.readStreamRef(Stream);
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
  uint32_t Begin = Writer.getOffset();
  uint32_t End = Begin + StringSize;

  // Write a null string at the beginning.
  if (auto EC = Writer.writeCString(StringRef()))
    return EC;

  for (auto &Pair : Strings) {
    StringRef S = Pair.getKey();
    uint32_t Offset = Begin + Pair.getValue();
    Writer.setOffset(Offset);
    if (auto EC = Writer.writeCString(S))
      return EC;
    assert(Writer.getOffset() <= End);
  }

  Writer.setOffset(End);
  assert((End - Begin) == StringSize);
  return Error::success();
}

uint32_t DebugStringTableSubsection::size() const { return Strings.size(); }

uint32_t DebugStringTableSubsection::getStringId(StringRef S) const {
  auto Iter = Strings.find(S);
  assert(Iter != Strings.end());
  return Iter->second;
}
