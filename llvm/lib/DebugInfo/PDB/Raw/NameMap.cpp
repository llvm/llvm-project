//===- NameMap.cpp - PDB Name Map -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/NameMap.h"

#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/DebugInfo/MSF/StreamReader.h"
#include "llvm/DebugInfo/PDB/Raw/HashTable.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <cstdint>

using namespace llvm;
using namespace llvm::msf;
using namespace llvm::pdb;

NameMap::NameMap() = default;

Error NameMap::load(StreamReader &Stream) {
  uint32_t StringBufferSize;
  if (auto EC = Stream.readInteger(StringBufferSize))
    return joinErrors(std::move(EC),
                      make_error<RawError>(raw_error_code::corrupt_file,
                                           "Expected string buffer size"));

  msf::ReadableStreamRef StringsBuffer;
  if (auto EC = Stream.readStreamRef(StringsBuffer, StringBufferSize))
    return EC;

  HashTable OffsetIndexMap;
  if (auto EC = OffsetIndexMap.load(Stream))
    return EC;

  uint32_t NameOffset;
  uint32_t NameIndex;
  for (const auto &Entry : OffsetIndexMap) {
    std::tie(NameOffset, NameIndex) = Entry;

    // Compute the offset of the start of the string relative to the stream.
    msf::StreamReader NameReader(StringsBuffer);
    NameReader.setOffset(NameOffset);
    // Pump out our c-string from the stream.
    StringRef Str;
    if (auto EC = NameReader.readZeroString(Str))
      return joinErrors(std::move(EC),
                        make_error<RawError>(raw_error_code::corrupt_file,
                                             "Expected name map name"));

    // Add this to a string-map from name to stream number.
    Mapping.insert({Str, NameIndex});
  }

  return Error::success();
}

iterator_range<StringMapConstIterator<uint32_t>> NameMap::entries() const {
  return make_range<StringMapConstIterator<uint32_t>>(Mapping.begin(),
                                                      Mapping.end());
}

bool NameMap::tryGetValue(StringRef Name, uint32_t &Value) const {
  auto Iter = Mapping.find(Name);
  if (Iter == Mapping.end())
    return false;
  Value = Iter->second;
  return true;
}
