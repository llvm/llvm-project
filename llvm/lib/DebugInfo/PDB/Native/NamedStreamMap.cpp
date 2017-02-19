//===- NamedStreamMap.cpp - PDB Named Stream Map ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/NamedStreamMap.h"

#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/DebugInfo/MSF/StreamReader.h"
#include "llvm/DebugInfo/PDB/Native/HashTable.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <cstdint>

using namespace llvm;
using namespace llvm::msf;
using namespace llvm::pdb;

NamedStreamMap::NamedStreamMap() = default;

Error NamedStreamMap::load(StreamReader &Stream) {
  Mapping.clear();
  FinalizedHashTable.clear();
  FinalizedInfo.reset();

  uint32_t StringBufferSize;
  if (auto EC = Stream.readInteger(StringBufferSize, llvm::support::little))
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

Error NamedStreamMap::commit(msf::StreamWriter &Writer) const {
  assert(FinalizedInfo.hasValue());

  // The first field is the number of bytes of string data.
  if (auto EC = Writer.writeInteger(FinalizedInfo->StringDataBytes,
                                    llvm::support::little))
    return EC;

  // Now all of the string data itself.
  for (const auto &Item : Mapping) {
    if (auto EC = Writer.writeZeroString(Item.getKey()))
      return EC;
  }

  // And finally the Offset Index map.
  if (auto EC = FinalizedHashTable.commit(Writer))
    return EC;

  return Error::success();
}

uint32_t NamedStreamMap::finalize() {
  if (FinalizedInfo.hasValue())
    return FinalizedInfo->SerializedLength;

  // Build the finalized hash table.
  FinalizedHashTable.clear();
  FinalizedInfo.emplace();
  for (const auto &Item : Mapping) {
    FinalizedHashTable.set(FinalizedInfo->StringDataBytes, Item.getValue());
    FinalizedInfo->StringDataBytes += Item.getKeyLength() + 1;
  }

  // Number of bytes of string data.
  FinalizedInfo->SerializedLength += sizeof(support::ulittle32_t);
  // Followed by that many actual bytes of string data.
  FinalizedInfo->SerializedLength += FinalizedInfo->StringDataBytes;
  // Followed by the mapping from Offset to Index.
  FinalizedInfo->SerializedLength +=
      FinalizedHashTable.calculateSerializedLength();
  return FinalizedInfo->SerializedLength;
}

iterator_range<StringMapConstIterator<uint32_t>>
NamedStreamMap::entries() const {
  return make_range<StringMapConstIterator<uint32_t>>(Mapping.begin(),
                                                      Mapping.end());
}

bool NamedStreamMap::get(StringRef Stream, uint32_t &StreamNo) const {
  auto Iter = Mapping.find(Stream);
  if (Iter == Mapping.end())
    return false;
  StreamNo = Iter->second;
  return true;
}

void NamedStreamMap::set(StringRef Stream, uint32_t StreamNo) {
  FinalizedInfo.reset();
  Mapping[Stream] = StreamNo;
}

void NamedStreamMap::remove(StringRef Stream) {
  FinalizedInfo.reset();
  Mapping.erase(Stream);
}
