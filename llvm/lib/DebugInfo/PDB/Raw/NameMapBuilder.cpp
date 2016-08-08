//===- NameMapBuilder.cpp - PDB Name Map Builder ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/NameMapBuilder.h"

#include "llvm/DebugInfo/MSF/StreamWriter.h"
#include "llvm/DebugInfo/PDB/Raw/NameMap.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::pdb;

NameMapBuilder::NameMapBuilder() {}

void NameMapBuilder::addMapping(StringRef Name, uint32_t Mapping) {
  StringDataBytes += Name.size() + 1;
  Map.insert({Name, Mapping});
}

Expected<std::unique_ptr<NameMap>> NameMapBuilder::build() {
  auto Result = llvm::make_unique<NameMap>();
  Result->Mapping = Map;
  return std::move(Result);
}

uint32_t NameMapBuilder::calculateSerializedLength() const {
  uint32_t TotalLength = 0;

  TotalLength += sizeof(support::ulittle32_t); // StringDataBytes value
  TotalLength += StringDataBytes;              // actual string data

  TotalLength += sizeof(support::ulittle32_t); // Hash Size
  TotalLength += sizeof(support::ulittle32_t); // Max Number of Strings
  TotalLength += sizeof(support::ulittle32_t); // Num Present Words
  // One bitmask word for each present entry
  TotalLength += Map.size() * sizeof(support::ulittle32_t);
  TotalLength += sizeof(support::ulittle32_t); // Num Deleted Words

  // For each present word, which we are treating as equivalent to the number of
  // entries in the table, we have a pair of integers.  An offset into the
  // string data, and a corresponding stream number.
  TotalLength += Map.size() * 2 * sizeof(support::ulittle32_t);

  return TotalLength;
}

Error NameMapBuilder::commit(msf::StreamWriter &Writer) const {
  // The first field is the number of bytes of string data.  So add
  // up the length of all strings plus a null terminator for each
  // one.
  uint32_t NumBytes = 0;
  for (auto B = Map.begin(), E = Map.end(); B != E; ++B) {
    NumBytes += B->getKeyLength() + 1;
  }

  if (auto EC = Writer.writeInteger(NumBytes)) // Number of bytes of string data
    return EC;
  // Now all of the string data itself.
  for (auto B = Map.begin(), E = Map.end(); B != E; ++B) {
    if (auto EC = Writer.writeZeroString(B->getKey()))
      return EC;
  }

  if (auto EC = Writer.writeInteger(Map.size())) // Hash Size
    return EC;

  if (auto EC = Writer.writeInteger(Map.size())) // Max Number of Strings
    return EC;

  if (auto EC = Writer.writeInteger(Map.size())) // Num Present Words
    return EC;

  // For each entry in the mapping, write a bit mask which represents a bucket
  // to store it in.  We don't use this, so the value we write isn't important
  // to us, it just has to be there.
  for (auto B = Map.begin(), E = Map.end(); B != E; ++B) {
    if (auto EC = Writer.writeInteger(1U))
      return EC;
  }

  if (auto EC = Writer.writeInteger(0U)) // Num Deleted Words
    return EC;

  // Mappings of each word.
  uint32_t OffsetSoFar = 0;
  for (auto B = Map.begin(), E = Map.end(); B != E; ++B) {
    // This is a list of key value pairs where the key is the offset into the
    // strings buffer, and the value is a stream number.  Write each pair.
    if (auto EC = Writer.writeInteger(OffsetSoFar))
      return EC;

    if (auto EC = Writer.writeInteger(B->second))
      return EC;

    OffsetSoFar += B->getKeyLength() + 1;
  }

  return Error::success();
}
