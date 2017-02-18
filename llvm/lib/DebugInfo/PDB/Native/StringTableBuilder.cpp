//===- StringTableBuilder.cpp - PDB String Table ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/StringTableBuilder.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/MSF/StreamWriter.h"
#include "llvm/DebugInfo/PDB/Native/Hash.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::support;
using namespace llvm::support::endian;
using namespace llvm::pdb;

uint32_t StringTableBuilder::insert(StringRef S) {
  auto P = Strings.insert({S, StringSize});

  // If a given string didn't exist in the string table, we want to increment
  // the string table size.
  if (P.second)
    StringSize += S.size() + 1; // +1 for '\0'
  return P.first->second;
}

static uint32_t computeBucketCount(uint32_t NumStrings) {
  // The /names stream is basically an on-disk open-addressing hash table.
  // Hash collisions are resolved by linear probing. We cannot make
  // utilization 100% because it will make the linear probing extremely
  // slow. But lower utilization wastes disk space. As a reasonable
  // load factor, we choose 80%. We need +1 because slot 0 is reserved.
  return (NumStrings + 1) * 1.25;
}

uint32_t StringTableBuilder::finalize() {
  uint32_t Size = 0;
  Size += sizeof(StringTableHeader);
  Size += StringSize;
  Size += sizeof(uint32_t); // Hash table begins with 4-byte size field.

  uint32_t BucketCount = computeBucketCount(Strings.size());
  Size += BucketCount * sizeof(uint32_t);

  Size +=
      sizeof(uint32_t); // The /names stream ends with the number of strings.
  return Size;
}

Error StringTableBuilder::commit(msf::StreamWriter &Writer) const {
  // Write a header
  StringTableHeader H;
  H.Signature = StringTableSignature;
  H.HashVersion = 1;
  H.ByteSize = StringSize;
  if (auto EC = Writer.writeObject(H))
    return EC;

  // Write a string table.
  uint32_t StringStart = Writer.getOffset();
  for (auto Pair : Strings) {
    StringRef S = Pair.first;
    uint32_t Offset = Pair.second;
    Writer.setOffset(StringStart + Offset);
    if (auto EC = Writer.writeZeroString(S))
      return EC;
  }
  Writer.setOffset(StringStart + StringSize);

  // Write a hash table.
  uint32_t BucketCount = computeBucketCount(Strings.size());
  if (auto EC = Writer.writeInteger(BucketCount, llvm::support::little))
    return EC;
  std::vector<ulittle32_t> Buckets(BucketCount);

  for (auto Pair : Strings) {
    StringRef S = Pair.first;
    uint32_t Offset = Pair.second;
    uint32_t Hash = hashStringV1(S);

    for (uint32_t I = 0; I != BucketCount; ++I) {
      uint32_t Slot = (Hash + I) % BucketCount;
      if (Slot == 0)
        continue; // Skip reserved slot
      if (Buckets[Slot] != 0)
        continue;
      Buckets[Slot] = Offset;
      break;
    }
  }

  if (auto EC = Writer.writeArray(ArrayRef<ulittle32_t>(Buckets)))
    return EC;
  if (auto EC = Writer.writeInteger(static_cast<uint32_t>(Strings.size()),
                                    llvm::support::little))
    return EC;
  return Error::success();
}
