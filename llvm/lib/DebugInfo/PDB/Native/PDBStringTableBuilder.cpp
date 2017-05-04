//===- PDBStringTableBuilder.cpp - PDB String Table -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/PDBStringTableBuilder.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Native/Hash.h"
#include "llvm/DebugInfo/PDB/Native/PDBFileBuilder.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::msf;
using namespace llvm::support;
using namespace llvm::support::endian;
using namespace llvm::pdb;

uint32_t PDBStringTableBuilder::insert(StringRef S) {
  return Strings.insert(S);
}

static uint32_t computeBucketCount(uint32_t NumStrings) {
  // The /names stream is basically an on-disk open-addressing hash table.
  // Hash collisions are resolved by linear probing. We cannot make
  // utilization 100% because it will make the linear probing extremely
  // slow. But lower utilization wastes disk space. As a reasonable
  // load factor, we choose 80%. We need +1 because slot 0 is reserved.
  return (NumStrings + 1) * 1.25;
}

uint32_t PDBStringTableBuilder::calculateHashTableSize() const {
  uint32_t Size = sizeof(uint32_t); // Hash table begins with 4-byte size field.
  Size += sizeof(uint32_t) * computeBucketCount(Strings.size());

  return Size;
}

uint32_t PDBStringTableBuilder::calculateSerializedSize() const {
  uint32_t Size = 0;
  Size += sizeof(PDBStringTableHeader);
  Size += Strings.calculateSerializedSize();
  Size += calculateHashTableSize();
  Size += sizeof(uint32_t); // The /names stream ends with the string count.
  return Size;
}

Error PDBStringTableBuilder::writeHeader(BinaryStreamWriter &Writer) const {
  // Write a header
  PDBStringTableHeader H;
  H.Signature = PDBStringTableSignature;
  H.HashVersion = 1;
  H.ByteSize = Strings.calculateSerializedSize();
  if (auto EC = Writer.writeObject(H))
    return EC;
  assert(Writer.bytesRemaining() == 0);
  return Error::success();
}

Error PDBStringTableBuilder::writeStrings(BinaryStreamWriter &Writer) const {
  if (auto EC = Strings.commit(Writer))
    return EC;

  assert(Writer.bytesRemaining() == 0);
  return Error::success();
}

Error PDBStringTableBuilder::writeHashTable(BinaryStreamWriter &Writer) const {
  // Write a hash table.
  uint32_t BucketCount = computeBucketCount(Strings.size());
  if (auto EC = Writer.writeInteger(BucketCount))
    return EC;
  std::vector<ulittle32_t> Buckets(BucketCount);

  for (auto &Pair : Strings) {
    StringRef S = Pair.getKey();
    uint32_t Offset = Pair.getValue();
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

  assert(Writer.bytesRemaining() == 0);
  return Error::success();
}

Error PDBStringTableBuilder::writeEpilogue(BinaryStreamWriter &Writer) const {
  if (auto EC = Writer.writeInteger<uint32_t>(Strings.size()))
    return EC;
  assert(Writer.bytesRemaining() == 0);
  return Error::success();
}

Error PDBStringTableBuilder::commit(BinaryStreamWriter &Writer) const {
  BinaryStreamWriter SectionWriter;

  std::tie(SectionWriter, Writer) = Writer.split(sizeof(PDBStringTableHeader));
  if (auto EC = writeHeader(SectionWriter))
    return EC;

  std::tie(SectionWriter, Writer) =
      Writer.split(Strings.calculateSerializedSize());
  if (auto EC = writeStrings(SectionWriter))
    return EC;

  std::tie(SectionWriter, Writer) = Writer.split(calculateHashTableSize());
  if (auto EC = writeHashTable(SectionWriter))
    return EC;

  std::tie(SectionWriter, Writer) = Writer.split(sizeof(uint32_t));
  if (auto EC = writeEpilogue(SectionWriter))
    return EC;

  return Error::success();
}
