//===- PDBStringTable.cpp - PDB String Table -----------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/PDBStringTable.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/PDB/Native/Hash.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::support;
using namespace llvm::pdb;

PDBStringTable::PDBStringTable() {}

Error PDBStringTable::load(BinaryStreamReader &Stream) {
  ByteSize = Stream.getLength();

  const PDBStringTableHeader *H;
  if (auto EC = Stream.readObject(H))
    return EC;

  if (H->Signature != PDBStringTableSignature)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Invalid hash table signature");
  if (H->HashVersion != 1 && H->HashVersion != 2)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Unsupported hash version");

  Signature = H->Signature;
  HashVersion = H->HashVersion;
  if (auto EC = Stream.readStreamRef(NamesBuffer, H->ByteSize))
    return joinErrors(std::move(EC),
                      make_error<RawError>(raw_error_code::corrupt_file,
                                           "Invalid hash table byte length"));

  const support::ulittle32_t *HashCount;
  if (auto EC = Stream.readObject(HashCount))
    return EC;

  if (auto EC = Stream.readArray(IDs, *HashCount))
    return joinErrors(std::move(EC),
                      make_error<RawError>(raw_error_code::corrupt_file,
                                           "Could not read bucket array"));

  if (Stream.bytesRemaining() < sizeof(support::ulittle32_t))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Missing name count");

  if (auto EC = Stream.readInteger(NameCount))
    return EC;

  if (Stream.bytesRemaining() > 0)
    return make_error<RawError>(raw_error_code::stream_too_long,
                                "Unexpected bytes found in string table");

  return Error::success();
}

uint32_t PDBStringTable::getByteSize() const { return ByteSize; }

StringRef PDBStringTable::getStringForID(uint32_t ID) const {
  if (ID == IDs[0])
    return StringRef();

  // NamesBuffer is a buffer of null terminated strings back to back.  ID is
  // the starting offset of the string we're looking for.  So just seek into
  // the desired offset and a read a null terminated stream from that offset.
  StringRef Result;
  BinaryStreamReader NameReader(NamesBuffer);
  NameReader.setOffset(ID);
  if (auto EC = NameReader.readCString(Result))
    consumeError(std::move(EC));
  return Result;
}

uint32_t PDBStringTable::getIDForString(StringRef Str) const {
  uint32_t Hash = (HashVersion == 1) ? hashStringV1(Str) : hashStringV2(Str);
  size_t Count = IDs.size();
  uint32_t Start = Hash % Count;
  for (size_t I = 0; I < Count; ++I) {
    // The hash is just a starting point for the search, but if it
    // doesn't work we should find the string no matter what, because
    // we iterate the entire array.
    uint32_t Index = (Start + I) % Count;

    uint32_t ID = IDs[Index];
    StringRef S = getStringForID(ID);
    if (S == Str)
      return ID;
  }
  // IDs[0] contains the ID of the "invalid" entry.
  return IDs[0];
}

FixedStreamArray<support::ulittle32_t> PDBStringTable::name_ids() const {
  return IDs;
}
