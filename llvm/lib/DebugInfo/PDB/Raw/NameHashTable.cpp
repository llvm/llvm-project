//===- NameHashTable.cpp - PDB Name Hash Table ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/NameHashTable.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/MSF/StreamReader.h"
#include "llvm/DebugInfo/PDB/Raw/Hash.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::msf;
using namespace llvm::support;
using namespace llvm::pdb;

NameHashTable::NameHashTable() : Signature(0), HashVersion(0), NameCount(0) {}

Error NameHashTable::load(StreamReader &Stream) {
  struct Header {
    support::ulittle32_t Signature;
    support::ulittle32_t HashVersion;
    support::ulittle32_t ByteSize;
  };

  const Header *H;
  if (auto EC = Stream.readObject(H))
    return EC;

  if (H->Signature != 0xEFFEEFFE)
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
  return Error::success();
}

StringRef NameHashTable::getStringForID(uint32_t ID) const {
  if (ID == IDs[0])
    return StringRef();

  // NamesBuffer is a buffer of null terminated strings back to back.  ID is
  // the starting offset of the string we're looking for.  So just seek into
  // the desired offset and a read a null terminated stream from that offset.
  StringRef Result;
  StreamReader NameReader(NamesBuffer);
  NameReader.setOffset(ID);
  if (auto EC = NameReader.readZeroString(Result))
    consumeError(std::move(EC));
  return Result;
}

uint32_t NameHashTable::getIDForString(StringRef Str) const {
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

FixedStreamArray<support::ulittle32_t> NameHashTable::name_ids() const {
  return IDs;
}
