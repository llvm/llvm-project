//===- StringTable.cpp - PDB String Table -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/StringTable.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/MSF/BinaryStreamReader.h"
#include "llvm/DebugInfo/PDB/Native/Hash.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::msf;
using namespace llvm::support;
using namespace llvm::pdb;

StringTable::StringTable() : Signature(0), HashVersion(0), NameCount(0) {}

Error StringTable::load(StreamReader &Stream) {
  const StringTableHeader *H;
  if (auto EC = Stream.readObject(H))
    return EC;

  if (H->Signature != StringTableSignature)
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

  if (auto EC = Stream.readInteger(NameCount, llvm::support::little))
    return EC;
  return Error::success();
}

StringRef StringTable::getStringForID(uint32_t ID) const {
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

uint32_t StringTable::getIDForString(StringRef Str) const {
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

FixedStreamArray<support::ulittle32_t> StringTable::name_ids() const {
  return IDs;
}
