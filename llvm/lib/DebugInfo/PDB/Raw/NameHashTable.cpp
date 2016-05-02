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
#include "llvm/DebugInfo/PDB/Raw/ByteStream.h"
#include "llvm/DebugInfo/PDB/Raw/StreamReader.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::support;
using namespace llvm::pdb;

typedef uint32_t *PUL;
typedef uint16_t *PUS;

static inline uint32_t HashStringV1(StringRef Str) {
  uint32_t Result = 0;
  uint32_t Size = Str.size();

  ArrayRef<ulittle32_t> Longs(reinterpret_cast<const ulittle32_t *>(Str.data()),
                              Size / 4);

  for (auto Value : Longs)
    Result ^= Value;

  const uint8_t *Remainder = reinterpret_cast<const uint8_t *>(Longs.end());
  uint32_t RemainderSize = Size - Longs.size() * 4;

  // Maximum of 3 bytes left.  Hash a 2 byte word if possible, then hash the
  // possibly remaining 1 byte.
  if (RemainderSize >= 2) {
    uint16_t Value = *reinterpret_cast<const ulittle16_t *>(Remainder);
    Result ^= static_cast<uint32_t>(Value);
    Remainder += 2;
    RemainderSize -= 2;
  }

  // hash possible odd byte
  if (RemainderSize == 1) {
    Result ^= *(Remainder++);
  }

  const uint32_t toLowerMask = 0x20202020;
  Result |= toLowerMask;
  Result ^= (Result >> 11);

  return Result ^ (Result >> 16);
}

static inline uint32_t HashStringV2(StringRef Str) {
  uint32_t Hash = 0xb170a1bf;

  ArrayRef<char> Buffer(Str.begin(), Str.end());

  ArrayRef<ulittle32_t> Items(
      reinterpret_cast<const ulittle32_t *>(Buffer.data()),
      Buffer.size() / sizeof(ulittle32_t));
  for (ulittle32_t Item : Items) {
    Hash += Item;
    Hash += (Hash << 10);
    Hash ^= (Hash >> 6);
  }
  Buffer = Buffer.slice(Items.size() * sizeof(ulittle32_t));
  for (uint8_t Item : Buffer) {
    Hash += Item;
    Hash += (Hash << 10);
    Hash ^= (Hash >> 6);
  }

  return Hash * 1664525L + 1013904223L;
}

NameHashTable::NameHashTable() : Signature(0), HashVersion(0), NameCount(0) {}

std::error_code NameHashTable::load(StreamReader &Stream) {
  struct Header {
    support::ulittle32_t Signature;
    support::ulittle32_t HashVersion;
    support::ulittle32_t ByteSize;
  };

  Header H;
  Stream.readObject(&H);
  if (H.Signature != 0xEFFEEFFE)
    return std::make_error_code(std::errc::illegal_byte_sequence);
  if (H.HashVersion != 1 && H.HashVersion != 2)
    return std::make_error_code(std::errc::not_supported);

  Signature = H.Signature;
  HashVersion = H.HashVersion;
  NamesBuffer.initialize(Stream, H.ByteSize);

  support::ulittle32_t HashCount;
  Stream.readObject(&HashCount);
  std::vector<support::ulittle32_t> BucketArray(HashCount);
  Stream.readArray<support::ulittle32_t>(BucketArray);
  IDs.assign(BucketArray.begin(), BucketArray.end());

  if (Stream.bytesRemaining() < sizeof(support::ulittle32_t))
    return std::make_error_code(std::errc::illegal_byte_sequence);

  Stream.readInteger(NameCount);
  return std::error_code();
}

StringRef NameHashTable::getStringForID(uint32_t ID) const {
  if (ID == IDs[0])
    return StringRef();

  return StringRef(NamesBuffer.str().begin() + ID);
}

uint32_t NameHashTable::getIDForString(StringRef Str) const {
  uint32_t Hash = (HashVersion == 1) ? HashStringV1(Str) : HashStringV2(Str);
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

ArrayRef<uint32_t> NameHashTable::name_ids() const {
  return ArrayRef<uint32_t>(IDs).slice(1, NameCount);
}
