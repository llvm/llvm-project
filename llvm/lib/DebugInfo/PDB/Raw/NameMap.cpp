//===- NameMap.cpp - PDB Name Map -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/NameMap.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/DebugInfo/PDB/Raw/StreamReader.h"

using namespace llvm;
using namespace llvm::pdb;

NameMap::NameMap() {}

std::error_code NameMap::load(StreamReader &Stream) {

  // This is some sort of weird string-set/hash table encoded in the stream.
  // It starts with the number of bytes in the table.
  uint32_t NumberOfBytes;
  Stream.readInteger(NumberOfBytes);

  // Following that field is the starting offset of strings in the name table.
  uint32_t StringsOffset = Stream.getOffset();
  Stream.setOffset(StringsOffset + NumberOfBytes);

  // This appears to be equivalent to the total number of strings *actually*
  // in the name table.
  uint32_t HashSize;
  Stream.readInteger(HashSize);

  // This appears to be an upper bound on the number of strings in the name
  // table.
  uint32_t MaxNumberOfStrings;
  Stream.readInteger(MaxNumberOfStrings);

  // This appears to be a hash table which uses bitfields to determine whether
  // or not a bucket is 'present'.
  uint32_t NumPresentWords;
  Stream.readInteger(NumPresentWords);

  // Store all the 'present' bits in a vector for later processing.
  SmallVector<uint32_t, 1> PresentWords;
  for (uint32_t I = 0; I != NumPresentWords; ++I) {
    uint32_t Word;
    Stream.readInteger(Word);
    PresentWords.push_back(Word);
  }

  // This appears to be a hash table which uses bitfields to determine whether
  // or not a bucket is 'deleted'.
  uint32_t NumDeletedWords;
  Stream.readInteger(NumDeletedWords);

  // Store all the 'deleted' bits in a vector for later processing.
  SmallVector<uint32_t, 1> DeletedWords;
  for (uint32_t I = 0; I != NumDeletedWords; ++I) {
    uint32_t Word;
    Stream.readInteger(Word);
    DeletedWords.push_back(Word);
  }

  BitVector Present(MaxNumberOfStrings, false);
  if (!PresentWords.empty())
    Present.setBitsInMask(PresentWords.data(), PresentWords.size());
  BitVector Deleted(MaxNumberOfStrings, false);
  if (!DeletedWords.empty())
    Deleted.setBitsInMask(DeletedWords.data(), DeletedWords.size());

  for (uint32_t I = 0; I < MaxNumberOfStrings; ++I) {
    if (!Present.test(I))
      continue;

    // For all present entries, dump out their mapping.

    // This appears to be an offset relative to the start of the strings.
    // It tells us where the null-terminated string begins.
    uint32_t NameOffset;
    Stream.readInteger(NameOffset);

    // This appears to be a stream number into the stream directory.
    uint32_t NameIndex;
    Stream.readInteger(NameIndex);

    // Compute the offset of the start of the string relative to the stream.
    uint32_t StringOffset = StringsOffset + NameOffset;
    uint32_t OldOffset = Stream.getOffset();
    // Pump out our c-string from the stream.
    std::string Str;
    Stream.setOffset(StringOffset);
    Stream.readZeroString(Str);

    Stream.setOffset(OldOffset);
    // Add this to a string-map from name to stream number.
    Mapping.insert({Str, NameIndex});
  }

  return std::error_code();
}

bool NameMap::tryGetValue(StringRef Name, uint32_t &Value) const {
  auto Iter = Mapping.find(Name);
  if (Iter == Mapping.end())
    return false;
  Value = Iter->second;
  return true;
}
