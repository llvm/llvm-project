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
#include "llvm/DebugInfo/CodeView/StreamReader.h"
#include "llvm/DebugInfo/CodeView/StreamWriter.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

NameMap::NameMap() {}

Error NameMap::load(codeview::StreamReader &Stream) {

  // This is some sort of weird string-set/hash table encoded in the stream.
  // It starts with the number of bytes in the table.
  uint32_t NumberOfBytes;
  if (auto EC = Stream.readInteger(NumberOfBytes))
    return joinErrors(std::move(EC),
                      make_error<RawError>(raw_error_code::corrupt_file,
                                           "Expected name map length"));
  if (Stream.bytesRemaining() < NumberOfBytes)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Invalid name map length");

  // Following that field is the starting offset of strings in the name table.
  uint32_t StringsOffset = Stream.getOffset();
  Stream.setOffset(StringsOffset + NumberOfBytes);

  // This appears to be equivalent to the total number of strings *actually*
  // in the name table.
  uint32_t HashSize;
  if (auto EC = Stream.readInteger(HashSize))
    return joinErrors(std::move(EC),
                      make_error<RawError>(raw_error_code::corrupt_file,
                                           "Expected name map hash size"));

  // This appears to be an upper bound on the number of strings in the name
  // table.
  uint32_t MaxNumberOfStrings;
  if (auto EC = Stream.readInteger(MaxNumberOfStrings))
    return joinErrors(std::move(EC),
                      make_error<RawError>(raw_error_code::corrupt_file,
                                           "Expected name map max strings"));

  if (MaxNumberOfStrings > (UINT32_MAX / sizeof(uint32_t)))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Implausible number of strings");

  const uint32_t MaxNumberOfWords = UINT32_MAX / (sizeof(uint32_t) * 8);

  // This appears to be a hash table which uses bitfields to determine whether
  // or not a bucket is 'present'.
  uint32_t NumPresentWords;
  if (auto EC = Stream.readInteger(NumPresentWords))
    return joinErrors(std::move(EC),
                      make_error<RawError>(raw_error_code::corrupt_file,
                                           "Expected name map num words"));

  if (NumPresentWords > MaxNumberOfWords)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Number of present words is too large");

  SparseBitVector<> Present;
  for (uint32_t I = 0; I != NumPresentWords; ++I) {
    uint32_t Word;
    if (auto EC = Stream.readInteger(Word))
      return joinErrors(std::move(EC),
                        make_error<RawError>(raw_error_code::corrupt_file,
                                             "Expected name map word"));
    for (unsigned Idx = 0; Idx < 32; ++Idx)
      if (Word & (1U << Idx))
        Present.set((I * 32) + Idx);
  }

  // This appears to be a hash table which uses bitfields to determine whether
  // or not a bucket is 'deleted'.
  uint32_t NumDeletedWords;
  if (auto EC = Stream.readInteger(NumDeletedWords))
    return joinErrors(
        std::move(EC),
        make_error<RawError>(raw_error_code::corrupt_file,
                             "Expected name map num deleted words"));

  if (NumDeletedWords > MaxNumberOfWords)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Number of deleted words is too large");

  SparseBitVector<> Deleted;
  for (uint32_t I = 0; I != NumDeletedWords; ++I) {
    uint32_t Word;
    if (auto EC = Stream.readInteger(Word))
      return joinErrors(std::move(EC),
                        make_error<RawError>(raw_error_code::corrupt_file,
                                             "Expected name map word"));
    for (unsigned Idx = 0; Idx < 32; ++Idx)
      if (Word & (1U << Idx))
        Deleted.set((I * 32) + Idx);
  }

  for (unsigned I : Present) {
    // For all present entries, dump out their mapping.
    (void)I;

    // This appears to be an offset relative to the start of the strings.
    // It tells us where the null-terminated string begins.
    uint32_t NameOffset;
    if (auto EC = Stream.readInteger(NameOffset))
      return joinErrors(std::move(EC),
                        make_error<RawError>(raw_error_code::corrupt_file,
                                             "Expected name map name offset"));

    // This appears to be a stream number into the stream directory.
    uint32_t NameIndex;
    if (auto EC = Stream.readInteger(NameIndex))
      return joinErrors(std::move(EC),
                        make_error<RawError>(raw_error_code::corrupt_file,
                                             "Expected name map name index"));

    // Compute the offset of the start of the string relative to the stream.
    uint32_t StringOffset = StringsOffset + NameOffset;
    uint32_t OldOffset = Stream.getOffset();
    // Pump out our c-string from the stream.
    StringRef Str;
    Stream.setOffset(StringOffset);
    if (auto EC = Stream.readZeroString(Str))
      return joinErrors(std::move(EC),
                        make_error<RawError>(raw_error_code::corrupt_file,
                                             "Expected name map name"));

    Stream.setOffset(OldOffset);
    // Add this to a string-map from name to stream number.
    Mapping.insert({Str, NameIndex});
  }

  return Error::success();
}

Error NameMap::commit(codeview::StreamWriter &Writer) {
  // The first field is the number of bytes of string data.  So add
  // up the length of all strings plus a null terminator for each
  // one.
  uint32_t NumBytes = 0;
  for (auto B = Mapping.begin(), E = Mapping.end(); B != E; ++B) {
    NumBytes += B->getKeyLength() + 1;
  }

  if (auto EC = Writer.writeInteger(NumBytes)) // Number of bytes of string data
    return EC;
  // Now all of the string data itself.
  for (auto B = Mapping.begin(), E = Mapping.end(); B != E; ++B) {
    if (auto EC = Writer.writeZeroString(B->getKey()))
      return EC;
  }

  if (auto EC = Writer.writeInteger(Mapping.size())) // Hash Size
    return EC;

  if (auto EC = Writer.writeInteger(Mapping.size())) // Max Number of Strings
    return EC;

  if (auto EC = Writer.writeInteger(Mapping.size())) // Num Present Words
    return EC;

  // For each entry in the mapping, write a bit mask which represents a bucket
  // to store it in.  We don't use this, so the value we write isn't important
  // to us, it just has to be there.
  for (auto B = Mapping.begin(), E = Mapping.end(); B != E; ++B) {
    if (auto EC = Writer.writeInteger(1U))
      return EC;
  }

  if (auto EC = Writer.writeInteger(0U)) // Num Deleted Words
    return EC;

  // Mappings of each word.
  uint32_t OffsetSoFar = 0;
  for (auto B = Mapping.begin(), E = Mapping.end(); B != E; ++B) {
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

iterator_range<StringMapConstIterator<uint32_t>> NameMap::entries() const {
  return llvm::make_range<StringMapConstIterator<uint32_t>>(Mapping.begin(),
                                                            Mapping.end());
}

bool NameMap::tryGetValue(StringRef Name, uint32_t &Value) const {
  auto Iter = Mapping.find(Name);
  if (Iter == Mapping.end())
    return false;
  Value = Iter->second;
  return true;
}
