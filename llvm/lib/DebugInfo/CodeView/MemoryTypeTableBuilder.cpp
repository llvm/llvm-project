//===-- MemoryTypeTableBuilder.cpp ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/MemoryTypeTableBuilder.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"

using namespace llvm;
using namespace codeview;

TypeIndex MemoryTypeTableBuilder::writeRecord(StringRef Data) {
  assert(Data.size() <= UINT16_MAX);
  auto I = HashedRecords.find(Data);
  if (I != HashedRecords.end()) {
    return I->second;
  }

  // The record provided by the user lacks the 2 byte size field prefix and is
  // not padded to 4 bytes. Ultimately, that is what gets emitted in the object
  // file, so pad it out now.
  const int SizeOfRecLen = 2;
  const int Align = 4;
  int TotalSize = alignTo(Data.size() + SizeOfRecLen, Align);
  assert(TotalSize - SizeOfRecLen <= UINT16_MAX);
  char *Mem =
      reinterpret_cast<char *>(RecordStorage.Allocate(TotalSize, Align));
  *reinterpret_cast<ulittle16_t *>(Mem) = uint16_t(TotalSize - SizeOfRecLen);
  memcpy(Mem + SizeOfRecLen, Data.data(), Data.size());
  for (int I = Data.size() + SizeOfRecLen; I < TotalSize; ++I)
    Mem[I] = LF_PAD0 + (TotalSize - I);

  TypeIndex TI(static_cast<uint32_t>(Records.size()) +
               TypeIndex::FirstNonSimpleIndex);

  // Use only the data supplied by the user as a key to the hash table, so that
  // future lookups will succeed.
  HashedRecords.insert(std::make_pair(StringRef(Mem + SizeOfRecLen, Data.size()), TI));
  Records.push_back(StringRef(Mem, TotalSize));

  return TI;
}
