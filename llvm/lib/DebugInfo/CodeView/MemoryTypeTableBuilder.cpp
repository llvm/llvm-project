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

MemoryTypeTableBuilder::Record::Record(StringRef RData)
    : Size(RData.size()), Data(new char[RData.size()]) {
  memcpy(Data.get(), RData.data(), RData.size());
}

TypeIndex MemoryTypeTableBuilder::writeRecord(StringRef Data) {
  auto I = HashedRecords.find(Data);
  if (I != HashedRecords.end()) {
    return I->second;
  }

  std::unique_ptr<Record> R(new Record(Data));

  TypeIndex TI(static_cast<uint32_t>(Records.size()) +
               TypeIndex::FirstNonSimpleIndex);
  HashedRecords.insert(std::make_pair(StringRef(R->data(), R->size()), TI));
  Records.push_back(std::move(R));

  return TI;
}
