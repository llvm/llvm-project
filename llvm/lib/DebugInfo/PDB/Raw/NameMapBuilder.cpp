//===- NameMapBuilder.cpp - PDB Name Map Builder ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/NameMapBuilder.h"

#include "llvm/DebugInfo/PDB/Raw/NameMap.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::pdb;

NameMapBuilder::NameMapBuilder() {}

void NameMapBuilder::addMapping(StringRef Name, uint32_t Mapping) {
  StringDataBytes += Name.size() + 1;
  Map.insert({Name, Mapping});
}

Expected<std::unique_ptr<NameMap>> NameMapBuilder::build() {
  auto Result = llvm::make_unique<NameMap>();
  Result->Mapping = Map;
  return std::move(Result);
}

uint32_t NameMapBuilder::calculateSerializedLength() const {
  uint32_t TotalLength = 0;

  TotalLength += sizeof(support::ulittle32_t); // StringDataBytes value
  TotalLength += StringDataBytes;              // actual string data

  TotalLength += sizeof(support::ulittle32_t); // Hash Size
  TotalLength += sizeof(support::ulittle32_t); // Max Number of Strings
  TotalLength += sizeof(support::ulittle32_t); // Num Present Words
  // One bitmask word for each present entry
  TotalLength += Map.size() * sizeof(support::ulittle32_t);
  TotalLength += sizeof(support::ulittle32_t); // Num Deleted Words

  // For each present word, which we are treating as equivalent to the number of
  // entries in the table, we have a pair of integers.  An offset into the
  // string data, and a corresponding stream number.
  TotalLength += Map.size() * 2 * sizeof(support::ulittle32_t);

  return TotalLength;
}
