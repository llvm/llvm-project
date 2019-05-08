//===- RemarkStringTable.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the Remark string table used at remark generation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Remarks/RemarkStringTable.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/Error.h"
#include <vector>

using namespace llvm;
using namespace llvm::remarks;

std::pair<unsigned, StringRef> StringTable::add(StringRef Str) {
  size_t NextID = StrTab.size();
  auto KV = StrTab.insert({Str, NextID});
  // If it's a new string, add it to the final size.
  if (KV.second)
    SerializedSize += KV.first->first().size() + 1; // +1 for the '\0'
  // Can be either NextID or the previous ID if the string is already there.
  return {KV.first->second, KV.first->first()};
}

void StringTable::serialize(raw_ostream &OS) const {
  // Emit the number of strings.
  uint64_t StrTabSize = SerializedSize;
  support::endian::write(OS, StrTabSize, support::little);
  // Emit the sequence of strings.
  for (StringRef Str : serialize()) {
    OS << Str;
    // Explicitly emit a '\0'.
    OS.write('\0');
  }
}

std::vector<StringRef> StringTable::serialize() const {
  std::vector<StringRef> Strings{StrTab.size()};
  for (const auto &KV : StrTab)
    Strings[KV.second] = KV.first();
  return Strings;
}
