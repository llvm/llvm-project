//===- Formatters.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/Formatters.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/CodeView/GUID.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::codeview::detail;

GuidAdapter::GuidAdapter(StringRef Guid)
    : FormatAdapter(makeArrayRef(Guid.bytes_begin(), Guid.bytes_end())) {}

GuidAdapter::GuidAdapter(ArrayRef<uint8_t> Guid)
    : FormatAdapter(std::move(Guid)) {}

void GuidAdapter::format(raw_ostream &Stream, StringRef Style) {
  static const char *Lookup = "0123456789ABCDEF";

  assert(Item.size() == 16 && "Expected 16-byte GUID");
  Stream << "{";
  for (int i = 0; i < 16;) {
    uint8_t Byte = Item[i];
    uint8_t HighNibble = (Byte >> 4) & 0xF;
    uint8_t LowNibble = Byte & 0xF;
    Stream << Lookup[HighNibble] << Lookup[LowNibble];
    ++i;
    if (i >= 4 && i <= 10 && i % 2 == 0)
      Stream << "-";
  }
  Stream << "}";
}

raw_ostream &llvm::codeview::operator<<(raw_ostream &OS, const GUID &Guid) {
  codeview::detail::GuidAdapter A(Guid.Guid);
  A.format(OS, "");
  return OS;
}
