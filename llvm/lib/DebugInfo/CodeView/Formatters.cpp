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
  assert(Item.size() == 16 && "Expected 16-byte GUID");
  struct MSGuid {
    support::ulittle32_t Data1;
    support::ulittle16_t Data2;
    support::ulittle16_t Data3;
    support::ubig64_t Data4;
  };
  const MSGuid *G = reinterpret_cast<const MSGuid *>(Item.data());
  Stream
      << '{' << format_hex_no_prefix(G->Data1, sizeof(G->Data1), /*Upper=*/true)
      << '-' << format_hex_no_prefix(G->Data2, sizeof(G->Data2), /*Upper=*/true)
      << '-' << format_hex_no_prefix(G->Data3, sizeof(G->Data3), /*Upper=*/true)
      << '-' << format_hex_no_prefix(G->Data4 >> 48, 2, /*Upper=*/true) << '-'
      << format_hex_no_prefix(G->Data4 & ((1ULL << 48) - 1), 6, /*Upper=*/true)
      << '}';
}

raw_ostream &llvm::codeview::operator<<(raw_ostream &OS, const GUID &Guid) {
  codeview::detail::GuidAdapter A(Guid.Guid);
  A.format(OS, "");
  return OS;
}
