//===------------------- SancovIngestor.cpp - LLVM Advisor -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of SancovIngestor in Runtime
//
//===----------------------------------------------------------------------===//
#include "Runtime/SancovIngestor.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<json::Value> SancovIngestor::load(StringRef Path) {
  if (Path.empty())
    return createStringError(inconvertibleErrorCode(), "empty sancov path");
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer = MemoryBuffer::getFile(Path);
  if (!Buffer)
    return createStringError(Buffer.getError(), "cannot read sancov '%s'",
                             Path.data());

  StringRef Data = (*Buffer)->getBuffer();
  json::Array PCs;
  uint64_t Count = 0;

  if (Data.bytes_begin() != Data.bytes_end() && Data.contains('\n')) {
    for (line_iterator Line(**Buffer, false); !Line.is_at_eof(); ++Line) {
      StringRef Text = (*Line).trim();
      if (Text.empty())
        continue;
      uint64_t PC = 0;
      Text.consume_front("0x");
      (void)Text.getAsInteger(16, PC);
      if (PCs.size() < 1024)
        PCs.push_back(("0x" + Twine::utohexstr(PC)).str());
      ++Count;
    }
  } else {
    Count = Data.size() / sizeof(uint64_t);
  }

  return json::Object{
      {"kind", "sancov-points"},
      {"format", Data.contains('\n') ? "sancov-text" : "sancov-binary"},
      {"path", Path},
      {"point_count", static_cast<int64_t>(Count)},
      {"pcs", std::move(PCs)}};
}
