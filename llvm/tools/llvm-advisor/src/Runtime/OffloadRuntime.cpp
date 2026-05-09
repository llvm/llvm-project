//===------------------- OffloadRuntime.cpp - LLVM Advisor ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of OffloadRuntime in Runtime
//
//===----------------------------------------------------------------------===//
#include "Runtime/OffloadRuntime.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::advisor;

static StringRef field(ArrayRef<StringRef> Fields, unsigned Index) {
  if (Index >= Fields.size())
    return "";
  return Fields[Index].trim();
}

Expected<json::Value> OffloadRuntime::load(StringRef Path) {
  if (Path.empty())
    return createStringError(inconvertibleErrorCode(),
                             "empty offload trace path");
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer = MemoryBuffer::getFile(Path);
  if (!Buffer)
    return createStringError(Buffer.getError(),
                             "cannot read offload trace '%s'",
                             Path.data());

  json::Array Kernels;
  json::Array Transfers;
  uint64_t KernelCount = 0;
  uint64_t TransferCount = 0;
  uint64_t SyncCount = 0;

  for (line_iterator Line(**Buffer, false); !Line.is_at_eof(); ++Line) {
    StringRef Text = (*Line).trim();
    if (Text.empty() || Text.starts_with("#"))
      continue;
    SmallVector<StringRef, 8> Fields;
    Text.split(Fields, ',');
    StringRef Type = field(Fields, 0);
    if (Type.equals_insensitive("kernel")) {
      ++KernelCount;
      if (Kernels.size() < 512) {
        int64_t Dur = 0;
        field(Fields, 2).getAsInteger(10, Dur);
        Kernels.push_back(json::Object{{"kernel", field(Fields, 1)},
                                       {"duration_ns", Dur},
                                       {"queue", field(Fields, 3)}});
      }
      continue;
    }
    if (Type.equals_insensitive("transfer")) {
      ++TransferCount;
      if (Transfers.size() < 512) {
        int64_t Bytes = 0, Dur = 0;
        field(Fields, 2).getAsInteger(10, Bytes);
        field(Fields, 3).getAsInteger(10, Dur);
        Transfers.push_back(
            json::Object{{"direction", field(Fields, 1)},
                         {"bytes", Bytes},
                         {"duration_ns", Dur}});
      }
      continue;
    }
    if (Type.equals_insensitive("sync"))
      ++SyncCount;
  }

  return json::Object{{"kind", "offload-trace"},
                      {"format", "csv"},
                      {"path", Path},
                      {"kernel_count", static_cast<int64_t>(KernelCount)},
                      {"transfer_count", static_cast<int64_t>(TransferCount)},
                      {"sync_count", static_cast<int64_t>(SyncCount)},
                      {"kernels", std::move(Kernels)},
                      {"transfers", std::move(Transfers)}};
}
