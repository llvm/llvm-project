//===--- TraceDiscovery.h - LLVM Advisor ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

namespace llvm::advisor {

/// Look for a Chrome-format trace JSON file in the working directory.
/// Checks well-known names first, then scans for any .json containing
/// "traceEvents".
inline std::string findTraceJSON(StringRef WorkingDirectory) {
  if (WorkingDirectory.empty())
    return {};

  for (StringRef Name : {"results.json", "hip_trace.json", "rocprof.json",
                         "trace.json", "sys_trace.json",
                         "device_trace.json", "sync_trace.json"}) {
    SmallString<256> P(WorkingDirectory);
    sys::path::append(P, Name);
    if (sys::fs::exists(P))
      return P.str().str();
  }

  std::error_code EC;
  for (sys::fs::directory_iterator It(WorkingDirectory, EC), End;
       !EC && It != End; It.increment(EC)) {
    StringRef P = It->path();
    if (!P.ends_with(".json"))
      continue;
    ErrorOr<std::unique_ptr<MemoryBuffer>> MB = MemoryBuffer::getFile(P);
    if (MB && (*MB)->getBuffer().contains("traceEvents"))
      return P.str();
  }
  return {};
}

} // namespace llvm::advisor
