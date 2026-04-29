//===--- TimeTraceAnalyzer.cpp - LLVM Advisor ---------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "Analysis/Build/TimeTraceAnalyzer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

namespace llvm::advisor {

// Derive the .time-trace path from the object path or source path.
// Clang writes <output>.time-trace when given -ftime-trace.
static std::string findTimeTracePath(const CapabilityContext &Ctx) {
  auto Try = [](const std::string &Base) -> std::string {
    // <file>.time-trace
    std::string Candidate = Base + ".time-trace";
    if (sys::fs::exists(Candidate))
      return Candidate;
    // <stem>.time-trace in same directory
    SmallString<256> AltPath(sys::path::parent_path(Base));
    sys::path::append(AltPath, sys::path::stem(Base));
    AltPath += ".time-trace";
    if (sys::fs::exists(AltPath))
      return std::string(AltPath);
    return {};
  };

  if (!Ctx.ObjectPath.empty()) {
    std::string P = Try(Ctx.ObjectPath);
    if (!P.empty())
      return P;
  }
  if (!Ctx.SourcePath.empty()) {
    std::string P = Try(Ctx.SourcePath);
    if (!P.empty())
      return P;
  }
  return {};
}

Expected<std::unique_ptr<CapabilityResult>>
TimeTraceAnalyzer::run(const CapabilityContext &Context) {
  std::string TracePath = findTimeTracePath(Context);
  if (TracePath.empty())
    return createStringError(
        inconvertibleErrorCode(),
        "no .time-trace file found — compile with -ftime-trace to enable");

  ErrorOr<std::unique_ptr<MemoryBuffer>> Buf = MemoryBuffer::getFile(TracePath);
  if (!Buf)
    return createStringError(Buf.getError(), "cannot read %s",
                             TracePath.c_str());

  Expected<json::Value> Parsed = json::parse((*Buf)->getBuffer());
  if (!Parsed)
    return Parsed.takeError();

  // Validate: must have traceEvents array.
  const json::Object *Root = Parsed->getAsObject();
  if (!Root || !Root->getArray("traceEvents"))
    return createStringError(inconvertibleErrorCode(),
                             "invalid Chrome tracing format in %s",
                             TracePath.c_str());

  return std::make_unique<JSONCapabilityResult>(std::move(*Parsed));
}

} // namespace llvm::advisor
