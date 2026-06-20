//===--- RuntimeRunner.cpp - LLVM Advisor --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of RuntimeRunner in Analysis
//
//===----------------------------------------------------------------------===//

#include "Analysis/Runtime/RuntimeRunner.h"
#include "Runtime/CoverageIngestor.h"
#include "Runtime/MemProfIngestor.h"
#include "Runtime/PGOInstrIngestor.h"
#include "Runtime/PGOSampleIngestor.h"
#include "Runtime/SancovIngestor.h"
#include "Runtime/SanitizerIngestor.h"
#include "Runtime/XRayIngestor.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::advisor;

static std::string findFileByPattern(StringRef Dir, StringRef Pattern) {
  if (Dir.empty())
    return {};
  std::error_code EC;
  for (sys::fs::directory_iterator It(Dir, EC), End; It != End;
       It.increment(EC)) {
    if (EC)
      break;
    StringRef Name = sys::path::filename(It->path());
    if (Name.contains(Pattern))
      return It->path();
  }
  return {};
}

static std::string findExactFile(StringRef Dir, StringRef Name) {
  if (Dir.empty())
    return {};
  std::string Path = (Dir + "/" + Name).str();
  if (sys::fs::exists(Path))
    return Path;
  return {};
}

static Expected<std::unique_ptr<CapabilityResult>>
runIngestor(const CapabilityContext &Context, StringRef CapID,
            function_ref<Expected<json::Value>(StringRef)> Load,
            ArrayRef<StringRef> ExactNames, ArrayRef<StringRef> Patterns) {
  std::string Path;
  for (StringRef Name : ExactNames) {
    Path = findExactFile(Context.WorkingDirectory, Name);
    if (!Path.empty())
      break;
  }
  if (Path.empty()) {
    for (StringRef Pat : Patterns) {
      Path = findFileByPattern(Context.WorkingDirectory, Pat);
      if (!Path.empty())
        break;
    }
  }
  if (Path.empty())
    return makeUnavailableResult(
        CapID, Context.Unit.ID,
        "no runtime artifact found in working directory");

  Expected<json::Value> Result = Load(Path);
  if (!Result)
    return Result.takeError();

  if (json::Object *Obj = Result->getAsObject()) {
    (*Obj)["capability"] = CapID;
    (*Obj)["unit_id"] = Context.Unit.ID;
    (*Obj)["available"] = true;
    return std::make_unique<JSONCapabilityResult>(std::move(*Result));
  }

  return createStringError(inconvertibleErrorCode(),
                           "ingestor did not return a JSON object");
}

Expected<std::unique_ptr<CapabilityResult>>
PGOInstrRunner::run(const CapabilityContext &Context) {
  return runIngestor(Context, getCapabilityID(),
                     [](StringRef P) { return PGOInstrIngestor().load(P); },
                     {"default.profdata"}, {"profdata"});
}

Expected<std::unique_ptr<CapabilityResult>>
PGOSampleRunner::run(const CapabilityContext &Context) {
  return runIngestor(Context, getCapabilityID(),
                     [](StringRef P) { return PGOSampleIngestor().load(P); },
                     {}, {"profraw", "profdata"});
}

Expected<std::unique_ptr<CapabilityResult>>
MemProfRunner::run(const CapabilityContext &Context) {
  return runIngestor(Context, getCapabilityID(),
                     [](StringRef P) { return MemProfIngestor().load(P); }, {},
                     {"memprof"});
}

Expected<std::unique_ptr<CapabilityResult>>
CoverageRunner::run(const CapabilityContext &Context) {
  return runIngestor(Context, getCapabilityID(),
                     [](StringRef P) { return CoverageIngestor().load(P); },
                     {"coverage.json"}, {"coverage"});
}

Expected<std::unique_ptr<CapabilityResult>>
XRayRunner::run(const CapabilityContext &Context) {
  return runIngestor(Context, getCapabilityID(),
                     [](StringRef P) { return XRayIngestor().load(P); }, {},
                     {"xray"});
}

Expected<std::unique_ptr<CapabilityResult>>
SancovRunner::run(const CapabilityContext &Context) {
  return runIngestor(Context, getCapabilityID(),
                     [](StringRef P) { return SancovIngestor().load(P); }, {},
                     {"sancov"});
}

Expected<std::unique_ptr<CapabilityResult>>
SanitizerRunner::run(const CapabilityContext &Context) {
  return runIngestor(Context, CapabilityID,
                     [](StringRef P) { return SanitizerIngestor().load(P); },
                     {}, {Pattern});
}
