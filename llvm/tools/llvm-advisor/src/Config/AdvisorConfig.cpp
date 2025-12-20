//===------------------ AdvisorConfig.cpp - LLVM Advisor ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AdvisorConfig class.
//
//===----------------------------------------------------------------------===//

#include "AdvisorConfig.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

namespace llvm {
namespace advisor {

AdvisorConfig::AdvisorConfig() { outputDir = ".llvm-advisor"; }

Expected<bool> AdvisorConfig::loadFromFile(llvm::StringRef path) {
  auto BufferOrError = MemoryBuffer::getFile(path);
  if (!BufferOrError)
    return createStringError(BufferOrError.getError(),
                             "Cannot read config file");

  auto Buffer = std::move(*BufferOrError);
  Expected<json::Value> JsonOrError = json::parse(Buffer->getBuffer());
  if (!JsonOrError)
    return JsonOrError.takeError();

  auto &Json = *JsonOrError;
  auto *Obj = Json.getAsObject();
  if (!Obj)
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "Config file must contain JSON object");

  if (auto OutputDirOpt = Obj->getString("outputDir"); OutputDirOpt)
    outputDir = OutputDirOpt->str();

  if (auto VerboseOpt = Obj->getBoolean("verbose"); VerboseOpt)
    isVerbose = *VerboseOpt;

  if (auto KeepTempsOpt = Obj->getBoolean("keepTemps"); KeepTempsOpt)
    keepTemps = *KeepTempsOpt;

  if (auto RunProfileOpt = Obj->getBoolean("runProfiler"); RunProfileOpt)
    runProfiler = *RunProfileOpt;

  if (auto TimeoutOpt = Obj->getInteger("timeout"); TimeoutOpt)
    timeoutSeconds = static_cast<int>(*TimeoutOpt);

  // Optional per-tool path overrides. Example config:
  // { "tools": { "clang": "/usr/local/bin/clang" } }
  if (auto *ToolsObj = Obj->getObject("tools"); ToolsObj) {
    for (const auto &KV : *ToolsObj) {
      if (auto S = KV.getSecond().getAsString())
        toolPaths[StringRef(KV.getFirst()).str()] = S->str();
    }
  }

  return true;
}

std::string AdvisorConfig::getToolPath(llvm::StringRef tool) const {
  // First consult any explicit override from the configuration file.
  auto It = toolPaths.find(tool.str());
  if (It != toolPaths.end())
    return It->second;

  // Otherwise try to find the program in PATH.
  if (auto P = llvm::sys::findProgramByName(tool))
    return *P;

  // Fall back to the given tool name (let the OS PATH lookup handle it).
  return tool.str();
}

} // namespace advisor
} // namespace llvm
