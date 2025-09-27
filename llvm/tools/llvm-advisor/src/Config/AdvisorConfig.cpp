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

namespace llvm {
namespace advisor {

AdvisorConfig::AdvisorConfig() { OutputDir_ = ".llvm-advisor"; }

Expected<bool> AdvisorConfig::loadFromFile(llvm::StringRef path) {
  auto BufferOrError = MemoryBuffer::getFile(path);
  if (!BufferOrError) {
    return createStringError(BufferOrError.getError(),
                             "Cannot read config file");
  }

  auto Buffer = std::move(*BufferOrError);
  Expected<json::Value> JsonOrError = json::parse(Buffer->getBuffer());
  if (!JsonOrError) {
    return JsonOrError.takeError();
  }

  auto &Json = *JsonOrError;
  auto *Obj = Json.getAsObject();
  if (!Obj) {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "Config file must contain JSON object");
  }

  if (auto outputDirOpt = Obj->getString("outputDir"); outputDirOpt) {
    OutputDir_ = outputDirOpt->str();
  }

  if (auto verboseOpt = Obj->getBoolean("verbose"); verboseOpt) {
    Verbose_ = *verboseOpt;
  }

  if (auto keepTempsOpt = Obj->getBoolean("keepTemps"); keepTempsOpt) {
    KeepTemps_ = *keepTempsOpt;
  }

  if (auto runProfileOpt = Obj->getBoolean("runProfiler"); runProfileOpt) {
    RunProfiler_ = *runProfileOpt;
  }

  if (auto timeoutOpt = Obj->getInteger("timeout"); timeoutOpt) {
    TimeoutSeconds_ = static_cast<int>(*timeoutOpt);
  }

  return true;
}

std::string AdvisorConfig::getToolPath(llvm::StringRef tool) const {
  return tool.str();
}

} // namespace advisor
} // namespace llvm
