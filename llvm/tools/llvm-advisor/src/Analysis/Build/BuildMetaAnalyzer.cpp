//===--- BuildMetaAnalyzer.cpp - LLVM Advisor -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of BuildMetaAnalyzer in Analysis::Build
//
//===----------------------------------------------------------------------===//

#include "Analysis/Build/BuildMetaAnalyzer.h"
#include "Utils/Redaction.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
BuildMetaAnalyzer::run(const CapabilityContext &Context) {
  json::Array Args;
  for (StringRef Arg : redactCommand(Context.Unit.Arguments))
    Args.push_back(Arg);

  json::Object Result;
  Result["capability"] = getCapabilityID();
  Result["unit_id"] = Context.Unit.ID;
  Result["source_path"] = Context.Unit.SourcePath;
  Result["directory"] = Context.Unit.Directory;
  Result["language"] = Context.Unit.Language;
  Result["target_triple"] = Context.Unit.TargetTriple;
  Result["arguments"] = std::move(Args);
  return std::make_unique<JSONCapabilityResult>(std::move(Result));
}
