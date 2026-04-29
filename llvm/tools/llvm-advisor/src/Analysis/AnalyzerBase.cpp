//===--- AnalyzerBase.cpp - LLVM Advisor ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of AnalyzerBase in Analysis
//
//===----------------------------------------------------------------------===//

#include "Analysis/AnalyzerBase.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::advisor;

std::string llvm::advisor::findRemarksPath(const CapabilityContext &Context) {
  // 0. Use path synthesized by CaptureCore (in-process or pre-discovered).
  if (!Context.RemarksPath.empty() && sys::fs::exists(Context.RemarksPath))
    return Context.RemarksPath;

  // 1. Source-adjacent: replace extension in-place (cmake -save-temps style).
  SmallString<256> FromSource(Context.SourcePath);
  sys::path::replace_extension(FromSource, "opt.yaml");
  if (sys::fs::exists(FromSource))
    return std::string(FromSource);
  sys::path::replace_extension(FromSource, "opt.json");
  if (sys::fs::exists(FromSource))
    return std::string(FromSource);

  if (!Context.WorkingDirectory.empty()) {
    StringRef Stem = sys::path::stem(Context.SourcePath);

    // 2. Build-dir/<stem>.opt.yaml — the most common clang layout.
    for (StringRef Ext : {"opt.yaml", "opt.json"}) {
      SmallString<256> P(Context.WorkingDirectory);
      sys::path::append(P, Stem);
      sys::path::replace_extension(P, Ext);
      if (sys::fs::exists(P))
        return std::string(P);
    }

    // 3. Build-dir/remarks.yaml / remarks.opt.yaml — project-wide aggregates.
    for (StringRef Alt : {"remarks.yaml", "remarks.opt.yaml"}) {
      SmallString<256> P(Context.WorkingDirectory);
      sys::path::append(P, Alt);
      if (sys::fs::exists(P))
        return std::string(P);
    }
  }
  return {};
}

// SimpleAnalyzer serves as a base for capabilities that have no dedicated
// analyzer implementation yet, reporting unavailable until one is provided.
Expected<std::unique_ptr<CapabilityResult>>
SimpleAnalyzer::run(const CapabilityContext &Context) {
  json::Object Result;
  Result["capability"] = CapabilityID;
  Result["unit_id"] = Context.Unit.ID;
  Result["source_path"] = Context.Unit.SourcePath;
  Result["summary"] = Summary;
  Result["available"] = false;
  return std::make_unique<JSONCapabilityResult>(std::move(Result));
}
