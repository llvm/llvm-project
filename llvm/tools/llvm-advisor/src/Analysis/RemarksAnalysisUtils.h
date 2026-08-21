//===--- RemarksAnalysisUtils.h - LLVM Advisor ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Analysis/AnalyzerBase.h"
#include "llvm/Remarks/Remark.h"

namespace llvm::advisor {

using RemarkVisitor = function_ref<Error(const remarks::Remark &)>;

/// Open a YAML remark file and invoke Visitor for each remark.
Error foreachRemark(StringRef Path, RemarkVisitor Visitor);

/// Run Analyze over the remarks file discovered for Context.  If no
/// remarks artifact is found, returns an unavailable result.
template <typename Func>
Expected<std::unique_ptr<CapabilityResult>>
withRemarksFile(const CapabilityContext &Context, StringRef CapabilityID,
                StringRef UnitID, Func &&Analyze) {
  std::string Path = findRemarksPath(Context);
  if (Path.empty())
    return makeUnavailableResult(CapabilityID, UnitID, "missing remarks artifact");
  return std::forward<Func>(Analyze)(Path);
}

} // namespace llvm::advisor
