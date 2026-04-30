//===--- RemarksAnalyzer.cpp - LLVM Advisor ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/IR/RemarksAnalyzer.h"
#include "Analysis/RemarksAnalysisUtils.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
RemarksAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withRemarksFile(
      Context, CapID, UnitID,
      [&](StringRef Path) -> Expected<std::unique_ptr<CapabilityResult>> {
        int64_t Count = 0;
        json::Object ByPass;
        json::Object ByType;
        if (Error E = foreachRemark(
                Path, [&](const remarks::Remark &R) -> Error {
                  ++Count;
                  std::string PassKey = R.PassName.str();
                  json::Value &PassVal = ByPass[PassKey];
                  PassVal = PassVal.getAsInteger().value_or(0) + 1;

                  StringRef Ty = remarks::typeToStr(R.RemarkType);
                  json::Value &TypeVal = ByType[Ty];
                  TypeVal = TypeVal.getAsInteger().value_or(0) + 1;
                  return Error::success();
                }))
          return std::move(E);

        return makeJSONResult(CapID, UnitID, json::Object{
            {"remarks_path", Path},
            {"remark_count", Count},
            {"by_pass", std::move(ByPass)},
            {"by_type", std::move(ByType)}});
      });
}
