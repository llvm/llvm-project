//===--- SARIFDiagnosticsAnalyzer.cpp - LLVM Advisor -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Clang/SARIFDiagnosticsAnalyzer.h"
#include "Analysis/Clang/ClangAnalyzerUtils.h"

#include "clang/Basic/Diagnostic.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
SARIFDiagnosticsAnalyzer::run(const CapabilityContext &Context) {
  Expected<std::unique_ptr<clang::ASTUnit>> ASTOrErr = buildASTUnit(Context);
  if (!ASTOrErr)
    return ASTOrErr.takeError();

  json::Array SarifResults;
  clang::ASTUnit &AST = **ASTOrErr;
  for (auto It = AST.stored_diag_begin(); It != AST.stored_diag_end(); ++It) {
    const clang::StoredDiagnostic &Diag = *It;
    StringRef SarifLevel;
    switch (Diag.getLevel()) {
    case clang::DiagnosticsEngine::Fatal:
    case clang::DiagnosticsEngine::Error:
      SarifLevel = "error";
      break;
    case clang::DiagnosticsEngine::Warning:
      SarifLevel = "warning";
      break;
    case clang::DiagnosticsEngine::Note:
      SarifLevel = "note";
      break;
    case clang::DiagnosticsEngine::Ignored:
    case clang::DiagnosticsEngine::Remark:
      continue;
    }

    json::Object Result;
    Result["message"] = json::Object{{"text", Diag.getMessage().str()}};
    Result["level"] = SarifLevel;
    if (Diag.getLocation().isValid()) {
      clang::PresumedLoc PLoc =
          Diag.getLocation().getManager().getPresumedLoc(Diag.getLocation());
      if (PLoc.isValid()) {
        json::Object Region;
        Region["startLine"] = static_cast<int64_t>(PLoc.getLine());
        Region["startColumn"] = static_cast<int64_t>(PLoc.getColumn());
        Result["locations"] = json::Array{json::Object{
            {"physicalLocation",
             json::Object{
                 {"artifactLocation",
                  json::Object{{"uri", std::string(PLoc.getFilename())}}},
                 {"region", std::move(Region)}}}}};
      }
    }
    SarifResults.push_back(std::move(Result));
  }

  json::Object ToolDriver{{"name", "llvm-advisor"}};
  json::Object Tool{{"driver", std::move(ToolDriver)}};
  json::Object Run{{"tool", std::move(Tool)},
                   {"results", std::move(SarifResults)}};
  json::Object Sarif{{"version", "2.1.0"},
                     {"runs", json::Array{std::move(Run)}}};

  return makeJSONResult(getCapabilityID(), Context.Unit.ID, json::Object{
      {"sarif_json", std::move(Sarif)},
  });
}
