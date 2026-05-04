//===--- DiagnosticsAnalyzer.cpp - LLVM Advisor ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Clang/DiagnosticsAnalyzer.h"
#include "Analysis/Clang/ClangAnalyzerUtils.h"

#include "clang/Basic/Diagnostic.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
DiagnosticsAnalyzer::run(const CapabilityContext &Context) {
  Expected<std::unique_ptr<clang::ASTUnit>> ASTOrErr = buildASTUnit(Context);
  if (!ASTOrErr)
    return ASTOrErr.takeError();

  json::Array Diagnostics;
  int64_t NumErrors = 0;
  int64_t NumWarnings = 0;
  int64_t NumNotes = 0;
  int64_t NumRemarks = 0;

  clang::ASTUnit &AST = **ASTOrErr;
  for (auto It = AST.stored_diag_begin(); It != AST.stored_diag_end(); ++It) {
    const clang::StoredDiagnostic &Diag = *It;
    StringRef Level = "remark";
    switch (Diag.getLevel()) {
    case clang::DiagnosticsEngine::Fatal:
    case clang::DiagnosticsEngine::Error:
      Level = "error";
      ++NumErrors;
      break;
    case clang::DiagnosticsEngine::Warning:
      Level = "warning";
      ++NumWarnings;
      break;
    case clang::DiagnosticsEngine::Note:
      Level = "note";
      ++NumNotes;
      break;
    case clang::DiagnosticsEngine::Ignored:
    case clang::DiagnosticsEngine::Remark:
      ++NumRemarks;
      break;
    }

    json::Object Item;
    Item["level"] = Level;
    Item["message"] = Diag.getMessage().str();
    if (Diag.getLocation().isValid())
      addPresumedLoc(Item,
                     Diag.getLocation().getManager().getPresumedLoc(
                         Diag.getLocation()));
    Diagnostics.push_back(std::move(Item));
  }

  return makeJSONResult(getCapabilityID(), Context.Unit.ID, json::Object{
      {"errors", NumErrors},
      {"warnings", NumWarnings},
      {"notes", NumNotes},
      {"remarks", NumRemarks},
      {"diagnostics", std::move(Diagnostics)},
  });
}
