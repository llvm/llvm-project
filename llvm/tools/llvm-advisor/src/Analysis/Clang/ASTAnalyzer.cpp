//===--- ASTAnalyzer.cpp - LLVM Advisor ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Clang/ASTAnalyzer.h"
#include "Analysis/Clang/ClangAnalyzerUtils.h"

#include "clang/AST/Decl.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
ASTAnalyzer::run(const CapabilityContext &Context) {
  Expected<std::unique_ptr<clang::ASTUnit>> ASTOrErr = buildASTUnit(Context);
  if (!ASTOrErr)
    return ASTOrErr.takeError();

  clang::TranslationUnitDecl *TU =
      (*ASTOrErr)->getASTContext().getTranslationUnitDecl();

  struct Counter {
    int64_t NumFunctions = 0;
    int64_t NumCxxRecords = 0;
    int64_t NumTemplates = 0;

    void count(const clang::Decl *D) {
      if (llvm::isa<clang::FunctionDecl>(D))
        ++NumFunctions;
      if (llvm::isa<clang::CXXRecordDecl>(D))
        ++NumCxxRecords;
      if (llvm::isa<clang::ClassTemplateDecl>(D) ||
          llvm::isa<clang::FunctionTemplateDecl>(D))
        ++NumTemplates;
      if (const auto *ND = llvm::dyn_cast<clang::NamespaceDecl>(D))
        for (const clang::Decl *Inner : ND->decls())
          count(Inner);
      if (const auto *LD = llvm::dyn_cast<clang::LinkageSpecDecl>(D))
        for (const clang::Decl *Inner : LD->decls())
          count(Inner);
    }
  };

  Counter C;
  for (const clang::Decl *D : TU->decls())
    C.count(D);

  return makeJSONResult(getCapabilityID(), Context.Unit.ID, json::Object{
      {"functions", C.NumFunctions},
      {"cxx_records", C.NumCxxRecords},
      {"templates", C.NumTemplates},
  });
}
