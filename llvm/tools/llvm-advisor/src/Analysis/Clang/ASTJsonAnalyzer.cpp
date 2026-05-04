//===--- ASTJsonAnalyzer.cpp - LLVM Advisor --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Clang/ASTJsonAnalyzer.h"
#include "Analysis/Clang/ClangAnalyzerUtils.h"

#include "clang/AST/Decl.h"
#include "clang/Basic/SourceManager.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
ASTJsonAnalyzer::run(const CapabilityContext &Context) {
  Expected<std::unique_ptr<clang::ASTUnit>> ASTOrErr = buildASTUnit(Context);
  if (!ASTOrErr)
    return ASTOrErr.takeError();

  clang::ASTContext &AST = (*ASTOrErr)->getASTContext();
  clang::TranslationUnitDecl *TU = AST.getTranslationUnitDecl();
  const clang::SourceManager &SM = AST.getSourceManager();

  json::Array DeclArray;
  for (const clang::Decl *D : TU->decls()) {
    json::Object Item;
    Item["kind"] = D->getDeclKindName();
    if (const auto *ND = llvm::dyn_cast<clang::NamedDecl>(D))
      Item["name"] = ND->getNameAsString();
    if (!D->getLocation().isValid()) {
      Item["implicit"] = true;
    } else {
      addPresumedLoc(Item, SM.getPresumedLoc(D->getLocation()));
    }
    DeclArray.push_back(std::move(Item));
  }

  return makeJSONResult(getCapabilityID(), Context.Unit.ID, json::Object{
      {"decl_count", static_cast<int64_t>(DeclArray.size())},
      {"decls", std::move(DeclArray)},
  });
}
