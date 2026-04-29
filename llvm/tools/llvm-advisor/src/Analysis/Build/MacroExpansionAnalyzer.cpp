//===--- MacroExpansionAnalyzer.cpp - LLVM Advisor ----------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "Analysis/Build/MacroExpansionAnalyzer.h"
#include "Analysis/Clang/ClangAnalyzerUtils.h"
#include "clang/Lex/Preprocessor.h"

namespace llvm::advisor {
namespace {

constexpr int MaxMacrosInOutput = 200;

struct MacroEntry {
  std::string Name;
  bool IsFunctionLike;
  bool IsBuiltin;
  int NumTokens;
  int NumArgs;
};

} // namespace

Expected<std::unique_ptr<CapabilityResult>>
MacroExpansionAnalyzer::run(const CapabilityContext &Context) {
  Expected<std::unique_ptr<clang::ASTUnit>> ASTOrErr = buildASTUnit(Context);
  if (!ASTOrErr)
    return ASTOrErr.takeError();

  clang::ASTUnit &AST = **ASTOrErr;
  clang::Preprocessor &PP = AST.getPreprocessor();

  SmallVector<MacroEntry, 256> Entries;
  MacroStats Stats;

  for (auto It = PP.macro_begin(), End = PP.macro_end(); It != End; ++It) {
    const clang::MacroInfo *MI = getMacroInfo(It->second);
    if (!MI)
      continue;
    Stats.account(MI);
    MacroEntry E;
    E.Name = It->first->getName().str();
    E.IsFunctionLike = MI->isFunctionLike();
    E.IsBuiltin = MI->isBuiltinMacro();
    E.NumTokens = static_cast<int>(MI->getNumTokens());
    E.NumArgs = MI->isFunctionLike() ? static_cast<int>(MI->getNumParams()) : 0;
    Entries.push_back(std::move(E));
  }

  // Sort by token count descending — large macros are the most interesting.
  llvm::sort(Entries, [](const MacroEntry &A, const MacroEntry &B) {
    return A.NumTokens > B.NumTokens;
  });

  json::Array MacroArray;
  int Count = 0;
  for (const MacroEntry &E : Entries) {
    if (E.IsBuiltin)
      continue; // Skip builtins from the detail list.
    if (Count++ >= MaxMacrosInOutput)
      break;
    MacroArray.push_back(json::Object{
        {"name", E.Name},
        {"function_like", E.IsFunctionLike},
        {"num_tokens", E.NumTokens},
        {"num_args", E.NumArgs},
    });
  }

  json::Value Result = json::Object{
      {"total_macros", Stats.Total},
      {"function_like_macros", Stats.FunctionLike},
      {"object_like_macros", Stats.ObjectLike},
      {"builtin_macros", Stats.Builtin},
      {"user_macros", Stats.FunctionLike + Stats.ObjectLike},
      {"top_by_expansion_size", std::move(MacroArray)},
  };
  return std::make_unique<JSONCapabilityResult>(std::move(Result));
}

} // namespace llvm::advisor
