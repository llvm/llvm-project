//===--- PreprocessorAnalyzer.cpp - LLVM Advisor ------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "Analysis/Build/PreprocessorAnalyzer.h"
#include "Analysis/Clang/ClangAnalyzerUtils.h"
#include "clang/Lex/Preprocessor.h"

namespace llvm::advisor {

Expected<std::unique_ptr<CapabilityResult>>
PreprocessorAnalyzer::run(const CapabilityContext &Context) {
  Expected<std::unique_ptr<clang::ASTUnit>> ASTOrErr = buildASTUnit(Context);
  if (!ASTOrErr)
    return ASTOrErr.takeError();

  clang::ASTUnit &AST = **ASTOrErr;
  const clang::SourceManager &SM = AST.getSourceManager();
  clang::Preprocessor &PP = AST.getPreprocessor();

  // Count unique included files (excluding main file).
  int64_t IncludeCount = 0;
  forEachIncludedFile(SM, [&](const clang::FileEntry &, clang::FileID) {
    ++IncludeCount;
  });

  // Count macro definitions visible at end of translation unit.
  MacroStats Stats;
  for (auto It = PP.macro_begin(), End = PP.macro_end(); It != End; ++It)
    Stats.account(getMacroInfo(It->second));

  // Derive line count from the main file buffer. Empty files report 0 lines;
  // otherwise count newline characters and add one for the final line if it
  // lacks a trailing newline.
  int64_t Lines = 0;
  if (auto MainBuf = SM.getBufferOrNone(SM.getMainFileID())) {
    StringRef Buf = MainBuf->getBuffer();
    if (!Buf.empty()) {
      Lines = static_cast<int64_t>(Buf.count('\n'));
      if (!Buf.ends_with("\n"))
        ++Lines;
    }
  }

  json::Value Result = json::Object{
      {"included_files", IncludeCount},
      {"source_lines", Lines},
      {"total_macros", Stats.Total},
      {"function_like_macros", Stats.FunctionLike},
      {"object_like_macros", Stats.ObjectLike},
      {"builtin_macros", Stats.Builtin},
      {"user_defined_macros", Stats.FunctionLike + Stats.ObjectLike},
  };
  return std::make_unique<JSONCapabilityResult>(std::move(Result));
}

} // namespace llvm::advisor
