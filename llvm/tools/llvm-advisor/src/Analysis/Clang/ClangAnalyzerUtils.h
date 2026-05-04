//===--- ClangAnalyzerUtils.h - LLVM Advisor ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for clang-based analyzers.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Analysis/AnalyzerBase.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Lex/Preprocessor.h"

namespace llvm::advisor {

// Returns the unit's compile args with output/dependency flags stripped,
// ready to be augmented with an action-specific suffix.
Expected<SmallVector<std::string, 32>>
buildBaseClangArgs(const CapabilityContext &Context);

// buildBaseClangArgs + -fsyntax-only + any ExtraArgs.
Expected<SmallVector<std::string, 32>>
buildClangArgs(const CapabilityContext &Context,
               ArrayRef<StringRef> ExtraArgs = {});

Expected<std::unique_ptr<clang::ASTUnit>>
buildASTUnit(const CapabilityContext &Context,
             ArrayRef<StringRef> ExtraArgs = {});

// Emit LLVM IR to OutPath by replaying the unit's compiler invocation.
Expected<std::string> emitLLVMIR(const CapabilityContext &Context,
                                 StringRef OutPath);

// Replay the unit's compiler invocation with optimization-record flags,
// writing remarks to OutPath. Returns OutPath on success.
Expected<std::string> emitOptRemarks(const CapabilityContext &Context,
                                     StringRef OutPath);

// Replay the unit's compiler invocation to emit assembly (.s) to OutPath.
Expected<std::string> emitAssembly(const CapabilityContext &Context,
                                   StringRef OutPath);

// Replay the unit's compiler invocation to emit an object file to OutPath.
Expected<std::string> emitObject(const CapabilityContext &Context,
                                 StringRef OutPath);

// Returns the MacroInfo for the latest macro definition, or nullptr.
template <typename MacroStateT>
inline const clang::MacroInfo *getMacroInfo(MacroStateT &&State) {
  auto *Latest = State.getLatest();
  return Latest ? Latest->getMacroInfo() : nullptr;
}

// Accumulates macro statistics across a translation unit.
struct MacroStats {
  int64_t Total = 0;
  int64_t Builtin = 0;
  int64_t FunctionLike = 0;
  int64_t ObjectLike = 0;

  void account(const clang::MacroInfo *MI) {
    if (!MI)
      return;
    ++Total;
    if (MI->isBuiltinMacro())
      ++Builtin;
    else if (MI->isFunctionLike())
      ++FunctionLike;
    else
      ++ObjectLike;
  }
};

// Iterates every file included by the translation unit (excluding the main
// file) and invokes Callback with the FileEntry and FileID.
template <typename Callback>
void forEachIncludedFile(const clang::SourceManager &SM, Callback &&CB) {
  for (auto It = SM.fileinfo_begin(), End = SM.fileinfo_end(); It != End;
       ++It) {
    const clang::FileEntry *FE = It->getFirst();
    if (!FE)
      continue;
    clang::FileID FID = SM.translateFile(FE);
    if (!FID.isValid() || FID == SM.getMainFileID())
      continue;
    CB(*FE, FID);
  }
}

// Append PresumedLoc fields (file, line, column) to Obj if Loc is valid.
inline void addPresumedLoc(json::Object &Obj, const clang::PresumedLoc &Loc,
                           StringRef FileKey = "file",
                           StringRef LineKey = "line",
                           StringRef ColKey = "column") {
  if (!Loc.isValid())
    return;
  Obj[FileKey] = std::string(Loc.getFilename());
  Obj[LineKey] = static_cast<int64_t>(Loc.getLine());
  Obj[ColKey] = static_cast<int64_t>(Loc.getColumn());
}

} // namespace llvm::advisor
