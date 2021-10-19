//===------ DumpModulePass.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Write a module to a file.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SUPPORT_DUMPMODULEPASS_H
#define POLLY_SUPPORT_DUMPMODULEPASS_H

#include "llvm/IR/PassManager.h"
#include <string>

namespace llvm {
class ModulePass;
} // namespace llvm

namespace polly {
/// Create a pass that prints the module into a file.
///
/// The meaning of @p Filename depends on @p IsSuffix. If IsSuffix==false, then
/// the module is written to the @p Filename. If it is true, the filename is
/// generated from the module's name, @p Filename with an '.ll' extension.
///
/// The intent of IsSuffix is to avoid the file being overwritten when
/// processing multiple modules and/or with multiple dump passes in the
/// pipeline.
llvm::ModulePass *createDumpModuleWrapperPass(std::string Filename,
                                              bool IsSuffix);

/// A pass that prints the module into a file.
struct DumpModulePass : llvm::PassInfoMixin<DumpModulePass> {
  std::string Filename;
  bool IsSuffix;

  DumpModulePass(std::string Filename, bool IsSuffix)
      : Filename(std::move(Filename)), IsSuffix(IsSuffix) {}

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

} // namespace polly

namespace llvm {
class PassRegistry;
void initializeDumpModuleWrapperPassPass(llvm::PassRegistry &);
} // namespace llvm

#endif /* POLLY_SUPPORT_DUMPMODULEPASS_H */
