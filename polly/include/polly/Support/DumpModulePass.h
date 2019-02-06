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

namespace llvm {
class StringRef;
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
llvm::ModulePass *createDumpModulePass(llvm::StringRef Filename, bool IsSuffix);
} // namespace polly

namespace llvm {
class PassRegistry;
void initializeDumpModulePass(llvm::PassRegistry &);
} // namespace llvm

#endif /* POLLY_SUPPORT_DUMPMODULEPASS_H */
