//===------ DumpModulePass.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
