//===--------- Definition of the AddressSanitizer class ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the AddressSanitizer class which is a port of the legacy
// AddressSanitizer pass to use the new PassManager infrastructure.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_ADDRESSSANITIZERPASS_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_ADDRESSSANITIZERPASS_H

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// Public interface to the address sanitizer pass for instrumenting code to
/// check for various memory bugs.
class AddressSanitizerPass : public PassInfoMixin<AddressSanitizerPass> {
public:
  explicit AddressSanitizerPass(bool CompileKernel = false,
                                bool Recover = false,
                                bool UseAfterScope = false);
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  bool CompileKernel;
  bool Recover;
  bool UseAfterScope;
};

} // namespace llvm

#endif
