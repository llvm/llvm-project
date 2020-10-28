//===--------- Definition of the HWAddressSanitizer class -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Hardware AddressSanitizer class which is a port of the
// legacy HWAddressSanitizer pass to use the new PassManager infrastructure.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_HWADDRESSSANITIZERPASS_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_HWADDRESSSANITIZERPASS_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// This is a public interface to the hardware address sanitizer pass for
/// instrumenting code to check for various memory errors at runtime, similar to
/// AddressSanitizer but based on partial hardware assistance.
class HWAddressSanitizerPass : public PassInfoMixin<HWAddressSanitizerPass> {
public:
  explicit HWAddressSanitizerPass(bool CompileKernel = false,
                                  bool Recover = false);
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  static bool isRequired() { return true; }

private:
  bool CompileKernel;
  bool Recover;
};

FunctionPass *createHWAddressSanitizerLegacyPassPass(bool CompileKernel = false,
                                                     bool Recover = false);

} // namespace llvm

#endif
