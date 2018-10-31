//===- Instrumentation/CilkSanitizer.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file is part of CilkSan, a determinacy-race detector for Cilk and Tapir
/// programs.
///
/// This instrumentation pass inserts calls to the CilkSan runtime library
/// before appropriate memory accesses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_CILKSANITIZER_H
#define LLVM_TRANSFORMS_CILKSANITIZER_H

#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Instrumentation.h"

namespace llvm {

/// CilkSanitizer pass for new pass manager.
class CilkSanitizerPass : public PassInfoMixin<CilkSanitizerPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_CILKSANITIZER_H
