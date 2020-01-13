//===- CilkSanitizer.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
