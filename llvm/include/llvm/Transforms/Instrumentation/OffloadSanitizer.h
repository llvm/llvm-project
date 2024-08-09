//===- Transforms/Instrumentation/OffloadSanitizer.h ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass to instrument offload code in order to detect errors and communicate
// them to the LLVM/Offload runtimes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_OFFLOADSAN_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_OFFLOADSAN_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class OffloadSanitizerPass : public PassInfoMixin<OffloadSanitizerPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_OFFLOADSAN_H
