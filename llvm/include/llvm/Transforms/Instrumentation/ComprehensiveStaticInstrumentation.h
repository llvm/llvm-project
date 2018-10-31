//===- ComprehensiveStaticInstrumentation.h ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file is part of CSI, a framework that provides comprehensive static
/// instrumentation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_COMPREHENSIVESTATICINSTRUMENTATION_H
#define LLVM_TRANSFORMS_COMPREHENSIVESTATICINSTRUMENTATION_H

#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Instrumentation.h"

namespace llvm {

/// ComprehensiveStaticInstrumentation pass for new pass manager.
class ComprehensiveStaticInstrumentationPass :
    public PassInfoMixin<ComprehensiveStaticInstrumentationPass> {
public:
  ComprehensiveStaticInstrumentationPass(const CSIOptions &Options = CSIOptions());
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  CSIOptions Options;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_COMPREHENSIVESTATICINSTRUMENTATION_H
