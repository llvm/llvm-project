//===- IPO/OpenMPOpt.h - Collection of OpenMP optimizations -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_OPENMP_OPT_H
#define LLVM_TRANSFORMS_IPO_OPENMP_OPT_H

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

namespace omp {

/// Helper to remember if the module contains OpenMP (runtime calls), to be used
/// foremost with containsOpenMP.
struct OpenMPInModule {
  OpenMPInModule &operator=(bool Found) {
    if (Found)
      Value = OpenMPInModule::OpenMP::FOUND;
    else
      Value = OpenMPInModule::OpenMP::NOT_FOUND;
    return *this;
  }
  bool isKnown() { return Value != OpenMP::UNKNOWN; }
  operator bool() { return Value != OpenMP::NOT_FOUND; }

private:
  enum class OpenMP { FOUND, NOT_FOUND, UNKNOWN } Value = OpenMP::UNKNOWN;
};

/// Helper to determine if \p M contains OpenMP (runtime calls).
bool containsOpenMP(Module &M, OpenMPInModule &OMPInModule);

} // namespace omp

/// OpenMP optimizations pass.
class OpenMPOptPass : public PassInfoMixin<OpenMPOptPass> {
  /// Helper to remember if the module contains OpenMP (runtime calls).
  omp::OpenMPInModule OMPInModule;

public:
  PreservedAnalyses run(LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &UR);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_OPENMP_OPT_H
