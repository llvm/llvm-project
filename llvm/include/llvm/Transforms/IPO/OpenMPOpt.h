//===- IPO/OpenMPOpt.h - Collection of OpenMP optimizations -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_OPENMPOPT_H
#define LLVM_TRANSFORMS_IPO_OPENMPOPT_H

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

namespace omp {

/// Summary of a kernel (=entry point for target offloading).
using Kernel = Function *;

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

  /// Does this function \p F contain any OpenMP runtime calls?
  bool containsOMPRuntimeCalls(Function *F) const {
    return FuncsWithOMPRuntimeCalls.contains(F);
  }

  /// Return the known kernels (=GPU entry points) in the module.
  SmallPtrSetImpl<Kernel> &getKernels() { return Kernels; }

  /// Identify kernels in the module and populate the Kernels set.
  void identifyKernels(Module &M);

private:
  enum class OpenMP { FOUND, NOT_FOUND, UNKNOWN } Value = OpenMP::UNKNOWN;

  friend bool containsOpenMP(Module &M, OpenMPInModule &OMPInModule);

  /// In which functions are OpenMP runtime calls present?
  SmallPtrSet<Function *, 32> FuncsWithOMPRuntimeCalls;

  /// Collection of known kernels (=GPU entry points) in the module.
  SmallPtrSet<Kernel, 8> Kernels;
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

#endif // LLVM_TRANSFORMS_IPO_OPENMPOPT_H
