//===- SCCP.h - Sparse Conditional Constant Propagation ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements  interprocedural sparse conditional constant
// propagation and merging.
//
// Specifically, this:
//   * Assumes values are constant unless proven otherwise
//   * Assumes BasicBlocks are dead unless proven otherwise
//   * Proves values to be constant, and replaces them with constants
//   * Proves conditional branches to be unconditional
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_SCCP_H
#define LLVM_TRANSFORMS_IPO_SCCP_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;

/// Pass to perform interprocedural constant propagation.
class IPSCCPPass : public PassInfoMixin<IPSCCPPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_SCCP_H
