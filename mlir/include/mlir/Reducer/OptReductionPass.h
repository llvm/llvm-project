//===- OptReductionPass.h - Optimization Reduction Pass Wrapper -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Opt Reduction Pass Wrapper. It creates a MLIR pass to
// run any optimization pass within it and only replaces the output module with
// the transformed version if it is smaller and interesting.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REDUCER_OPTREDUCTIONPASS_H
#define MLIR_REDUCER_OPTREDUCTIONPASS_H

#include "PassDetail.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Reducer/ReductionNode.h"
#include "mlir/Reducer/ReductionTreePass.h"
#include "mlir/Reducer/Tester.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir {

class OptReductionPass : public OptReductionBase<OptReductionPass> {
public:
  OptReductionPass(const Tester &test, MLIRContext *context,
                   std::unique_ptr<Pass> optPass);

  OptReductionPass(const OptReductionPass &srcPass);

  /// Runs the pass instance in the pass pipeline.
  void runOnOperation() override;

private:
  // Points to the context to be used in the pass manager.
  MLIRContext *context;

  // This is used to test the interesting behavior of the transformed module.
  const Tester &test;

  // Points to the mlir-opt pass to be called.
  std::unique_ptr<Pass> optPass;
};

} // end namespace mlir

#endif
