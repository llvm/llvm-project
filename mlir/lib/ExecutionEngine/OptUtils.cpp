//===- OptUtils.cpp - MLIR Execution Engine optimization pass utilities ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the utility functions to trigger LLVM optimizations from
// MLIR Execution Engine.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/OptUtils.h"

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

static Optional<OptimizationLevel> mapToLevel(unsigned optLevel,
                                              unsigned sizeLevel) {
  switch (optLevel) {
  case 0:
    return OptimizationLevel::O0;

  case 1:
    return OptimizationLevel::O1;

  case 2:
    switch (sizeLevel) {
    case 0:
      return OptimizationLevel::O2;

    case 1:
      return OptimizationLevel::Os;

    case 2:
      return OptimizationLevel::Oz;
    }

  case 3:
    return OptimizationLevel::O3;
  }
  return None;
}
// Create and return a lambda that uses LLVM pass manager builder to set up
// optimizations based on the given level.
std::function<Error(Module *)>
mlir::makeOptimizingTransformer(unsigned optLevel, unsigned sizeLevel,
                                TargetMachine *targetMachine) {
  return [optLevel, sizeLevel, targetMachine](Module *m) -> Error {
    Optional<OptimizationLevel> ol = mapToLevel(optLevel, sizeLevel);
    if (!ol) {
      return make_error<StringError>(
          formatv("invalid optimization/size level {0}/{1}", optLevel,
                  sizeLevel)
              .str(),
          inconvertibleErrorCode());
    }
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;

    PassBuilder pb(targetMachine);

    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm;
    if (*ol == OptimizationLevel::O0)
      mpm.addPass(pb.buildO0DefaultPipeline(*ol));
    else
      mpm.addPass(pb.buildPerModuleDefaultPipeline(*ol));

    mpm.run(*m, mam);
    return Error::success();
  };
}
