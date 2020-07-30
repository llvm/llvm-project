//===- MLModelRunner.h ---- ML model runner interface -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#ifndef LLVM_ANALYSIS_MLMODELRUNNER_H
#define LLVM_ANALYSIS_MLMODELRUNNER_H

#include "llvm/Analysis/InlineModelFeatureMaps.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// MLModelRunner interface: abstraction of a mechanism for evaluating a
/// tensorflow "saved model".
class MLModelRunner {
public:
  // Disallows copy and assign.
  MLModelRunner(const MLModelRunner &) = delete;
  MLModelRunner &operator=(const MLModelRunner &) = delete;
  virtual ~MLModelRunner() = default;

  virtual bool run() = 0;
  virtual void setFeature(FeatureIndex Index, int64_t Value) = 0;
  virtual int64_t getFeature(int Index) const = 0;

protected:
  MLModelRunner(LLVMContext &Ctx) : Ctx(Ctx) {}

  LLVMContext &Ctx;
};
} // namespace llvm

#endif // LLVM_ANALYSIS_MLMODELRUNNER_H
