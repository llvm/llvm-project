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

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// MLModelRunner interface: abstraction of a mechanism for evaluating a
/// tensorflow "saved model".
/// NOTE: feature indices are expected to be consistent all accross
/// MLModelRunners (pertaining to the same model), and also Loggers (see
/// TFUtils.h)
class MLModelRunner {
public:
  // Disallows copy and assign.
  MLModelRunner(const MLModelRunner &) = delete;
  MLModelRunner &operator=(const MLModelRunner &) = delete;
  virtual ~MLModelRunner() = default;

  template <typename T> T evaluate() {
    return *reinterpret_cast<T *>(evaluateUntyped());
  }

  template <typename T, typename I> T *getTensor(I FeatureID) {
    return reinterpret_cast<T *>(
        getTensorUntyped(static_cast<size_t>(FeatureID)));
  }

  template <typename T, typename I> const T *getTensor(I FeatureID) const {
    return reinterpret_cast<const T *>(
        getTensorUntyped(static_cast<size_t>(FeatureID)));
  }

protected:
  MLModelRunner(LLVMContext &Ctx) : Ctx(Ctx) {}
  virtual void *evaluateUntyped() = 0;
  virtual void *getTensorUntyped(size_t Index) = 0;
  const void *getTensorUntyped(size_t Index) const {
    return (const_cast<MLModelRunner *>(this))->getTensorUntyped(Index);
  }

  LLVMContext &Ctx;
};
} // namespace llvm

#endif // LLVM_ANALYSIS_MLMODELRUNNER_H
