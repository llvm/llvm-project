//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_SCALAR_SCALAROPTIONS_H
#define LLVM_LIB_TRANSFORMS_SCALAR_SCALAROPTIONS_H

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include <limits>
#include <optional>

namespace llvm {
enum class CRCStrategyKind { Disable, Auto, Table, Clmul };
enum class LoopInterchangeRule {
  PerLoopCacheAnalysis,
  PerInstrOrderCost,
  ForVectorization,
  Ignore
};
enum class MatrixLayoutTy { ColumnMajor, RowMajor };
} // namespace llvm

#define OPTIONS_STRUCT_DECL
#include "ScalarOptions.inc"

namespace llvm {

class Loop;

inline const ScalarOptions &getScalarOptions(const LLVMContext &C) {
  return C.getLibraryOptions().get<ScalarOptions>();
}

inline const ScalarOptions &getScalarOptions(const Function &F) {
  return getScalarOptions(F.getContext());
}

const ScalarOptions &getScalarOptions(const Loop &L);

} // namespace llvm

#endif // LLVM_LIB_TRANSFORMS_SCALAR_SCALAROPTIONS_H
