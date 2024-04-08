//===- VectorTypes.cpp - MLIR Vector Types --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/VectorTypes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::vector;

bool ScalableVectorType::classof(Type type) {
  auto vecTy = dyn_cast<VectorType>(type);
  if (!vecTy)
    return false;
  return vecTy.isScalable();
}

bool FixedWidthVectorType::classof(Type type) {
  auto vecTy = llvm::dyn_cast<VectorType>(type);
  if (!vecTy)
    return false;
  return !vecTy.isScalable();
}
