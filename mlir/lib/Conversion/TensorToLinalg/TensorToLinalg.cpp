//===- TensorToLinalg.cpp - Tensor to Linalg Patterns ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert Tensor dialect to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TensorToLinalg/TensorToLinalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tensor-to-linalg-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateTensorToLinalgPatterns(RewritePatternSet &patterns) {
  // TODO: Add the remaining patterns, e.g. to decompose Pack/Unpack Ops.
  // Alternatively, delete this file.
  patterns.add<mlir::linalg::DecomposePadOpPattern>(patterns.getContext());
}
