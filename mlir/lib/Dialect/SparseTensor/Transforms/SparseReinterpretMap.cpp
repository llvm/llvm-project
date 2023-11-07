//===- SparseReinterpretMap.cpp - reinterpret sparse tensor maps ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"

namespace {

// TODO:
//   (1) insert the zero-cost sparse_tensor.reinterpret_map ops
//   (2) rewrite linalg.generic ops traits on level crds
//   (3) compute topsort, and resolve cyles with sparse_tensor.convert ops

} // namespace

void mlir::populateSparseReinterpretMap(RewritePatternSet &patterns,
                                        ReinterpretMapScope scope) {}
