//===- Builders.cpp - MLIR Declarative Linalg Builders --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/VectorOps/EDSC/Builders.h"
#include "mlir/Dialect/VectorOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/VectorOps/VectorOps.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/Functional.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::edsc::ops;

Value mlir::edsc::ops::vector_contraction(
    StructuredIndexed A, StructuredIndexed B, StructuredIndexed C,
    ArrayRef<IteratorType> iteratorTypes) {
  using IndexingExprs = ArrayRef<ArrayRef<AffineExpr>>;
  return vector_contract(
      A.getValue(), B.getValue(), C.getValue(),
      IndexingExprs{A.getExprs(), B.getExprs(), C.getExprs()},
      ArrayRef<StringRef>{functional::map(toString, iteratorTypes)});
}

Value mlir::edsc::ops::vector_matmul(Value A, Value B, Value C) {
  AffineExpr m, n, k;
  bindDims(ScopedContext::getContext(), m, n, k);
  return vector_contraction(StructuredIndexed(A, {m, k}),
                            StructuredIndexed(B, {k, n}),
                            StructuredIndexed(C, {m, n}),
                            {IteratorType::Parallel, IteratorType::Parallel,
                             IteratorType::Reduction});
}
