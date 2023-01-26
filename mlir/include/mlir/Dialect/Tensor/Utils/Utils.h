//===- Utils.h -  Utilities to support the Tensor dialect -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TENSOR_UTILS_UTILS_H_
#define MLIR_DIALECT_TENSOR_UTILS_UTILS_H_

#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace tensor {

// Return a PadOp that pads `source` to `type` size where the static
// sizes are assumed to be greater than the dynamic sizes. If `type` has dynamic
// dimensions the padding width is set to zero. The op performs "high" padding
// (i.e. it adds trailing padding values until the desired size is met).
PadOp createPadHighOp(RankedTensorType type, Value source, Value pad,
                      bool nofold, Location loc, OpBuilder &builder);

// Creates dim ops for each dynamic dimension of the ranked tensor argument and
// returns these as values.
SmallVector<Value> createDynamicDimValues(OpBuilder &b, Location loc,
                                          Value rankedTensor);

// Creates dim ops or constant ops for each dimension of the ranked tensor
// argument and returns these as values.
SmallVector<OpFoldResult> createDimValues(OpBuilder &b, Location loc,
                                          Value rankedTensor);

} // namespace tensor
} // namespace mlir

#endif // MLIR_DIALECT_TENSOR_UTILS_UTILS_H_
