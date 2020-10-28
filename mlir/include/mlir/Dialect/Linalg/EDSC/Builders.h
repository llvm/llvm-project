//===- Builders.h - MLIR Declarative Linalg Builders ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides intuitive composable interfaces for building structured MLIR
// snippets in a declarative fashion.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_LINALG_EDSC_BUILDERS_H_
#define MLIR_DIALECT_LINALG_EDSC_BUILDERS_H_

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"

namespace mlir {
class AffineForOp;
class BlockArgument;

namespace scf {
class ParallelOp;
} // namespace scf

namespace edsc {
inline void defaultRegionBuilder(ValueRange args) {}

/// Build a `linalg.generic` op with the specified `inputs`, `outputBuffers`,
/// `initTensors`, `resultTensorsTypes` and `region`.
///
/// `otherValues` and `otherAttributes` may be passed and will be appended as
/// operands and attributes respectively.
///
/// Prerequisites:
/// =============
///
/// 1. `inputs` may contain StructuredIndexed that capture either buffer or
/// tensor values.
/// 2. `outputsBuffers` may contain StructuredIndexed that capture buffer
/// values.
/// 3. `initTensors` contain tensor values, without indexing maps.
/// 4. `resultTensorTypes` may contain StructuredIndexed that capture return
/// tensor types.
Operation *makeGenericLinalgOp(
    ArrayRef<IteratorType> iteratorTypes, ArrayRef<StructuredIndexed> inputs,
    ArrayRef<StructuredIndexed> outputBuffers, ArrayRef<Value> initTensors,
    ArrayRef<StructuredIndexed> resultTensorTypes,
    function_ref<void(ValueRange)> regionBuilder = defaultRegionBuilder,
    ArrayRef<Value> otherValues = {}, ArrayRef<Attribute> otherAttributes = {});

namespace ops {
using edsc::StructuredIndexed;

//===----------------------------------------------------------------------===//
// EDSC builders for linalg generic operations.
//===----------------------------------------------------------------------===//

/// Build the body of a region to compute a scalar multiply, under the current
/// ScopedContext, at the current insert point.
void mulRegionBuilder(ValueRange args);

/// Build the body of a region to compute a scalar multiply-accumulate, under
/// the current ScopedContext, at the current insert point.
void macRegionBuilder(ValueRange args);

/// TODO: In the future we should tie these implementations to something in
/// Tablegen that generates the proper interfaces and the proper sugared named
/// ops.

/// Build a linalg.pointwise, under the current ScopedContext, at the current
/// insert point, that computes:
/// ```
///    (i0, ..., in) = (par, ..., par)
///    |
///    |  O...(some_subset...(i0, ..., in)) =
///    |    some_pointwise_func...(I...(some_other_subset...(i0, ..., in)))
/// ```
///
/// This is a very generic entry point that can be configured in many ways to
/// build a perfect loop nest of parallel loops with arbitrarily complex
/// innermost loop code and whatever (explicit) broadcast semantics.
///
/// This can be used with both out-of-place and in-place semantics.
/// The client is responsible for ensuring the region operations are compatible
/// with in-place semantics and parallelism.

/// Unary pointwise operation (with broadcast) entry point.
using UnaryPointwiseOpBuilder = function_ref<Value(Value)>;
Operation *linalg_generic_pointwise(UnaryPointwiseOpBuilder unaryOp,
                                    StructuredIndexed I, StructuredIndexed O);

/// Build a linalg.pointwise with all `parallel` iterators and a region that
/// computes `O = tanh(I)`. The client is responsible for specifying the proper
/// indexings when creating the StructuredIndexed.
Operation *linalg_generic_pointwise_tanh(StructuredIndexed I,
                                         StructuredIndexed O);

/// Binary pointwise operation (with broadcast) entry point.
using BinaryPointwiseOpBuilder = function_ref<Value(Value, Value)>;
Operation *linalg_generic_pointwise(BinaryPointwiseOpBuilder binaryOp,
                                    StructuredIndexed I1, StructuredIndexed I2,
                                    StructuredIndexed O);

/// Build a linalg.pointwise with all `parallel` iterators and a region that
/// computes `O = I1 + I2`. The client is responsible for specifying the proper
/// indexings when creating the StructuredIndexed.
Operation *linalg_generic_pointwise_add(StructuredIndexed I1,
                                        StructuredIndexed I2,
                                        StructuredIndexed O);

/// Build a linalg.pointwise with all `parallel` iterators and a region that
/// computes `O = max(I1, I2)`. The client is responsible for specifying the
/// proper indexings when creating the StructuredIndexed.
Operation *linalg_generic_pointwise_max(StructuredIndexed I1,
                                        StructuredIndexed I2,
                                        StructuredIndexed O);

// TODO: Implement more useful pointwise operations on a per-need basis.

using MatmulRegionBuilder = function_ref<void(ValueRange args)>;

/// Build a linalg.generic, under the current ScopedContext, at the current
/// insert point, that computes:
/// ```
///    (m, n, k) = (par, par, seq)
///    |
///    |  C(m, n) += A(m, k) * B(k, n)
/// ```
Operation *
linalg_generic_matmul(Value vA, Value vB, Value vC,
                      MatmulRegionBuilder regionBuilder = macRegionBuilder);

/// Build a linalg.generic, under the current ScopedContext, at the current
/// insert point, that computes:
/// ```
///    (m, n, k) = (par, par, seq)
///    |
///    |  D(m, n) = C(m, n) + sum_k(A(m, k) * B(k, n))
/// ```
/// and returns the tensor `D`.
Operation *
linalg_generic_matmul(Value vA, Value vB, Value vC, RankedTensorType tD,
                      MatmulRegionBuilder regionBuilder = macRegionBuilder);

template <typename Container>
Operation *
linalg_generic_matmul(Container values,
                      MatmulRegionBuilder regionBuilder = macRegionBuilder) {
  assert(values.size() == 3 && "Expected exactly 3 values");
  return linalg_generic_matmul(values[0], values[1], values[2], regionBuilder);
}

/// Build a linalg.generic, under the current ScopedContext, at the current
/// insert point, that computes:
/// ```
///    (batch, f, [h, w, ...], [kh, kw, ...], c) =
///    |  (par, par, [par, par, ...], [red, red, ...], red)
///    |
///    | O(batch, [h, w, ...], f) +=
///    |   I(batch,
///    |     [
///    |       stride[0] * h + dilations[0] * kh,
///    |       stride[1] * w + dilations[1] * kw, ...
///          ],
///    |     c)
///    |   *
///    |   W([kh, kw, ...], c, f)
/// ```
/// If `dilations` or `strides` are left empty, the default value of `1` is used
/// along each relevant dimension.
///
/// For now `...` must be empty (i.e. only 2-D convolutions are supported).
///
// TODO: Extend convolution rank with some template magic.
Operation *linalg_generic_conv_nhwc(Value vI, Value vW, Value vO,
                                    ArrayRef<int> strides = {},
                                    ArrayRef<int> dilations = {});

template <typename Container>
Operation *linalg_generic_conv_nhwc(Container values,
                                    ArrayRef<int> strides = {},
                                    ArrayRef<int> dilations = {}) {
  assert(values.size() == 3 && "Expected exactly 3 values");
  return linalg_generic_conv_nhwc(values[0], values[1], values[2], strides,
                                  dilations);
}

/// Build a linalg.generic, under the current ScopedContext, at the current
/// insert point, that computes:
/// ```
///    (batch, dm, c, [h, w, ...], [kh, kw, ...]) =
///    |  (par, par, par, [par, par, ...], [red, red, ...])
///    |
///    | O(batch, [h, w, ...], c * depth_multiplier) +=
///    |   I(batch,
///    |     [
///    |       stride[0] * h + dilations[0] * kh,
///    |       stride[1] * w + dilations[1] * kw, ...
///          ],
///    |     c)
///    |   *
///    |   W([kh, kw, ...], c, depth_multiplier)
/// ```
/// If `dilations` or `strides` are left empty, the default value of `1` is used
/// along each relevant dimension.
///
/// For now `...` must be empty (i.e. only 2-D convolutions are supported).
///
// TODO: Extend convolution rank with some template magic.
Operation *linalg_generic_dilated_conv_nhwc(Value vI, Value vW, Value vO,
                                            int depth_multiplier = 1,
                                            ArrayRef<int> strides = {},
                                            ArrayRef<int> dilations = {});

template <typename Container>
Operation *linalg_generic_dilated_conv_nhwc(Container values,
                                            int depth_multiplier,
                                            ArrayRef<int> strides = {},
                                            ArrayRef<int> dilations = {}) {
  assert(values.size() == 3 && "Expected exactly 3 values");
  return linalg_generic_dilated_conv_nhwc(values[0], values[1], values[2],
                                          depth_multiplier, strides, dilations);
}

} // namespace ops
} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_EDSC_BUILDERS_H_
