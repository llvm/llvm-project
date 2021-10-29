//===- HoistPadding.h - Hoisting transformation for PadTensorOp -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_HOIST_PADDING_H_
#define MLIR_DIALECT_LINALG_TRANSFORMS_HOIST_PADDING_H_

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Value;

namespace linalg {
class PadTensorOp;

/// Mechanically hoist padding operations on tensors by `nLoops` into a new,
/// generally larger tensor. This achieves packing of multiple padding ops into
/// a larger tensor. On success, `padTensorOp` is replaced by the cloned version
/// in the packing loop so the caller can continue reasoning about the padding
/// operation.
///
/// Example in pseudo-mlir:
/// =======================
///
/// If hoistPaddingOnTensors is called with `nLoops` = 2 on the following IR.
/// ```
///    scf.for (%i, %j, %k)
///      %st0 = tensor.extract_slice f(%i, %k) : ... to tensor<?x?xf32>
///      %0 = linalg.pad_tensor %st0 low[0, 0] high[...] {
///      ^bb0( ... ):
///        linalg.yield %pad
///      } : tensor<?x?xf32> to tensor<4x8xf32>
///      compute(%0)
/// ```
///
/// IR resembling the following is produced:
///
/// ```
///    scf.for (%i) {
///      %packed_init = linalg.init_tensor range(%j) : tensor<?x4x8xf32>
///      %packed = scf.for (%k) iter_args(%p : %packed_init) {
///        %st0 = tensor.extract_slice f(%i, %k) : ... to tensor<?x?xf32>
///        %0 = linalg.pad_tensor %st0 low[0, 0] high[...] {
///        ^bb0( ... ):
///          linalg.yield %pad
///        } : tensor<?x?xf32> to tensor<4x8xf32>
///        %1 = tensor.insert_slice %0 ...
///            : tensor<4x8xf32> to tensor<?x4x8xf32>
///        scf.yield %1: tensor<?x4x8xf32>
///      } -> tensor<?x4x8xf32>
///      scf.for (%j, %k) {
///        %st0 = tensor.extract_slice %packed [%k, 0, 0][1, 4, 8][1, 1, 1] :
///                 tensor<?x4x8xf32> to tensor<4x8xf32>
///        compute(%st0)
///      }
///    }
/// ```
FailureOr<Value> hoistPaddingOnTensors(PadTensorOp opToHoist, int numLoops,
                                       PadTensorOp &hoistedOp);

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_HOIST_PADDING_H_
