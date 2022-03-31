//===- HoistPadding.h - Hoisting for tensor::PadOp -*- C++ --------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_HOISTPADDING_H
#define MLIR_DIALECT_LINALG_TRANSFORMS_HOISTPADDING_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Value;

namespace tensor {
class PadOp;
} // namespace tensor

namespace linalg {
class GenericOp;

/// Mechanically hoist padding operations on tensors by `numLoops` into a new,
/// generally larger tensor. This achieves packing of multiple padding ops into
/// a larger tensor. On success, `opToHoist` is replaced by the cloned version
/// in the packing loop so the caller can continue reasoning about the padding
/// operation. If `transposeVector` is non-empty, hoist padding introduces a
/// GenericOp to transpose the padded tensor before inserting it into the packed
/// tensor. A `transposeVector` can change the storage order of the padded
/// tensor but does not change the order of the pack or compute loops.
///
///
/// Example in pseudo-mlir:
/// =======================
///
/// If hoistPaddingOnTensors is called with `nLoops` = 2 on the following IR.
/// ```
///    scf.for (%i, %j, %k)
///      %st0 = tensor.extract_slice f(%i, %k) : ... to tensor<?x?xf32>
///      %0 = tensor.pad %st0 low[0, 0] high[...] {
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
///        %0 = tensor.pad %st0 low[0, 0] high[...] {
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
FailureOr<Value> hoistPaddingOnTensors(
    tensor::PadOp opToHoist, int numLoops, ArrayRef<int64_t> transposeVector,
    tensor::PadOp &hoistedOp, SmallVectorImpl<GenericOp> &transposeOps);

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_HOISTPADDING_H
