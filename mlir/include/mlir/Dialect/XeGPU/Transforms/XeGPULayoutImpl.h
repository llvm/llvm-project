//===- XeGPULayoutImpl.h - Layout utility functions ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_UTILS_XeGPULayoutImpl_H_
#define MLIR_DIALECT_XEGPU_UTILS_XeGPULayoutImpl_H_

#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Dialect/XeGPU/uArch/IntelGpuXe2.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {

class VectorType;
class OpOperand;
class OpResult;
class OpBuilder;
class ValueRange;
class TypeConverter;
class OpFoldResult;

namespace xegpu {
class DistributeLayoutAttr;
class LayoutAttr;
class TensorDescType;
} // namespace xegpu

namespace xegpu {

LogicalResult propagateLayouts(OpBuilder &builder, Operation *target,
                               LayoutKind layoutKind, bool printOnly = false);

LogicalResult resolveLayoutConflicts(Operation *target);

/// [to-be-deprecated] Set the DistributeLayoutAttr for each OpOperand and
/// OpResult of of the given operation. If the operation contains regions, it is
/// also applied recursively to the contained operations operation.
/// TODO: To be replaced by recoverTemporaryLayouts()
void recoverTemporaryLayoutsDeprecated(Operation *op);

/// Attach layout attributes to all vector-type operands of operations within
/// the given operation's nested region. Reports an error if any vector operand
/// lacks a layout attribute.
bool recoverTemporaryLayouts(Operation *rootOp);

/// Removes the LayoutAttr for a given OpOperand or OpResult if it exists.
template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, OpOperand> ||
                                      std::is_same_v<T, OpResult>>>
void removeLayoutAttr(const T &operandOrResult);

/// Removes the DistributeLayoutAttr for each OpOperand and OpResult of the
/// given operation if they exist. If the operation contains regions, it is also
/// applied recursively to the contained operations
void removeLayoutAttrs(Operation *op);

/// Updates the NamedAttribute sequence by dropping sg-layout and
/// sg-data information from any DistributeLayoutAttr found.
SmallVector<NamedAttribute>
dropSgLayoutAndDataOnAttrs(ArrayRef<NamedAttribute> attrs);

/// Updates the NamedAttribute sequence by dropping inst-data information from
/// any DistributeLayoutAttr found.
SmallVector<NamedAttribute> dropInstDataOnAttrs(ArrayRef<NamedAttribute> attrs);

/// Infers the source layout attribute for a broadcast operation given the
/// result layout attribute, result shape, and source shape.
DistributeLayoutAttr inferBroadcastSourceLayout(DistributeLayoutAttr resLayout,
                                                ArrayRef<int64_t> resShape,
                                                ArrayRef<int64_t> srcShape);

/// Infers the source layout attribute for a reduction operation given the
/// result layout attribute and reduced dims.
DistributeLayoutAttr
inferMultiReductionSourceLayout(DistributeLayoutAttr resLayout,
                                SmallVector<int64_t> reduceDims);

/// Infers the source layout attribute for a bitcast operation given the
/// result layout attribute, result element type bitwidth, and source element
/// type bitwidth.
DistributeLayoutAttr inferBitCastSourceLayout(DistributeLayoutAttr resLayout,
                                              int resElemTyBitWidth,
                                              int srcElemTyBitWidth);

/// Infers the source layout attribute for a shape cast operation given the
/// result layout attribute, result shape, and source shape.
DistributeLayoutAttr inferShapeCastSourceLayout(DistributeLayoutAttr resLayout,
                                                ArrayRef<int64_t> resShape,
                                                ArrayRef<int64_t> srcShape);

/// Infers the source layout attribute for an insert strided slice operation
/// given the result layout attribute, result shape, and source shape. Removes
/// leading dimensions from the result layout to match the source shape size.
DistributeLayoutAttr
inferInsertStridedSliceSourceLayout(DistributeLayoutAttr resLayout,
                                    ArrayRef<int64_t> resShape,
                                    ArrayRef<int64_t> srcShape);

/// Sets up layout for reduction operations by creating a SliceAttr for the
/// result.
///
/// This function first attempts to construct a source layout that, when
/// sliced along reduction dimensions, produces a result layout compatible
/// with the consumer's preferred layout. This minimizes data redistribution
/// overhead. The SliceAttr for the result is then created based on the
/// derived source layout and the specified reduction dimensions.
SliceAttr setupMultiReductionResultLayout(LayoutKind layoutKind,
                                          VectorType srcVectorTy,
                                          DistributeLayoutAttr consumerLayout,
                                          SmallVector<int64_t> reductionDims,
                                          const uArch::uArch *uArch);

/// Setup the result layout attribute for a bitcast operation based on element
/// type bitwidths. This ensures the source layout can always be derived from
/// the result layout.
///
/// When casting from a narrower to a wider element type (srcElemTyBitWidth <
/// resElemTyBitWidth), the result layout's innermost dimension data sizes
/// (inst_data, lane_data) are scaled up by the bitwidth ratio. This maintains
/// the invariant that the source layout can be recovered by adjusting the
/// result layout based on bitwidth ratio of input vs output.
DistributeLayoutAttr setupBitCastResultLayout(
    LayoutKind layoutKind, VectorType srcVectorTy, VectorType resVectorTy,
    DistributeLayoutAttr consumerLayout, const uArch::uArch *uArch);

/// Sets up the result layout for an insert strided slice operation.
/// Creates a result layout based on the specified layout kind (InstData or
/// Lane).
DistributeLayoutAttr setupInsertStridedSliceResultLayout(
    LayoutKind layoutKind, VectorType srcVectorTy, VectorType resVectorTy,
    DistributeLayoutAttr consumerLayout, const uArch::uArch *uArch);

/// Sets up the anchor layout for a load gather operation.
DistributeLayoutAttr
setupLoadGatherAnchorLayout(LayoutKind layoutKind, VectorType vectorTy,
                            int chunkSize, DistributeLayoutAttr consumerLayout,
                            const uArch::uArch *uArch);

/// Sets up the anchor layout for load matrix operation.
DistributeLayoutAttr
setupLoadMatrixAnchorLayout(LayoutKind layoutKind, VectorType vectorTy,
                            DistributeLayoutAttr consumerLayout,
                            const uArch::uArch *uArch);

/// Sets up the anchor layout for a store scatter operation.
DistributeLayoutAttr setupStoreScatterAnchorLayout(LayoutKind layoutKind,
                                                   VectorType vectorTy,
                                                   int chunkSize,
                                                   const uArch::uArch *uArch);

/// Sets up the anchor layout for a store matrix operation.
DistributeLayoutAttr setupStoreMatrixAnchorLayout(LayoutKind layoutKind,
                                                  VectorType vectorTy,
                                                  const uArch::uArch *uArch);

/// Sets up the anchor layouts for a dpas operands (A, B, and C/D).
/// The numSg and consumerLayout (optional) are only used by sg layout creation.
std::optional<std::tuple<DistributeLayoutAttr, DistributeLayoutAttr,
                         DistributeLayoutAttr>>
setupDpasLayout(LayoutKind layoutKind, VectorType aTy, VectorType bTy,
                VectorType cdTy, DistributeLayoutAttr consumerLayout,
                const uArch::uArch *uArch, int numSg);

} // namespace xegpu

} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_UTILS_XEGPUUTILS_H_
