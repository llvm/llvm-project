//===- XeGPULayoutUtils.h - Layout Utilities --------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_UTILS_XEGPULAYOUTUTILS_H_
#define MLIR_DIALECT_XEGPU_UTILS_XEGPULAYOUTUTILS_H_

#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
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

/// Return the attribute name for the OpOperand to attach DistributeLayoutAttr
std::string getTemporaryLayoutName(const OpOperand &operand);

/// Return the attribute name for the OpResult to attach DistributeLayoutAttr
std::string getTemporaryLayoutName(const OpResult result);

/// Retrieves the DistributeLayoutAttr associated with a given Value. For
/// TensorDescType values, the DistributeLayoutAttr is extracted from the
/// TensorDescType itself. For other values, it is obtained from the attributes
/// of the defining operation. Returns nullptr if no DistributeLayoutAttr is
/// found.
DistributeLayoutAttr getDistributeLayoutAttr(const Value value);

/// Retrieves the DistributeLayoutAttr associated with a given OpOperand. It
/// will first check the operand_layout_{id} of the owner operation. If not
/// found, it will check the operand itself and its defining op.
DistributeLayoutAttr getDistributeLayoutAttr(const OpOperand &opr);

/// [to-be-deprecated] Sets the DistributeLayoutAttr for a given OpResult
/// user should use setAnchorLayout instead
void setDistributeLayoutAttr(const OpResult &Result,
                             const DistributeLayoutAttr layout);

/// [to-be-deprecated] Sets the DistributeLayoutAttr for a given OpOperand
/// user should use setAnchorLayout instead
void setDistributeLayoutAttr(const OpOperand &opr,
                             const DistributeLayoutAttr layout);

/// get and set distribute layout attribute for non-anchor operations
/// (and offsets/masks of load/store ops before we get rid of their temp attrs)
template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, OpOperand> ||
                                      std::is_same_v<T, OpResult>>>
DistributeLayoutAttr getTemporaryLayout(const T &operandOrResult);

template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, OpOperand> ||
                                      std::is_same_v<T, OpResult>>>
void setTemporaryLayout(const T &operandOrResult,
                        const DistributeLayoutAttr layout);

/// [to-be-deprecated] Set the DistributeLayoutAttr for each OpOperand and
/// OpResult of of the given operation. If the operation contains regions, it is
/// also applied recursively to the contained operations operation.
/// TODO: To be replaced by recoverTemporaryLayouts()
void recoverTemporaryLayoutsDeprecated(Operation *op);

/// Attach layout attributes to all vector-type operands of operations within
/// the given operation's region. Reports an error if any vector operand lacks
/// a layout attribute.
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

/// Infers the source layout attribute for a broadcast operation given the
/// result layout attribute, result shape, source shape, and broadcasted dims.
DistributeLayoutAttr inferBroadCastSourceLayout(MLIRContext *context,
                                                DistributeLayoutAttr resLayout,
                                                ArrayRef<int64_t> resShape,
                                                ArrayRef<int64_t> srcShape);

/// Infers the source layout attribute for a reduction operation given the
/// result layout attribute and reduced dims.
DistributeLayoutAttr
inferReductionSourceLayout(DistributeLayoutAttr resLayout,
                           SmallVector<int64_t> reduceDims);

/// Infers the source layout attribute for a bitcast operation given the
/// result layout attribute, result element type bitwidth, and source element
/// type bitwidth.
DistributeLayoutAttr inferBitCastSourceLayout(MLIRContext *context,
                                              DistributeLayoutAttr resLayout,
                                              int resElemTyBitWidth,
                                              int srcElemTyBitWidth);

/// Infers the source layout attribute for a shape cast operation given the
/// result layout attribute, result shape, and source shape.
DistributeLayoutAttr inferShapeCastSourceLayout(MLIRContext *context,
                                                DistributeLayoutAttr resLayout,
                                                ArrayRef<int64_t> resShape,
                                                ArrayRef<int64_t> srcShape);

/// Sets the the layout attribute for result based on a preferred Layout
/// propagated from consumer
/// the ouput must be a slice attribute
SliceAttr
reductionLayoutSetupRule(ArrayRef<int64_t> srcShape,
                         SmallVector<int64_t> reductionDims,
                         DistributeLayoutAttr consumerPreferredLayout);

} // namespace xegpu

} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_UTILS_XEGPUUTILS_H_
