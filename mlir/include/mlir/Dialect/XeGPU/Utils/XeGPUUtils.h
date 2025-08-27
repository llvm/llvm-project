//===- XeGPUUtils.h - Vector Utilities --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_UTILS_XEGPUUTILS_H_
#define MLIR_DIALECT_XEGPU_UTILS_XEGPUUTILS_H_

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

namespace xegpu {
class DistributeLayoutAttr;
class LayoutAttr;
class TensorDescType;
} // namespace xegpu

namespace xegpu {

/// Flatten a set of ValueRange into a single SmallVector<Value>
SmallVector<Value> flattenValues(ArrayRef<ValueRange> values);

/// If tensor descriptor has a layout attribute it is used in SIMT mode.
/// In this mode, the distributed vector shape is determined as follows:
/// Definitions:
///        lane_data_size = lane_data[0] × lane_data[1]
///        subgroup_size = lane_layout[0] × lane_layout[1]
///        distribution_unit_size = subgroup_size × lane_data_size
///
/// Case 1: Regular loads/stores.
/// The following conditions must be met:
///        * tensor_desc[0] == lane_layout[0]
/// Distributed vector is a 1D vector with shape:
///        [chunk_size]
///
/// Case 2: Block loads/stores
/// Additional definitions:
///        tensor_size = tensor_desc[0] * .. * tensor_desc[r-1] * array_length
///        n_distribution_units = tensor_size / distribution_unit_size
///        fragment_size = n_distribution_units * lane_data_size
/// Given above definitions, the following conditions must be met:
///        * tensor_desc[0] % (lane_layout[0] × lane_data[0]) == 0
///        * tensor_desc[1] % (lane_layout[1] × lane_data[1]) == 0
/// Distributed vector is a 1D vector with shape:
///        [fragment_size]
FailureOr<VectorType> getDistributedVectorType(xegpu::TensorDescType tdescTy);

/// Helper to get the distributed vector type for a given vector type according
/// to a given LayoutAttr.
FailureOr<VectorType> getDistributedVectorType(VectorType originalType,
                                               LayoutAttr layout);

/// Return the attribute name for the OpOperand to attach DistributeLayoutAttr
std::string getLayoutName(const OpOperand &operand);

/// Return the attribute name for the OpResult to attach DistributeLayoutAttr
std::string getLayoutName(const OpResult result);

/// Retrieves the DistributeLayoutAttr associated with a given Value. For
/// TensorDescType values, the DistributeLayoutAttr is extracted from the
/// TensorDescType itself. For other values, it is obtained from the attributes
/// of the defining operation. Returns nullptr if no DistributeLayoutAttr is
/// found.
DistributeLayoutAttr getDistributeLayoutAttr(const Value value);

template <typename AttrTy>
AttrTy getDistributeLayoutAttrOfType(const Value value) {
  return dyn_cast_if_present<AttrTy>(getDistributeLayoutAttr(value));
}

/// Retrieves the DistributeLayoutAttr associated with a given OpOperand. It
/// will first check the operand_layout_{id} of the owner operation. If not
/// found, it will check the operand itself and its defining op.
DistributeLayoutAttr getDistributeLayoutAttr(const OpOperand &opr);

template <typename AttrTy>
AttrTy getDistributeLayoutAttrOfType(const OpOperand &opr) {
  return dyn_cast_if_present<AttrTy>(getDistributeLayoutAttr(opr));
}

/// Removes the LayoutAttr for a given OpOperand or OpResult if it exists.
template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, OpOperand> ||
                                      std::is_same_v<T, OpResult>>>
void removeLayoutAttr(const T &operandOrResult);

/// Removes the DistributeLayoutAttr for each OpOperand and OpResult of the
/// given operation if they exist. If the operation contains regions, it is also
/// applied recursively to the contained operations
void removeLayoutAttrs(Operation *op);

/// Sets the DistributeLayoutAttr for a given OpOperand or OpResult by attaching
/// it to the owner's dictionary attributes
template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, OpOperand> ||
                                      std::is_same_v<T, OpResult>>>
void setDistributeLayoutAttr(const T &operandOrResult,
                             const DistributeLayoutAttr layout);

/// Set the DistributeLayoutAttr for each OpOperand and OpResult of the given
/// operation. If the operation contains regions, it is also applied recursively
/// to the contained operations
void setDistributeLayoutAttrs(
    Operation *op, function_ref<DistributeLayoutAttr(Value)> getLayoutImpl);

/// Extract a set of small vectors from a value with a given shape using
/// vector.extract_stride_slice
SmallVector<Value> extractVectorsWithShapeFromValue(OpBuilder &builder,
                                                    Location loc, Value value,
                                                    ArrayRef<int64_t> shape);

/// Create a vector of shape from a set of values using
/// vector.insert_stride_slice.
Value createVectorWithShapeFromValues(OpBuilder &builder, Location loc,
                                      ValueRange values,
                                      ArrayRef<int64_t> shape);

/// Do type conversion for SCF structural ops, e.g., scf.for using SCF structure
/// type convertion patterns. Since VectorType cannot carry the layout
/// attribute, which is needed to guide the type conversion for XeGPU, they are
/// first converted into RankedTensorType, where the layout attribute can be
/// attached. And then upstream SCF structural type conversion patterns are
/// applied with the provided converter.
/// TODO: This is a temporary solution. We should refactor it when context-aware
/// type conversion is available.
void doSCFStructuralTypeConversionWithTensorType(Operation *op,
                                                 TypeConverter converter);

/// Retrieves the chip string from the XeVM target attribute of the parent
/// GPU module operation. Returns the chip identifier if found, or nullopt
/// if no GPU module parent or XeVM target attribute exists.
std::optional<std::string> getChipStr(Operation *op);

/// Generates element-wise addition ops of two arrays with automatic alignment.
/// When the input arrays have different sizes, the shorter array is
/// right-aligned with the longer array, and the unmatched leading elements from
/// the longer array are preserved unchanged. This is commonly used for offset
/// computation where higher-dimensional offsets need to be added to
/// lower-dimensional adjustments.
///
/// Example:
///   lhs = [l1, l2, l3], rhs = [r1, r2]
///   Result: [11, l2+r1, l3+r2]
SmallVector<OpFoldResult> addWithRightAligned(OpBuilder &builder, Location loc,
                                              ArrayRef<OpFoldResult> lhs,
                                              ArrayRef<OpFoldResult> rhs);

} // namespace xegpu

} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_UTILS_XEGPUUTILS_H_
