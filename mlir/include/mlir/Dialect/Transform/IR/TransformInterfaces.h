//===- TransformInterfaces.h - Transform Dialect Interfaces -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_TRANSFORMINTERFACES_H
#define MLIR_DIALECT_TRANSFORM_IR_TRANSFORMINTERFACES_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace transform {

class TransformOpInterface;

/// The state maintained across applications of various ops implementing the
/// TransformOpInterface. The operations implementing this interface and the
/// surrounding structure are referred to as transform IR. The operations to
/// which transformations apply are referred to as payload IR. The state thus
/// contains the mapping between values defined in the transform IR ops and
/// payload IR ops. It assumes that each value in the transform IR can be used
/// at most once (since transformations are likely to change the payload IR ops
/// the value corresponds to). Checks that transform IR values correspond to
/// disjoint sets of payload IR ops throughout the transformation.
///
/// A reference to this class is passed as an argument to "apply" methods of the
/// transform op interface. Thus the "apply" method can call
/// `state.getPayloadOps( getSomeOperand() )` to obtain the list of operations
/// associated with its operand and subject to transformation. The method is
/// expected to populate the `TransformResults` class instance in order to
/// update the mapping. The `applyTransform` method takes care of propagating
/// the state of `TransformResults` into the instance of this class.
class TransformState {
  /// Mapping between a Value in the transform IR and the corresponding set of
  /// operations in the payload IR.
  using TransformOpMapping = DenseMap<Value, SmallVector<Operation *>>;

  /// Mapping between a payload IR operation and the transform IR value it is
  /// currently associated with.
  using TransformOpReverseMapping = DenseMap<Operation *, Value>;

public:
  /// Creates a state for the transformation rooted at the given op.
  explicit TransformState(Operation *root);

  /// Returns the op at which the transformation state is rooted. This is
  /// typically helpful for transformations that apply globally.
  Operation *getTopLevel() const;

  /// Returns the list of ops that the given transform IR value corresponds to.
  /// This is helpful for transformations that apply to a particular handle.
  ArrayRef<Operation *> getPayloadOps(Value value) const;

  /// Applies the transformation specified by the given transform op and updates
  /// the state accordingly.
  LogicalResult applyTransform(TransformOpInterface transform);

private:
  /// Identifier for storing top-level value in the `operations` mapping.
  static constexpr Value kTopLevelValue = Value();

  /// Sets the payload IR ops associated with the given transform IR value.
  /// Fails if this would result in multiple transform IR values with uses
  /// corresponding to the same payload IR ops. For example, a hypothetical
  /// "find function by name" transform op would (indirectly) call this
  /// function for its result. Having two such calls in a row with for different
  /// values, e.g. coming from different ops:
  ///
  ///   %0 = transform.find_func_by_name { name = "myfunc" }
  ///   %1 = transform.find_func_by_name { name = "myfunc" }
  ///
  /// would lead to both values pointing to the same operation. The second call
  /// to setPayloadOps will fail, unless the association with the %0 value is
  /// removed first by calling update/removePayloadOps.
  LogicalResult setPayloadOps(Value value, ArrayRef<Operation *> targets);

  /// Forgets the payload IR ops associated with the given transform IR value.
  void removePayloadOps(Value value);

  /// Updates the payload IR ops associated with the given transform IR value.
  /// The callback function is called once per associated operation and is
  /// expected to return the modified operation or nullptr. In the latter case,
  /// the corresponding operation is no longer associated with the transform IR
  /// value.
  void updatePayloadOps(Value value,
                        function_ref<Operation *(Operation *)> callback);

  /// The mapping between payload IR values and transform IR ops.
  TransformOpMapping operationMapping;
  TransformOpReverseMapping reverseMapping;
};

/// Local mapping between values defined by a specific op implementing the
/// TransformOpInterface and the payload IR ops they correspond to.
class TransformResults {
  friend class TransformState;

public:
  /// Indicates that the result of the transform IR op at the given position
  /// corresponds to the given list of payload IR ops. Each result must be set
  /// by the transformation exactly once.
  void set(OpResult value, ArrayRef<Operation *> ops);

private:
  /// Creates an instance of TransformResults that expects mappings for
  /// `numSegments` values.
  explicit TransformResults(unsigned numSegments);

  /// Gets the list of operations associated with the result identified by its
  /// number in the list of operation results.
  ArrayRef<Operation *> get(unsigned resultNumber) const;

  /// Storage for pointers to payload IR ops that are associated with results of
  /// a transform IR op. `segments` contains as many entries as the transform IR
  /// op has results. Each entry is a reference to a contiguous segment in
  /// the `operations` list that contains the pointers to operations. This
  /// allows for operations to be stored contiguously without nested vectors and
  /// for different segments to be set in any order.
  SmallVector<ArrayRef<Operation *>, 2> segments;
  SmallVector<Operation *> operations;
};

} // namespace transform
} // namespace mlir

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h.inc"

#endif // DIALECT_TRANSFORM_IR_TRANSFORMINTERFACES_H
