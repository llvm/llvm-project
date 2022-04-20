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
///
/// When applying transform IR operations with regions, the client is expected
/// to create a RegionScope RAII object to create a new "stack frame" for
/// values defined inside the region. The mappings from and to these values will
/// be automatically dropped when the object goes out of scope, typically at the
/// end of the "apply" function of the parent operation. If a region contains
/// blocks with arguments, the client can map those arguments to payload IR ops
/// using "mapBlockArguments".
class TransformState {
  /// Mapping between a Value in the transform IR and the corresponding set of
  /// operations in the payload IR.
  using TransformOpMapping = DenseMap<Value, SmallVector<Operation *>>;

  /// Mapping between a payload IR operation and the transform IR value it is
  /// currently associated with.
  using TransformOpReverseMapping = DenseMap<Operation *, Value>;

  /// Bidirectional mappings between transform IR values and payload IR
  /// operations.
  struct Mappings {
    TransformOpMapping direct;
    TransformOpReverseMapping reverse;
  };

public:
  /// Creates a state for transform ops living in the given region. The parent
  /// operation of the region. The second argument points to the root operation
  /// in the payload IR beind transformed, which may or may not contain the
  /// region with transform ops.
  TransformState(Region &region, Operation *root);

  /// Returns the op at which the transformation state is rooted. This is
  /// typically helpful for transformations that apply globally.
  Operation *getTopLevel() const;

  /// Returns the list of ops that the given transform IR value corresponds to.
  /// This is helpful for transformations that apply to a particular handle.
  ArrayRef<Operation *> getPayloadOps(Value value) const;

  /// Applies the transformation specified by the given transform op and updates
  /// the state accordingly.
  LogicalResult applyTransform(TransformOpInterface transform);

  /// Records the mapping between a block argument in the transform IR and a
  /// list of operations in the payload IR. The arguments must be defined in
  /// blocks of the currently processed transform IR region, typically after a
  /// region scope is defined.
  LogicalResult mapBlockArguments(BlockArgument argument,
                                  ArrayRef<Operation *> operations) {
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
    assert(argument.getParentRegion() == regionStack.back() &&
           "mapping block arguments from a region other than the active one");
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
    return setPayloadOps(argument, operations);
  }

  // Forward declarations to support limited visibility.
  class RegionScope;

  /// Creates a new region scope for the given region. The region is expected to
  /// be nested in the currently processed region.
  // Implementation note: this method is inline but implemented outside of the
  // class body to comply with visibility and full-declaration requirements.
  inline RegionScope make_region_scope(Region &region);

  /// A RAII object maintaining a "stack frame" for a transform IR region. When
  /// applying a transform IR operation that contains a region, the caller is
  /// expected to create a RegionScope before applying the ops contained in the
  /// region. This ensures that the mappings between values defined in the
  /// transform IR region and payload IR operations are cleared when the region
  /// processing ends; such values cannot be accessed outside the region.
  class RegionScope {
  public:
    /// Forgets the mapping from or to values defined in the associated
    /// transform IR region.
    ~RegionScope() {
      state.mappings.erase(region);
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
      state.regionStack.pop_back();
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
    }

  private:
    /// Creates a new scope for mappings between values defined in the given
    /// transform IR region and payload IR operations.
    RegionScope(TransformState &state, Region &region)
        : state(state), region(&region) {
      auto res = state.mappings.try_emplace(this->region);
      assert(res.second && "the region scope is already present");
      (void)res;
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
      assert(state.regionStack.back()->isProperAncestor(&region) &&
             "scope started at a non-nested region");
      state.regionStack.push_back(&region);
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
    }

    /// Back-reference to the transform state.
    TransformState &state;

    /// The region this scope is associated with.
    Region *region;

    friend RegionScope TransformState::make_region_scope(Region &);
  };
  friend class RegionScope;

private:
  /// Identifier for storing top-level value in the `operations` mapping.
  static constexpr Value kTopLevelValue = Value();

  /// Returns the mappings frame for the reigon in which the value is defined.
  const Mappings &getMapping(Value value) const {
    return const_cast<TransformState *>(this)->getMapping(value);
  }
  Mappings &getMapping(Value value) {
    auto it = mappings.find(value.getParentRegion());
    assert(it != mappings.end() &&
           "trying to find a mapping for a value from an unmapped region");
    return it->second;
  }

  /// Returns the mappings frame for the region in which the operation resides.
  const Mappings &getMapping(Operation *operation) const {
    return const_cast<TransformState *>(this)->getMapping(operation);
  }
  Mappings &getMapping(Operation *operation) {
    auto it = mappings.find(operation->getParentRegion());
    assert(it != mappings.end() &&
           "trying to find a mapping for an operation from an unmapped region");
    return it->second;
  }

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

  /// The mappings between transform IR values and payload IR ops, aggregated by
  /// the region in which the transform IR values are defined.
  llvm::SmallDenseMap<Region *, Mappings> mappings;

  /// The top-level operation that contains all payload IR, typically a module.
  Operation *topLevel;

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  /// A stack of nested regions that are being processed in the transform IR.
  /// Each region must be an ancestor of the following regions in this list.
  /// These are also the keys for "mappings".
  SmallVector<Region *> regionStack;
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
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

TransformState::RegionScope TransformState::make_region_scope(Region &region) {
  return RegionScope(*this, region);
}

} // namespace transform
} // namespace mlir

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h.inc"

#endif // DIALECT_TRANSFORM_IR_TRANSFORMINTERFACES_H
