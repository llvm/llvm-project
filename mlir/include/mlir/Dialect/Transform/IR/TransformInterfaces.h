//===- TransformInterfaces.h - Transform Dialect Interfaces -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_TRANSFORMINTERFACES_H
#define MLIR_DIALECT_TRANSFORM_IR_TRANSFORMINTERFACES_H

#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace transform {

class TransformOpInterface;

/// Options controlling the application of transform operations by the
/// TransformState.
class TransformOptions {
public:
  TransformOptions() {}

  /// Requests computationally expensive checks of the transform and payload IR
  /// well-formedness to be performed before each transformation. In particular,
  /// these ensure that the handles still point to valid operations when used.
  TransformOptions &enableExpensiveChecks(bool enable = true) {
    expensiveChecksEnabled = enable;
    return *this;
  }

  /// Returns true if the expensive checks are requested.
  bool getExpensiveChecksEnabled() const { return expensiveChecksEnabled; }

private:
  bool expensiveChecksEnabled = true;
};

/// Entry point to the Transform dialect infrastructure. Applies the
/// transformation specified by `transform` to payload IR contained in
/// `payloadRoot`. The `transform` operation may contain other operations that
/// will be executed following the internal logic of the operation. It must
/// have the `PossibleTopLevelTransformOp` trait and not have any operands.
/// This function internally keeps track of the transformation state.
LogicalResult
applyTransforms(Operation *payloadRoot, TransformOpInterface transform,
                const TransformOptions &options = TransformOptions());

/// The state maintained across applications of various ops implementing the
/// TransformOpInterface. The operations implementing this interface and the
/// surrounding structure are referred to as transform IR. The operations to
/// which transformations apply are referred to as payload IR. Transform IR
/// operates on values that can be associated either with a list of payload IR
/// operations (such values are referred to as handles) or with a list of
/// parameters represented as attributes. The state thus contains the mapping
/// between values defined in the transform IR ops and either payload IR ops or
/// parameters. For payload ops, the mapping is many-to-many and the reverse
/// mapping is also stored. The "expensive-checks" option can be passed to the
/// constructor at transformation execution time that transform IR values used
/// as operands by a transform IR operation are not associated with dangling
/// pointers to payload IR operations that are known to have been erased by
/// previous transformation through the same or a different transform IR value.
///
/// A reference to this class is passed as an argument to "apply" methods of the
/// transform op interface. Thus the "apply" method can call either
/// `state.getPayloadOps( getSomeOperand() )` to obtain the list of operations
/// or `state.getParams( getSomeOperand() )` to obtain the list of parameters
/// associated with its operand. The method is expected to populate the
/// `TransformResults` class instance in order to update the mapping. The
/// `applyTransform` method takes care of propagating the state of
/// `TransformResults` into the instance of this class.
///
/// When applying transform IR operations with regions, the client is expected
/// to create a `RegionScope` RAII object to create a new "stack frame" for
/// values defined inside the region. The mappings from and to these values will
/// be automatically dropped when the object goes out of scope, typically at the
/// end of the `apply` function of the parent operation. If a region contains
/// blocks with arguments, the client can map those arguments to payload IR ops
/// using `mapBlockArguments`.
class TransformState {
public:
  using Param = Attribute;

private:
  /// Mapping between a Value in the transform IR and the corresponding set of
  /// operations in the payload IR.
  using TransformOpMapping = DenseMap<Value, SmallVector<Operation *, 2>>;

  /// Mapping between a payload IR operation and the transform IR values it is
  /// associated with.
  using TransformOpReverseMapping =
      DenseMap<Operation *, SmallVector<Value, 2>>;

  /// Mapping between a Value in the transform IR and the corresponding list of
  /// parameters.
  using ParamMapping = DenseMap<Value, SmallVector<Param>>;

  /// The bidirectional mappings between transform IR values and payload IR
  /// operations, and the mapping between transform IR values and parameters.
  struct Mappings {
    TransformOpMapping direct;
    TransformOpReverseMapping reverse;
    ParamMapping params;
  };

  friend LogicalResult applyTransforms(Operation *payloadRoot,
                                       TransformOpInterface transform,
                                       const TransformOptions &options);

public:
  /// Returns the op at which the transformation state is rooted. This is
  /// typically helpful for transformations that apply globally.
  Operation *getTopLevel() const;

  /// Returns the list of ops that the given transform IR value corresponds to.
  /// This is helpful for transformations that apply to a particular handle.
  ArrayRef<Operation *> getPayloadOps(Value value) const;

  /// Returns the list of parameters that the given transform IR value
  /// corresponds to.
  ArrayRef<Attribute> getParams(Value value) const;

  /// Populates `handles` with all handles pointing to the given Payload IR op.
  /// Returns success if such handles exist, failure otherwise.
  LogicalResult getHandlesForPayloadOp(Operation *op,
                                       SmallVectorImpl<Value> &handles) const;

  /// Applies the transformation specified by the given transform op and updates
  /// the state accordingly.
  DiagnosedSilenceableFailure applyTransform(TransformOpInterface transform);

  /// Records the mapping between a block argument in the transform IR and a
  /// list of operations in the payload IR. The arguments must be defined in
  /// blocks of the currently processed transform IR region, typically after a
  /// region scope is defined.
  ///
  /// Returns failure if the payload does not satisfy the conditions associated
  /// with the type of the handle value.
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

  /// Base class for TransformState extensions that allow TransformState to
  /// contain user-specified information in the state object. Clients are
  /// expected to derive this class, add the desired fields, and make the
  /// derived class compatible with the MLIR TypeID mechanism:
  ///
  /// ```mlir
  /// class MyExtension final : public TransformState::Extension {
  /// public:
  ///   MyExtension(TranfsormState &state, int myData)
  ///     : Extension(state) {...}
  /// private:
  ///   int mySupplementaryData;
  /// };
  /// ```
  ///
  /// Instances of this and derived classes are not expected to be created by
  /// the user, instead they are directly constructed within a TransformState. A
  /// TransformState can only contain one extension with the given TypeID.
  /// Extensions can be obtained from a TransformState instance, and can be
  /// removed when they are no longer required.
  ///
  /// ```mlir
  /// transformState.addExtension<MyExtension>(/*myData=*/42);
  /// MyExtension *ext = transformState.getExtension<MyExtension>();
  /// ext->doSomething();
  /// ```
  class Extension {
    // Allow TransformState to allocate Extensions.
    friend class TransformState;

  public:
    /// Base virtual destructor.
    // Out-of-line definition ensures symbols are emitted in a single object
    // file.
    virtual ~Extension();

  protected:
    /// Constructs an extension of the given TransformState object.
    Extension(TransformState &state) : state(state) {}

    /// Provides read-only access to the parent TransformState object.
    const TransformState &getTransformState() const { return state; }

    /// Replaces the given payload op with another op. If the replacement op is
    /// null, removes the association of the payload op with its handle. Returns
    /// failure if the op is not associated with any handle.
    LogicalResult replacePayloadOp(Operation *op, Operation *replacement);

  private:
    /// Back-reference to the state that is being extended.
    TransformState &state;
  };

  /// Adds a new Extension of the type specified as template parameter,
  /// constructing it with the arguments provided. The extension is owned by the
  /// TransformState. It is expected that the state does not already have an
  /// extension of the same type. Extension constructors are expected to take
  /// a reference to TransformState as first argument, automatically supplied
  /// by this call.
  template <typename Ty, typename... Args>
  Ty &addExtension(Args &&...args) {
    static_assert(
        std::is_base_of<Extension, Ty>::value,
        "only an class derived from TransformState::Extension is allowed here");
    auto ptr = std::make_unique<Ty>(*this, std::forward<Args>(args)...);
    auto result = extensions.try_emplace(TypeID::get<Ty>(), std::move(ptr));
    assert(result.second && "extension already added");
    return *static_cast<Ty *>(result.first->second.get());
  }

  /// Returns the extension of the specified type.
  template <typename Ty>
  Ty *getExtension() {
    static_assert(
        std::is_base_of<Extension, Ty>::value,
        "only an class derived from TransformState::Extension is allowed here");
    auto iter = extensions.find(TypeID::get<Ty>());
    if (iter == extensions.end())
      return nullptr;
    return static_cast<Ty *>(iter->second.get());
  }

  /// Removes the extension of the specified type.
  template <typename Ty>
  void removeExtension() {
    static_assert(
        std::is_base_of<Extension, Ty>::value,
        "only an class derived from TransformState::Extension is allowed here");
    extensions.erase(TypeID::get<Ty>());
  }

private:
  /// Identifier for storing top-level value in the `operations` mapping.
  static constexpr Value kTopLevelValue = Value();

  /// Creates a state for transform ops living in the given region. The second
  /// argument points to the root operation in the payload IR being transformed,
  /// which may or may not contain the region with transform ops. Additional
  /// options can be provided through the trailing configuration object.
  TransformState(Region *region, Operation *payloadRoot,
                 const TransformOptions &options = TransformOptions());

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

  /// Removes the mapping between the given payload IR operation and the given
  /// transform IR value.
  void dropReverseMapping(Mappings &mappings, Operation *op, Value value);

  /// Sets the payload IR ops associated with the given transform IR value
  /// (handle). A payload op may be associated multiple handles as long as
  /// at most one of them gets consumed by further transformations.
  /// For example, a hypothetical "find function by name" may be called twice in
  /// a row to produce two handles pointing to the same function:
  ///
  ///   %0 = transform.find_func_by_name { name = "myfunc" }
  ///   %1 = transform.find_func_by_name { name = "myfunc" }
  ///
  /// which is valid by itself. However, calling a hypothetical "rewrite and
  /// rename function" transform on both handles:
  ///
  ///   transform.rewrite_and_rename %0 { new_name = "func" }
  ///   transform.rewrite_and_rename %1 { new_name = "func" }
  ///
  /// is invalid given the transformation "consumes" the handle as expressed
  /// by side effects. Practically, a transformation consuming a handle means
  /// that the associated payload operation may no longer exist.
  ///
  /// Returns failure if the payload does not satisfy the conditions associated
  /// with the type of the handle value. The value is expected to have a type
  /// implementing TransformHandleTypeInterface.
  LogicalResult setPayloadOps(Value value, ArrayRef<Operation *> targets);

  /// Sets the parameters associated with the given transform IR value. Returns
  /// failure if the parameters do not satisfy the conditions associated with
  /// the type of the value. The value is expected to have a type implementing
  /// TransformParamTypeInterface.
  LogicalResult setParams(Value value, ArrayRef<Param> params);

  /// Forgets the payload IR ops associated with the given transform IR value.
  void removePayloadOps(Value value);

  /// Updates the payload IR ops associated with the given transform IR value.
  /// The callback function is called once per associated operation and is
  /// expected to return the modified operation or nullptr. In the latter case,
  /// the corresponding operation is no longer associated with the transform IR
  /// value.
  ///
  /// Returns failure if the payload does not satisfy the conditions associated
  /// with the type of the handle value.
  LogicalResult
  updatePayloadOps(Value value,
                   function_ref<Operation *(Operation *)> callback);

  /// If the operand is a handle consumed by the operation, i.e. has the "free"
  /// memory effect associated with it, identifies other handles that are
  /// pointing to payload IR operations nested in the operations pointed to by
  /// the consumed handle. Marks all such handles as invalidated to trigger
  /// errors if they are used.
  void recordHandleInvalidation(OpOperand &handle);
  void recordHandleInvalidationOne(OpOperand &handle, Operation *payloadOp,
                                   Value otherHandle);

  /// Checks that the operation does not use invalidated handles as operands.
  /// Reports errors and returns failure if it does. Otherwise, invalidates the
  /// handles consumed by the operation as well as any handles pointing to
  /// payload IR operations nested in the operations associated with the
  /// consumed handles.
  LogicalResult
  checkAndRecordHandleInvalidation(TransformOpInterface transform);

  /// The mappings between transform IR values and payload IR ops, aggregated by
  /// the region in which the transform IR values are defined.
  llvm::SmallDenseMap<Region *, Mappings> mappings;

  /// Extensions attached to the TransformState, identified by the TypeID of
  /// their type. Only one extension of any given type is allowed.
  DenseMap<TypeID, std::unique_ptr<Extension>> extensions;

  /// The top-level operation that contains all payload IR, typically a module.
  Operation *topLevel;

  /// Additional options controlling the transformation state behavior.
  TransformOptions options;

  /// The mapping from invalidated handles to the error-reporting functions that
  /// describe when the handles were invalidated. Calling such a function emits
  /// a user-visible diagnostic with an additional note pointing to the given
  /// location.
  DenseMap<Value, std::function<void(Location)>> invalidatedHandles;

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
  /// by the transformation exactly once. The value must have a type
  /// implementing TransformHandleTypeInterface.
  void set(OpResult value, ArrayRef<Operation *> ops);

  /// Indicates that the result of the transform IR op at the given position
  /// corresponds to the given list of parameters. Each result must be set by
  /// the transformation exactly once. The value must have a type implementing
  /// TransformParamTypeInterface.
  void setParams(OpResult value, ArrayRef<TransformState::Param> params);

private:
  /// Creates an instance of TransformResults that expects mappings for
  /// `numSegments` values, which may be associated with payload operations or
  /// parameters.
  explicit TransformResults(unsigned numSegments);

  /// Gets the list of operations associated with the result identified by its
  /// number in the list of operation results. The result must have been set to
  /// be associated with payload IR operations.
  ArrayRef<Operation *> get(unsigned resultNumber) const;

  /// Gets the list of parameters associated with the result identified by its
  /// number in the list of operation results. The result must have been set to
  /// be associated with parameters.
  ArrayRef<TransformState::Param> getParams(unsigned resultNumber) const;

  /// Returns `true` if the result identified by its number in the list of
  /// operation results is associated with a list of parameters, `false` if it
  /// is associated with the list of payload IR operations.
  bool isParam(unsigned resultNumber) const;

  /// Returns `true` if the result identified by its number in the list of
  /// operation results is associated with something.
  bool isSet(unsigned resultNumber) const;

  /// Storage for pointers to payload IR ops that are associated with results of
  /// a transform IR op. `segments` contains as many entries as the transform IR
  /// op has results, even if some of them are not associated with payload IR
  /// operations. Each entry is a reference to a contiguous segment in the
  /// `operations` list that contains the pointers to operations. This allows
  /// for operations to be stored contiguously without nested vectors and for
  /// different segments to be set in any order.
  SmallVector<ArrayRef<Operation *>, 2> segments;
  SmallVector<Operation *> operations;

  /// Storage for parameters that are associated with results of the transform
  /// IR op. `paramSegments` contains as many entries as the transform IR op has
  /// results, even if some of them are not associated with parameters. Each
  /// entry is a reference to a contiguous segment in the `params` list that
  /// contains the actual parameters. This allows for parameters to be stored
  /// contiguously without nested vectors and for different segments to be set
  /// in any order.
  SmallVector<ArrayRef<TransformState::Param>, 2> paramSegments;
  SmallVector<TransformState::Param> params;
};

TransformState::RegionScope TransformState::make_region_scope(Region &region) {
  return RegionScope(*this, region);
}

namespace detail {
/// Maps the only block argument of the op with PossibleTopLevelTransformOpTrait
/// to either the list of operations associated with its operand or the root of
/// the payload IR, depending on what is available in the context.
LogicalResult
mapPossibleTopLevelTransformOpBlockArguments(TransformState &state,
                                             Operation *op, Region &region);

/// Verification hook for PossibleTopLevelTransformOpTrait.
LogicalResult verifyPossibleTopLevelTransformOpTrait(Operation *op);

/// Verification hook for TransformOpInterface.
LogicalResult verifyTransformOpInterface(Operation *op);
} // namespace detail

/// This trait is supposed to be attached to Transform dialect operations that
/// can be standalone top-level transforms. Such operations typically contain
/// other Transform dialect operations that can be executed following some
/// control flow logic specific to the current operation. The operations with
/// this trait are expected to have at least one single-block region with one
/// argument of PDL Operation type. The operations are also expected to be valid
/// without operands, in which case they are considered top-level, and with one
/// or more arguments, in which case they are considered nested. Top-level
/// operations have the block argument of the entry block in the Transform IR
/// correspond to the root operation of Payload IR. Nested operations have the
/// block argument of the entry block in the Transform IR correspond to a list
/// of Payload IR operations mapped to the first operand of the Transform IR
/// operation. The operation must implement TransformOpInterface.
template <typename OpTy>
class PossibleTopLevelTransformOpTrait
    : public OpTrait::TraitBase<OpTy, PossibleTopLevelTransformOpTrait> {
public:
  /// Verifies that `op` satisfies the invariants of this trait. Not expected to
  /// be called directly.
  static LogicalResult verifyTrait(Operation *op) {
    return detail::verifyPossibleTopLevelTransformOpTrait(op);
  }

  /// Returns the single block of the given region.
  Block *getBodyBlock(unsigned region = 0) {
    return &this->getOperation()->getRegion(region).front();
  }

  /// Sets up the mapping between the entry block of the given region of this op
  /// and the relevant list of Payload IR operations in the given state. The
  /// state is expected to be already scoped at the region of this operation.
  LogicalResult mapBlockArguments(TransformState &state, Region &region) {
    assert(region.getParentOp() == this->getOperation() &&
           "op comes from the wrong region");
    return detail::mapPossibleTopLevelTransformOpBlockArguments(
        state, this->getOperation(), region);
  }
  LogicalResult mapBlockArguments(TransformState &state) {
    assert(
        this->getOperation()->getNumRegions() == 1 &&
        "must indicate the region to map if the operation has more than one");
    return mapBlockArguments(state, this->getOperation()->getRegion(0));
  }
};

/// Trait implementing the TransformOpInterface for operations applying a
/// transformation to a single operation handle and producing an arbitrary
/// number of handles and parameter values.
/// The op must implement a method with the following signature:
///   - DiagnosedSilenceableFailure applyToOne(OpTy,
///       SmallVector<Operation*> &results, state)
/// to perform a transformation that is applied in turn to all payload IR
/// operations that correspond to the handle of the transform IR operation.
/// In `applyToOne`, OpTy is either Operation* or a concrete payload IR Op class
/// that the transformation is applied to (and NOT the class of the transform IR
/// op).
/// The `applyToOne` method takes an empty `results` vector that it fills with
/// zero, one or multiple operations depending on the number of resultd expected
/// by the transform op.
/// The number of results must match the number of results of the transform op.
/// `applyToOne` is allowed to fill the `results` with all null elements to
/// signify that the transformation did not apply to the payload IR operations.
/// Such null elements are filtered out from results before return.
///
/// The transform op having this trait is expected to have a single operand.
template <typename OpTy>
class TransformEachOpTrait
    : public OpTrait::TraitBase<OpTy, TransformEachOpTrait> {
public:
  /// Calls `applyToOne` for every payload operation associated with the operand
  /// of this transform IR op, the following case disjunction happens:
  ///   1. If not target payload ops are associated to the operand then fill the
  ///      results vector with the expected number of null elements and return
  ///      success. This is the corner case handling that allows propagating
  ///      the "no-op" case gracefully to improve usability.
  ///   2. If any `applyToOne` returns definiteFailure, the transformation is
  ///      immediately considered definitely failed and we return.
  ///   3. All applications of `applyToOne` are checked to return a number of
  ///      results expected by the transform IR op. If not, this is a definite
  ///      failure and we return early.
  ///   4. If `applyToOne` produces ops, associate them with the result of this
  ///      transform op.
  ///   5. If any `applyToOne` return silenceableFailure, the transformation is
  ///      considered silenceable.
  ///   6. Otherwise the transformation is considered successful.
  DiagnosedSilenceableFailure apply(TransformResults &transformResults,
                                    TransformState &state);

  /// Checks that the op matches the expectations of this trait.
  static LogicalResult verifyTrait(Operation *op);
};

/// Side effect resource corresponding to the mapping between Transform IR
/// values and Payload IR operations. An Allocate effect from this resource
/// means creating a new mapping entry, it is always accompanied by a Write
/// effet. A Read effect from this resource means accessing the mapping. A Free
/// effect on this resource indicates the removal of the mapping entry,
/// typically after a transformation that modifies the Payload IR operations
/// associated with one of the Transform IR operation's operands. It is always
/// accompanied by a Read effect. Read-after-Free and double-Free are not
/// allowed (they would be problematic with "regular" memory effects too) as
/// they indicate an attempt to access Payload IR operations that have been
/// modified, potentially erased, by the previous tranfsormations.
// TODO: consider custom effects if these are not enabling generic passes such
// as CSE/DCE to work.
struct TransformMappingResource
    : public SideEffects::Resource::Base<TransformMappingResource> {
  StringRef getName() override { return "transform.mapping"; }
};

/// Side effect resource corresponding to the Payload IR itself. Only Read and
/// Write effects are expected on this resource, with Write always accompanied
/// by a Read (short of fully replacing the top-level Payload IR operation, one
/// cannot modify the Payload IR without reading it first). This is intended
/// to disallow reordering of Transform IR operations that mutate the Payload IR
/// while still allowing the reordering of those that only access it.
struct PayloadIRResource
    : public SideEffects::Resource::Base<PayloadIRResource> {
  StringRef getName() override { return "transform.payload_ir"; }
};

/// Populates `effects` with the memory effects indicating the operation on the
/// given handle value:
///   - consumes = Read + Free,
///   - produces = Allocate + Write,
///   - onlyReads = Read.
void consumesHandle(ValueRange handles,
                    SmallVectorImpl<MemoryEffects::EffectInstance> &effects);
void producesHandle(ValueRange handles,
                    SmallVectorImpl<MemoryEffects::EffectInstance> &effects);
void onlyReadsHandle(ValueRange handles,
                     SmallVectorImpl<MemoryEffects::EffectInstance> &effects);

/// Checks whether the transform op consumes the given handle.
bool isHandleConsumed(Value handle, transform::TransformOpInterface transform);

/// Populates `effects` with the memory effects indicating the access to payload
/// IR resource.
void modifiesPayload(SmallVectorImpl<MemoryEffects::EffectInstance> &effects);
void onlyReadsPayload(SmallVectorImpl<MemoryEffects::EffectInstance> &effects);

/// Trait implementing the MemoryEffectOpInterface for operations that "consume"
/// their operands and produce new results.
template <typename OpTy>
class FunctionalStyleTransformOpTrait
    : public OpTrait::TraitBase<OpTy, FunctionalStyleTransformOpTrait> {
public:
  /// This op "consumes" the operands by reading and freeing then, "produces"
  /// the results by allocating and writing it and reads/writes the payload IR
  /// in the process.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    consumesHandle(this->getOperation()->getOperands(), effects);
    producesHandle(this->getOperation()->getResults(), effects);
    modifiesPayload(effects);
  }

  /// Checks that the op matches the expectations of this trait.
  static LogicalResult verifyTrait(Operation *op) {
    if (!op->getName().getInterface<MemoryEffectOpInterface>()) {
      op->emitError()
          << "FunctionalStyleTransformOpTrait should only be attached to ops "
             "that implement MemoryEffectOpInterface";
    }
    return success();
  }
};

/// Trait implementing the MemoryEffectOpInterface for single-operand
/// single-result operations that use their operand without consuming and
/// without modifying the Payload IR to produce a new handle.
template <typename OpTy>
class NavigationTransformOpTrait
    : public OpTrait::TraitBase<OpTy, NavigationTransformOpTrait> {
public:
  /// This op produces handles to the Payload IR without consuming the original
  /// handles and without modifying the IR itself.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    onlyReadsHandle(this->getOperation()->getOperands(), effects);
    producesHandle(this->getOperation()->getResults(), effects);
    onlyReadsPayload(effects);
  }

  /// Checks that the op matches the expectation of this trait.
  static LogicalResult verifyTrait(Operation *op) {
    static_assert(OpTy::template hasTrait<OpTrait::OneOperand>(),
                  "expected single-operand op");
    static_assert(OpTy::template hasTrait<OpTrait::OneResult>(),
                  "expected single-result op");
    if (!op->getName().getInterface<MemoryEffectOpInterface>()) {
      op->emitError() << "NavigationTransformOpTrait should only be attached "
                         "to ops that implement MemoryEffectOpInterface";
    }
    return success();
  }
};

namespace detail {
/// Non-template implementation of ParamProducerTransformOpTrait::getEffects().
void getParamProducerTransformOpTraitEffects(
    Operation *op, SmallVectorImpl<MemoryEffects::EffectInstance> &effects);
/// Non-template implementation of ParamProducerTransformOpTrait::verify().
LogicalResult verifyParamProducerTransformOpTrait(Operation *op);
} // namespace detail

/// Trait implementing the MemoryEffectsOpInterface for operations that produce
/// transform dialect parameters. It marks all op results of
/// TransformHandleTypeInterface as produced by the op, all operands as only
/// read by the op and, if at least one of the operand is a handle to payload
/// ops, the entire payload as potentially read. The op must only produce
/// parameter-typed results.
template <typename OpTy>
class ParamProducerTransformOpTrait
    : public OpTrait::TraitBase<OpTy, ParamProducerTransformOpTrait> {
public:
  /// Populates `effects` with effect instances described in the trait
  /// documentation.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    detail::getParamProducerTransformOpTraitEffects(this->getOperation(),
                                                    effects);
  }

  /// Checks that the op matches the expectation of this trait, i.e., that it
  /// implements the MemoryEffectsOpInterface and only produces parameter-typed
  /// results.
  static LogicalResult verifyTrait(Operation *op) {
    return detail::verifyParamProducerTransformOpTrait(op);
  }
};

} // namespace transform
} // namespace mlir

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h.inc"

namespace mlir {
namespace transform {

/// A single result of applying a transform op with `ApplyEachOpTrait` to a
/// single payload operation.
using ApplyToEachResult = llvm::PointerUnion<Operation *, Attribute>;

/// A list of results of applying a transform op with `ApplyEachOpTrait` to a
/// single payload operation, co-indexed with the results of the transform op.
class ApplyToEachResultList {
public:
  ApplyToEachResultList() = default;
  explicit ApplyToEachResultList(unsigned size) : results(size) {}

  /// Sets the list of results to `size` null pointers.
  void assign(unsigned size, std::nullptr_t) { results.assign(size, nullptr); }

  /// Sets the list of results to the given range of values.
  template <typename Range>
  void assign(Range &&range) {
    // This is roughly the implementation of SmallVectorImpl::assign.
    // Dispatching to it with map_range and template type inference would result
    // in more complex code here.
    results.clear();
    results.reserve(llvm::size(range));
    for (auto element : range) {
      if constexpr (std::is_convertible_v<decltype(*std::begin(range)),
                                          Operation *>) {
        results.push_back(static_cast<Operation *>(element));
      } else {
        results.push_back(static_cast<Attribute>(element));
      }
    }
  }

  /// Appends an element to the list.
  void push_back(Operation *op) { results.push_back(op); }
  void push_back(Attribute attr) { results.push_back(attr); }

  /// Reserves space for `size` elements in the list.
  void reserve(unsigned size) { results.reserve(size); }

  /// Iterators over the list.
  auto begin() { return results.begin(); }
  auto end() { return results.end(); }
  auto begin() const { return results.begin(); }
  auto end() const { return results.end(); }

  /// Returns the number of elements in the list.
  size_t size() const { return results.size(); }

  /// Element access. Expects the index to be in bounds.
  ApplyToEachResult &operator[](size_t index) { return results[index]; }
  const ApplyToEachResult &operator[](size_t index) const {
    return results[index];
  }

private:
  /// Underlying storage.
  SmallVector<ApplyToEachResult> results;
};

namespace detail {

/// Check that the contents of `partialResult` matches the number, kind (payload
/// op or parameter) and nullity (either all or none) requirements of
/// `transformOp`. Report errors and return failure otherwise.
LogicalResult checkApplyToOne(Operation *transformOp, Location payloadOpLoc,
                              const ApplyToEachResultList &partialResult);

/// "Transpose" the results produced by individual applications, arranging them
/// per result value of the transform op, and populate `transformResults` with
/// that. The number, kind and nullity of per-application results are assumed to
/// have been verified.
void setApplyToOneResults(Operation *transformOp,
                          TransformResults &transformResults,
                          ArrayRef<ApplyToEachResultList> results);

/// Applies a one-to-one or a one-to-many transform to each of the given
/// targets. Puts the results of transforms, if any, in `results` in the same
/// order. Fails if any of the application fails. Individual transforms must be
/// callable with the following signature:
///   - DiagnosedSilenceableFailure(OpTy,
///       SmallVector<Operation*> &results, state)
/// where OpTy is either
///   - Operation *, in which case the transform is always applied;
///   - a concrete Op class, in which case a check is performed whether
///   `targets` contains operations of the same class and a silenceable failure
///   is reported if it does not.
template <typename TransformOpTy>
DiagnosedSilenceableFailure
applyTransformToEach(TransformOpTy transformOp, ArrayRef<Operation *> targets,
                     SmallVectorImpl<ApplyToEachResultList> &results,
                     TransformState &state) {
  using OpTy = typename llvm::function_traits<
      decltype(&TransformOpTy::applyToOne)>::template arg_t<0>;
  static_assert(std::is_convertible<OpTy, Operation *>::value,
                "expected transform function to take an operation");

  SmallVector<Diagnostic> silenceableStack;
  unsigned expectedNumResults = transformOp->getNumResults();
  for (Operation *target : targets) {
    auto specificOp = dyn_cast<OpTy>(target);
    if (!specificOp) {
      Diagnostic diag(transformOp->getLoc(), DiagnosticSeverity::Error);
      diag << "transform applied to the wrong op kind";
      diag.attachNote(target->getLoc()) << "when applied to this op";
      silenceableStack.push_back(std::move(diag));
      continue;
    }

    ApplyToEachResultList partialResults;
    partialResults.reserve(expectedNumResults);
    Location specificOpLoc = specificOp->getLoc();
    DiagnosedSilenceableFailure res =
        transformOp.applyToOne(specificOp, partialResults, state);
    if (res.isDefiniteFailure())
      return DiagnosedSilenceableFailure::definiteFailure();

    if (res.isSilenceableFailure()) {
      res.takeDiagnostics(silenceableStack);
      continue;
    }

    if (failed(detail::checkApplyToOne(transformOp, specificOpLoc,
                                       partialResults))) {
      return DiagnosedSilenceableFailure::definiteFailure();
    }
    results.push_back(std::move(partialResults));
  }
  if (!silenceableStack.empty()) {
    return DiagnosedSilenceableFailure::silenceableFailure(
        std::move(silenceableStack));
  }
  return DiagnosedSilenceableFailure::success();
}

} // namespace detail
} // namespace transform
} // namespace mlir

template <typename OpTy>
mlir::DiagnosedSilenceableFailure
mlir::transform::TransformEachOpTrait<OpTy>::apply(
    TransformResults &transformResults, TransformState &state) {
  ArrayRef<Operation *> targets =
      state.getPayloadOps(this->getOperation()->getOperand(0));

  // Step 1. Handle the corner case where no target is specified.
  // This is typically the case when the matcher fails to apply and we need to
  // propagate gracefully.
  // In this case, we fill all results with an empty vector.
  if (targets.empty()) {
    SmallVector<Operation *> emptyPayload;
    SmallVector<Attribute> emptyParams;
    for (OpResult r : this->getOperation()->getResults()) {
      if (r.getType().isa<TransformParamTypeInterface>())
        transformResults.setParams(r, emptyParams);
      else
        transformResults.set(r, emptyPayload);
    }
    return DiagnosedSilenceableFailure::success();
  }

  // Step 2. Call applyToOne on each target and record newly produced ops in its
  // corresponding results entry.
  SmallVector<ApplyToEachResultList, 1> results;
  results.reserve(targets.size());
  DiagnosedSilenceableFailure result = detail::applyTransformToEach(
      cast<OpTy>(this->getOperation()), targets, results, state);

  // Step 3. Propagate the definite failure if any and bail out.
  if (result.isDefiniteFailure())
    return result;

  // Step 4. "Transpose" the results produced by individual applications,
  // arranging them per result value of the transform op. The number, kind and
  // nullity of per-application results have been verified by the callback
  // above.
  detail::setApplyToOneResults(this->getOperation(), transformResults, results);

  // Step 5. ApplyToOne may have returned silenceableFailure, propagate it.
  return result;
}

template <typename OpTy>
mlir::LogicalResult
mlir::transform::TransformEachOpTrait<OpTy>::verifyTrait(Operation *op) {
  static_assert(OpTy::template hasTrait<OpTrait::OneOperand>(),
                "expected single-operand op");
  if (!op->getName().getInterface<TransformOpInterface>()) {
    return op->emitError() << "TransformEachOpTrait should only be attached to "
                              "ops that implement TransformOpInterface";
  }

  return success();
}

#endif // DIALECT_TRANSFORM_IR_TRANSFORMINTERFACES_H
