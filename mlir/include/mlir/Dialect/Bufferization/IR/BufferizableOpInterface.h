//===- BufferizableOpInterface.h - Bufferizable Ops -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZABLEOPINTERFACE_H_
#define MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZABLEOPINTERFACE_H_

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
class OpBuilder;

namespace bufferization {

class AnalysisState;
class BufferizableOpInterface;
struct DialectAnalysisState;

class OpFilter {
public:
  /// An op filter entry. Filters can be used to specify which ops should be
  /// processed by the bufferization.
  struct Entry {
    /// If the filter function evaluates to `true`, the filter matches.
    using FilterFn = std::function<bool(Operation *)>;

    /// Filter type: A filter can either be a DENY filter or an ALLOW filter.
    enum FilterType : int8_t { DENY = 0, ALLOW = 1 };

    FilterFn fn;
    FilterType type;
  };

  /// Return whether the op is allowed or not.
  ///
  /// If the filter does not have an ALLOW rule, ops are allowed by default,
  /// unless they are explicitly marked as DENY. If the filter has at least one
  /// ALLOW rule, ops are denied by default and only allowed if they match
  /// an ALLOW rule and no DENY rule.
  bool isOpAllowed(Operation *op) const;

  /// Allow the given dialects.
  ///
  /// This function adds one or multiple ALLOW entries.
  template <typename... DialectTs> void allowDialect() {
    // The following expands a call to allowDialectImpl for each dialect
    // in 'DialectTs'. This magic is necessary due to a limitation in the places
    // that a parameter pack can be expanded in c++11.
    // FIXME: In c++17 this can be simplified by using 'fold expressions'.
    (void)std::initializer_list<int>{0, (allowDialectImpl<DialectTs>(), 0)...};
  }

  /// Deny the given dialects.
  ///
  /// This function adds one or multiple DENY entries.
  template <typename... DialectTs> void denyDialect() {
    // FIXME: In c++17 this can be simplified by using 'fold expressions'.
    (void)std::initializer_list<int>{0, (denyDialectImpl<DialectTs>(), 0)...};
  }

  /// Allow the given dialect.
  ///
  /// This function adds an ALLOW entry.
  void allowDialect(StringRef dialectNamespace) {
    Entry::FilterFn filterFn = [=](Operation *op) {
      return op->getDialect()->getNamespace() == dialectNamespace;
    };
    entries.push_back(Entry{filterFn, Entry::FilterType::ALLOW});
  }

  /// Allow the given ops.
  ///
  /// This function adds one or multiple ALLOW entries.
  template <typename... OpTys> void allowOperation() {
    // FIXME: In c++17 this can be simplified by using 'fold expressions'.
    (void)std::initializer_list<int>{0, (allowOperationImpl<OpTys>(), 0)...};
  }

  /// Deny the given ops.
  ///
  /// This function adds one or multiple DENY entries.
  template <typename... OpTys> void denyOperation() {
    // FIXME: In c++17 this can be simplified by using 'fold expressions'.
    (void)std::initializer_list<int>{0, (denyOperationImpl<OpTys>(), 0)...};
  }

  /// Allow the given op.
  ///
  /// This function adds an ALLOW entry.
  void allowOperation(StringRef opName) {
    Entry::FilterFn filterFn = [=](Operation *op) {
      return op->getName().getStringRef() == opName;
    };
    allowOperation(filterFn);
  }

  /// Deny the given op.
  ///
  /// This function adds a DENY entry.
  void denyOperation(StringRef opName) {
    Entry::FilterFn filterFn = [=](Operation *op) {
      return op->getName().getStringRef() == opName;
    };
    denyOperation(filterFn);
  }

  /// Allow ops that are matched by `fn`.
  ///
  /// This function adds an ALLOW entry.
  void allowOperation(Entry::FilterFn fn) {
    entries.push_back(Entry{fn, Entry::FilterType::ALLOW});
  }

  /// Deny ops that are matched by `fn`.
  ///
  /// This function adds a DENY entry.
  void denyOperation(Entry::FilterFn fn) {
    entries.push_back(Entry{fn, Entry::FilterType::DENY});
  }

private:
  /// Return `true` if the filter has at least one ALLOW rule.
  bool hasAllowRule() const {
    for (const Entry &e : entries)
      if (e.type == Entry::FilterType::ALLOW)
        return true;
    return false;
  }

  /// Allow a dialect.
  template <typename DialectT> void allowDialectImpl() {
    allowDialect(DialectT::getDialectNamespace());
  }

  /// Deny a dialect.
  template <typename DialectT> void denyDialectImpl() {
    denyDialect(DialectT::getDialectNamespace());
  }

  /// Allow an op.
  template <typename OpTy> void allowOperationImpl() {
    allowOperation(OpTy::getOperationName());
  }

  /// Deny an op.
  template <typename OpTy> void denyOperationImpl() {
    denyOperation(OpTy::getOperationName());
  }

  /// A list of filter entries that determine whether an op should be allowed or
  /// denied. If the filter has an ALLOW rule, only ops that are allowed and not
  /// denied are allowed. If the filter does not have an ALLOW rule, only ops
  /// that are not denied are allowed.
  SmallVector<Entry> entries;
};

/// Options for BufferizableOpInterface-based bufferization.
struct BufferizationOptions {
  /// Allocator function: Generate a memref allocation with the given type,
  /// dynamic extents and alignment.
  using AllocationFn = std::function<FailureOr<Value>(
      OpBuilder &, Location, MemRefType, ValueRange, unsigned int)>;
  /// Deallocator function: Deallocate a buffer that was allocated with
  /// AllocatorFn.
  using DeallocationFn =
      std::function<LogicalResult(OpBuilder &, Location, Value)>;
  /// Memcpy function: Generate a memcpy between two buffers.
  using MemCpyFn =
      std::function<LogicalResult(OpBuilder &, Location, Value, Value)>;
  /// Initializer function for analysis state.
  using AnalysisStateInitFn = std::function<void(AnalysisState &)>;
  /// Initializer function for dialect-specific analysis state.
  using DialectStateInitFn =
      std::function<std::unique_ptr<DialectAnalysisState>()>;

  enum class LayoutMapOption : int8_t {
    InferLayoutMap = 0,
    IdentityLayoutMap = 1,
    FullyDynamicLayoutMap = 2
  };

  BufferizationOptions();

  /// Try to cast the given op to BufferizableOpInterface if the op is allow
  /// listed.
  BufferizableOpInterface dynCastBufferizableOp(Operation *op) const;

  /// Try to cast the given value to BufferizableOpInterface if the op is allow
  /// listed.
  BufferizableOpInterface dynCastBufferizableOp(Value value) const;

  /// A filter that specifies which ops should be bufferized and which ops
  /// should be ignored.
  OpFilter opFilter;

  /// Return `true` if the given op should be bufferized.
  bool isOpAllowed(Operation *op) const;

  /// Helper functions for allocation, deallocation, memory copying.
  Optional<AllocationFn> allocationFn;
  Optional<DeallocationFn> deallocationFn;
  Optional<MemCpyFn> memCpyFn;

  /// Create a memref allocation with the given type and dynamic extents.
  FailureOr<Value> createAlloc(OpBuilder &b, Location loc, MemRefType type,
                               ValueRange dynShape) const;

  /// Creates a memref deallocation. The given memref buffer must have been
  /// allocated using `createAlloc`.
  LogicalResult createDealloc(OpBuilder &b, Location loc,
                              Value allocatedBuffer) const;

  /// Creates a memcpy between two given buffers.
  LogicalResult createMemCpy(OpBuilder &b, Location loc, Value from,
                             Value to) const;

  /// Specifies whether not bufferizable ops are allowed in the input. If so,
  /// bufferization.to_memref and bufferization.to_tensor ops are inserted at
  /// the boundaries.
  bool allowUnknownOps = false;

  /// Specifies whether function boundaries (ops in the func dialect) should be
  /// bufferized or not.
  bool bufferizeFunctionBoundaries = false;

  /// Certain ops have aliasing OpOperand/OpResult invariants (e.g., scf.for).
  /// If this flag is set to `false`, those invariants are no longer enforced
  /// with buffer copies.
  ///
  /// Note: Deactivating this flag can lead to incorrect bufferization results
  /// when used incorrectly. This flag is useful with
  /// `AlwaysCopyBufferizationState` which bufferizes all writing tensor
  /// OpOperands out-of-place.
  bool enforceAliasingInvariants = true;

  /// This flag controls buffer types on function signatures.
  ///
  /// * InferLayoutMap: All function parameter types have a fully dynamic layout
  ///   map, but function result types are inferred from the body of the
  ///   function.
  /// * FullyDynamicLayoutMap: All function parameter types and result types
  ///   have a fully dynamic layout map. This option is most efficient because
  ///   any layout map can be casted to a fully dynamic one.
  /// * IdentityLayoutMap: All function parameter types and result types have a
  ///   static identity layout (i.e., no layout map). This option may introduce
  ///   additional buffer allocs and copies because layout maps cannot be casted
  ///   away.
  ///
  /// If `bufferizeFunctionBoundaries` is not set, this flag has no effect.
  ///
  /// Note: Inferred layout maps may not be desireable when interacting with
  /// external functions, because the generated function signatures will be less
  /// predictable.
  LayoutMapOption functionBoundaryTypeConversion =
      LayoutMapOption::InferLayoutMap;

  /// This flag controls buffer types on unknown ops (to_memref wrappers) and in
  /// other cases where a precise memref type cannot be inferred (e.g., the
  /// bufferization of "tensor.cast").
  ///
  /// * InferLayoutMap: This option is invalid and cannot be used.
  /// * FullyDynamicLayoutMap: Assume that unknown ops have results with fully
  ///   dynamic layout maps after bufferization. This option is most efficient
  ///   because any layout map can be casted to a fully dynamic one.
  /// * IdentityLayoutMap: Assume that unknown ops have results with static
  ///   identity layout (i.e., no layout map) after bufferization. This option
  ///   introduces additional buffer allocs and copies if the unknown op is
  ///   eventually bufferized to an op that returns a buffer with non-identity
  ///   layout.
  LayoutMapOption unknownTypeConversion =
      LayoutMapOption::FullyDynamicLayoutMap;

  /// Specifies whether dealloc ops should be generated along with alloc ops. If
  /// not, new memory allocations will leak.
  bool createDeallocs = true;

  /// Seed for the analysis fuzzer. If set to `0`, the fuzzer is deactivated.
  /// Should be used only with `testAnalysisOnly = true`.
  unsigned analysisFuzzerSeed = 0;

  /// If set to `true`, does not modify the IR apart from adding attributes (for
  /// checking the results of the analysis) and post analysis steps.
  bool testAnalysisOnly = false;

  /// If set to `true`, the IR is annotated with details about RaW conflicts.
  /// For debugging only. Should be used together with `testAnalysisOnly`.
  bool printConflicts = false;

  /// Buffer alignment for new memory allocations.
  unsigned int bufferAlignment = 128;

  /// Initializer functions for analysis state. These can be used to
  /// initialize dialect-specific analysis state.
  SmallVector<AnalysisStateInitFn> stateInitializers;

  /// Add a analysis state initializer that initializes the specified
  /// dialect-specific analysis state.
  void addDialectStateInitializer(StringRef name, const DialectStateInitFn &fn);
};

/// Specify fine-grain relationship between buffers to enable more analysis.
enum class BufferRelation {
  None,
  // TODO: ResultContainsOperand,
  // TODO: OperandContainsResult,
  Equivalent
};

/// Return `true` if the given value is a BlockArgument of a func::FuncOp.
bool isFunctionArgument(Value value);

/// Dialect-specific analysis state. Analysis/bufferization information
/// that is specific to ops from a certain dialect can be stored in derived
/// variants of this struct.
struct DialectAnalysisState {
  DialectAnalysisState() = default;

  virtual ~DialectAnalysisState() = default;

  // Copying state is forbidden. Always pass as reference.
  DialectAnalysisState(const DialectAnalysisState &) = delete;
};

/// AnalysisState provides a variety of helper functions for dealing with
/// tensor values.
class AnalysisState {
public:
  /// Determine which OpOperand* will alias with `result` if the op is
  /// bufferized in place. Return an empty vector if the op is not bufferizable.
  SmallVector<OpOperand *> getAliasingOpOperand(OpResult result) const;

  /// Determine which OpResult will alias with `opOperand` if the op is
  /// bufferized in place. Return an empty vector if the op is not bufferizable.
  SmallVector<OpResult> getAliasingOpResult(OpOperand &opOperand) const;

  /// Return true if `opOperand` bufferizes to a memory read. Return `true` if
  /// the op is not bufferizable.
  bool bufferizesToMemoryRead(OpOperand &opOperand) const;

  /// Return true if `opOperand` bufferizes to a memory write. Return true` if
  /// the op is not bufferizable.
  bool bufferizesToMemoryWrite(OpOperand &opOperand) const;

  /// Return true if `opOperand` does neither read nor write but bufferizes to
  /// an alias. Return false if the op is not bufferizable.
  bool bufferizesToAliasOnly(OpOperand &opOperand) const;

  /// Return true if a copy can always be avoided when allocating a new tensor
  /// for the given OpOperand.
  bool canOmitTensorCopy(OpOperand &opOperand) const;

  /// Return true if the given value is read by an op that bufferizes to a
  /// memory read. Also takes into account ops that create an alias but do not
  /// read by themselves (e.g., ExtractSliceOp).
  bool isValueRead(Value value) const;

  /// Starting from `value`, follow the use-def chain in reverse, always
  /// selecting the aliasing OpOperands. Find and return Values for which
  /// `condition` evaluates to true. OpOperands of such matching Values are not
  /// traversed any further.
  ///
  /// When reaching the end of a chain (BlockArgument or Value without aliasing
  /// OpOperands), also return the last Value of that chain.
  ///
  /// Example:
  ///
  ///                               8
  ///                               |
  ///   6*         7*         +-----+----+
  ///   |          |          |          |
  ///   2*         3          4*         5
  ///   |          |          |          |
  ///   +----------+----------+----------+
  ///              |
  ///              1
  ///
  /// In the above example, Values with a star satisfy the condition. When
  /// starting the traversal from Value 1, the resulting SetVector is:
  /// { 2, 7, 8, 5 }
  SetVector<Value> findValueInReverseUseDefChain(
      Value value, llvm::function_ref<bool(Value)> condition) const;

  /// Find the Values of the last preceding write of a given Value.
  ///
  /// Note: Unknown ops are handled conservatively and assumed to be writes.
  /// Furthermore, BlockArguments are also assumed to be writes. There is no
  /// analysis across block boundaries.
  ///
  /// Note: When reaching an end of the reverse SSA use-def chain, that value
  /// is returned regardless of whether it is a memory write or not.
  SetVector<Value> findLastPrecedingWrite(Value value) const;

  /// Return `true` if the given OpResult has been decided to bufferize inplace.
  virtual bool isInPlace(OpOperand &opOperand) const = 0;

  /// Return true if `v1` and `v2` bufferize to equivalent buffers.
  virtual bool areEquivalentBufferizedValues(Value v1, Value v2) const = 0;

  /// Return true if `v1` and `v2` may bufferize to aliasing buffers.
  virtual bool areAliasingBufferizedValues(Value v1, Value v2) const = 0;

  /// Return `true` if the given tensor has undefined contents.
  virtual bool hasUndefinedContents(OpOperand *opOperand) const = 0;

  /// Return true if the given tensor (or an aliasing tensor) is yielded from
  /// the containing block. Also include all aliasing tensors in the same block.
  ///
  /// Note: In the absence of an analysis, an implementation may return true for
  /// any given tensor.
  virtual bool isTensorYielded(Value tensor) const = 0;

  /// Return `true` if the given dialect state exists.
  bool hasDialectState(StringRef name) const {
    auto it = dialectState.find(name);
    return it != dialectState.end();
  }

  /// Return dialect-specific bufferization state.
  template <typename StateT>
  Optional<const StateT *> getDialectState(StringRef name) const {
    auto it = dialectState.find(name);
    if (it == dialectState.end())
      return None;
    return static_cast<const StateT *>(it->getSecond().get());
  }

  /// Return dialect-specific analysis state or create one if none exists.
  template <typename StateT>
  StateT &getOrCreateDialectState(StringRef name) {
    // Create state if it does not exist yet.
    if (!hasDialectState(name))
      dialectState[name] = std::make_unique<StateT>();
    return static_cast<StateT &>(*dialectState[name]);
  }

  void insertDialectState(StringRef name,
                          std::unique_ptr<DialectAnalysisState> state) {
    assert(!dialectState.count(name) && "dialect state already initialized");
    dialectState[name] = std::move(state);
  }

  /// Return a reference to the BufferizationOptions.
  const BufferizationOptions &getOptions() const { return options; }

protected:
  explicit AnalysisState(const BufferizationOptions &options);

  // AnalysisState should be passed as a reference.
  AnalysisState(const AnalysisState &) = delete;

  ~AnalysisState() = default;

private:
  /// Dialect-specific analysis state.
  DenseMap<StringRef, std::unique_ptr<DialectAnalysisState>> dialectState;

  /// A reference to current bufferization options.
  const BufferizationOptions &options;
};

/// BufferizationState provides helper functions for performing bufferization
/// rewrites and handling memref buffers.
struct BufferizationState {
  enum ForceInPlacability { FORCE_INPLACE, FORCE_OUT_OF_PLACE };

  BufferizationState(const AnalysisState &analysisState)
      : analysisState(analysisState) {}

  /// Creates a memref allocation for the given shaped value. `dealloc`
  /// indicates whether the buffer should be deallocated or not. When `dealloc`
  /// is `false`, this would create a memory leak, unless the buffer is
  /// deallocated through some other mechanism.
  ///
  /// `dealloc` is optional. By default, this function will figure out by itself
  /// if it is safe to deallocate the buffer. In essence, when returning the
  /// buffer from a block, it is not safe to deallocate the buffer. This
  /// information is queried via `AnalysisState::isTensorYielded`.
  ///
  /// Note: `shapedValue` is typically a tensor value. However, if it is a
  /// memref value, `dealloc` is no longer optional and must be specified.
  FailureOr<Value> createAlloc(OpBuilder &b, Location loc, Value shapedValue,
                               Optional<bool> dealloc = None);

  /// Return the buffer (memref) for a given OpOperand (tensor). Allocate
  /// a new buffer and copy over data from the existing buffer if out-of-place
  /// bufferization was decided.
  ///
  /// Whether a buffer is in-place or out-of-place is queried from the analysis
  /// state. Some analyses may always conservatively opt for out-of-place
  /// bufferization. Inplacability decisions can be overridden with the optional
  /// `overrideInPlace` parameter.
  FailureOr<Value>
  getBuffer(RewriterBase &rewriter, OpOperand &opOperand,
            Optional<ForceInPlacability> overrideInPlace = None,
            Optional<Operation *> customCopyInsertionPoint = None);

  /// Return the buffer type for a given Value (tensor) after bufferization.
  ///
  /// Note: Op implementations should preferrably call `getBuffer()->getType()`.
  /// This function should only be used if `getBuffer` cannot be used.
  BaseMemRefType getBufferType(Value value) const;

  /// Return a reference to the BufferizationOptions.
  const BufferizationOptions &getOptions() const {
    return analysisState.getOptions();
  }

  const AnalysisState &getAnalysisState() const { return analysisState; }

protected:
  // BufferizationState should be passed as a reference.
  BufferizationState(const BufferizationState &) = delete;

private:
  const AnalysisState &analysisState;
};

/// Replace an op with replacement values. The op is deleted. Tensor OpResults
/// must be replaced with memref values.
void replaceOpWithBufferizedValues(RewriterBase &rewriter, Operation *op,
                                   ValueRange values);

/// Lookup the buffer for the given value. If the value was not bufferized yet,
/// wrap it in a ToMemrefOp. Otherwise, it is the result of a ToTensorOp, from
/// which the memref operand is returned.
///
/// Note: Use `BufferizationState::getBuffer` during bufferization.
/// `lookupBuffer` is just for compatibility and gradual migration of
/// bufferization patterns to BufferizableOpInterface-based bufferization. It
/// does not insert any buffer copies.
Value lookupBuffer(RewriterBase &rewriter, Value tensor,
                   const BufferizationOptions &options);

/// Replace an op with a new op. The new op must have the same number of
/// results as the replaced op. The new op may not return any tensor values.
template <typename OpTy, typename... Args>
OpTy replaceOpWithNewBufferizedOp(RewriterBase &rewriter, Operation *op,
                                  Args &&...args) {
  auto newOp = rewriter.create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
  replaceOpWithBufferizedValues(rewriter, op, newOp->getResults());
  return newOp;
}

/// Return a MemRefType to which the `tensorType` can be bufferized.
///
/// If possible, op bufferization implementations should not use this function
/// and instead infer precise memref types for tensor results by themselves.
///
/// Unless a layout map was specified, `options.unknownTypeConverter` determines
/// what kind of layout map will be used. For best composability (without
/// copies), the fully dynamic layout map is used by default.
///
/// Note: Canonicalization patterns could clean up layout maps and infer more
/// precise layout maps after bufferization. However, many possible
/// canonicalizations are currently not implemented.
BaseMemRefType getMemRefType(TensorType tensorType,
                             const BufferizationOptions &options,
                             MemRefLayoutAttrInterface layout = {},
                             Attribute memorySpace = {});

/// Return a MemRef type with fully dynamic layout. If the given tensor type
/// is unranked, return an unranked MemRef type.
BaseMemRefType getMemRefTypeWithFullyDynamicLayout(TensorType tensorType,
                                                   Attribute memorySpace = {});

/// Return a MemRef type with a static identity layout (i.e., no layout map). If
/// the given tensor type is unranked, return an unranked MemRef type.
BaseMemRefType
getMemRefTypeWithStaticIdentityLayout(TensorType tensorType,
                                      Attribute memorySpace = {});

} // namespace bufferization
} // namespace mlir

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h.inc"

#endif // MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZABLEOPINTERFACE_H_
