//===- BufferizableOpInterface.h - Bufferizable Ops -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZABLEOPINTERFACE_H_
#define MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZABLEOPINTERFACE_H_

#include <utility>

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
class BlockAndValueMapping;
class DominanceInfo;

namespace bufferization {

class AnalysisState;
class BufferizableOpInterface;
struct DialectAnalysisState;

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

  /// An op filter entry. Filters can be used to specify which ops should be
  /// processed by the bufferization.
  struct OpFilterEntry {
    /// If the filter function evaluates to `true`, the filter matches.
    using FilterFn = std::function<bool(Operation *)>;

    /// Filter type: A filter can either be a DENY filter or an ALLOW filter.
    enum FilterType : int8_t { DENY = 0, ALLOW = 1 };

    FilterFn fn;
    FilterType type;
  };

  BufferizationOptions();

  /// Return whether the op should be bufferized or not.
  ///
  /// If no filter is specified (`hasFilter` = false), every op will be
  /// bufferized. Otherwise, an op is bufferized if:
  ///
  /// - At least one ALLOW filter says `true`.
  /// - And, no DENY filter says `true`.
  bool isOpAllowed(Operation *op) const {
    if (!hasFilter)
      return true;
    bool isAllowed = false;
    for (const OpFilterEntry &entry : opFilter) {
      bool filterResult = entry.fn(op);
      switch (entry.type) {
      case OpFilterEntry::ALLOW:
        isAllowed |= filterResult;
        break;
      case OpFilterEntry::DENY:
        if (filterResult)
          // DENY filter matches. This op is no allowed. (Even if other ALLOW
          // filters may match.)
          return false;
      };
    }
    return isAllowed;
  }

  /// Allow the given dialects and activate the filter (`hasFilter`).
  ///
  /// This function adds one or multiple ALLOW filters.
  template <typename... DialectTs>
  void allowDialectInFilter() {
    // The following expands a call to allowDialectInFilterImpl for each dialect
    // in 'DialectTs'. This magic is necessary due to a limitation in the places
    // that a parameter pack can be expanded in c++11.
    // FIXME: In c++17 this can be simplified by using 'fold expressions'.
    (void)std::initializer_list<int>{
        0, (allowDialectInFilterImpl<DialectTs>(), 0)...};
  }

  /// Allow the given dialect and activate the filter (`hasFilter`).
  ///
  /// This function adds an ALLOW filter.
  void allowDialectInFilter(StringRef dialectNamespace) {
    hasFilter = true;
    OpFilterEntry::FilterFn filterFn = [=](Operation *op) {
      return op->getDialect()->getNamespace() == dialectNamespace;
    };
    opFilter.push_back(
        OpFilterEntry{filterFn, OpFilterEntry::FilterType::ALLOW});
  }

  /// Allow the given ops and activate the filter (`hasFilter`).
  ///
  /// This function adds one or multiple ALLOW filters.
  template <typename... OpTys>
  void allowOperationInFilter() {
    // FIXME: In c++17 this can be simplified by using 'fold expressions'.
    (void)std::initializer_list<int>{
        0, (allowOperationInFilterImpl<OpTys>(), 0)...};
  }

  /// Allow the given op and activate the filter (`hasFilter`).
  ///
  /// This function adds an ALLOW filter.
  void allowOperationInFilter(StringRef opName) {
    hasFilter = true;
    OpFilterEntry::FilterFn filterFn = [=](Operation *op) {
      return op->getName().getStringRef() == opName;
    };
    allowOperationInFilter(filterFn);
  }

  /// Deny the given op and activate the filter (`hasFilter`).
  ///
  /// This function adds a DENY filter.
  void denyOperationInFilter(StringRef opName) {
    hasFilter = true;
    OpFilterEntry::FilterFn filterFn = [=](Operation *op) {
      return op->getName().getStringRef() == opName;
    };
    denyOperationInFilter(filterFn);
  }

  /// Allow ops that are matched by `fn` and activate the filter (`hasFilter`).
  ///
  /// This function adds an ALLOW filter.
  void allowOperationInFilter(OpFilterEntry::FilterFn fn) {
    hasFilter = true;
    opFilter.push_back(OpFilterEntry{fn, OpFilterEntry::FilterType::ALLOW});
  }

  /// Deny ops that are matched by `fn` and activate the filter (`hasFilter`).
  ///
  /// This function adds a DENY filter.
  void denyOperationInFilter(OpFilterEntry::FilterFn fn) {
    hasFilter = true;
    opFilter.push_back(OpFilterEntry{fn, OpFilterEntry::FilterType::DENY});
  }

  /// Try to cast the given op to BufferizableOpInterface if the op is allow
  /// listed.
  BufferizableOpInterface dynCastBufferizableOp(Operation *op) const;

  /// Try to cast the given value to BufferizableOpInterface if the op is allow
  /// listed.
  BufferizableOpInterface dynCastBufferizableOp(Value value) const;

  /// Helper functions for allocation, deallocation, memory copying.
  Optional<AllocationFn> allocationFn;
  Optional<DeallocationFn> deallocationFn;
  Optional<MemCpyFn> memCpyFn;

  /// Specifies whether not bufferizable ops are allowed in the input. If so,
  /// bufferization.to_memref and bufferization.to_tensor ops are inserted at
  /// the boundaries.
  bool allowUnknownOps = false;

  /// Specifies whether dealloc ops should be generated along with alloc ops. If
  /// not, new memory allocations will leak.
  bool createDeallocs = true;

  /// Seed for the analysis fuzzer. If set to `0`, the fuzzer is deactivated.
  /// Should be used only with `testAnalysisOnly = true`.
  unsigned analysisFuzzerSeed = 0;

  /// Specifies whether fully dynamic layout maps should be used on ranked
  /// MemRef types. If false, MemRef types will have no layout maps.
  bool fullyDynamicLayoutMaps = true;

  /// If set to `true`, does not modify the IR apart from adding attributes (for
  /// checking the results of the analysis) and post analysis steps.
  bool testAnalysisOnly = false;

  /// If set to `true`, the IR is annotated with details about RaW conflicts.
  /// For debugging only. Should be used together with `testAnalysisOnly`.
  bool printConflicts = false;

  /// If set to `true`, an `getAliasingOpResult` will return the corresponding
  /// "out"/"dest" OpOperand for every op that has the notion of an "out"/"dest"
  /// operand. I.e., the aliasing OpOperand of the i-th tensor OpResult is
  /// usually the i-th "out" tensor OpOperand. This is in line with
  /// destination-passing style and the default behavior. Op interface
  /// implementations must follow this contract to avoid surprising behavior.
  ///
  /// If set to `false`, BufferizableOpInterface implementations can try to be
  /// smart and choose to alias with "in" operands or other operands. E.g., the
  /// result of a `linalg.generic` op could bufferize in-place with an "in"
  /// OpOperand if the corresponding "out" operand is not used within the
  /// computation. Whether this pays off or not can be very input IR-specific.
  bool alwaysAliasingWithDest = true;

  /// If set to `true`, try to hoist allocations out of blocks as much as
  /// possible. An allocation is not hoisted across allocation hoisting barriers
  /// as indicated by `BufferizableOpInterface::isAllocationHoistingBarrier`.
  ///
  /// Examples of allocation hoisting barriers are parallel loops or ops where
  /// SSA values cannot be captured from the outside.
  bool hoistAllocations = true;

  /// Buffer alignment for new memory allocations.
  unsigned int bufferAlignment = 128;

  /// If set to `false`, all ops are bufferized (as long as they implement
  /// BufferizableOpInterface). Otherwise, only filtered ops are bufferized.
  bool hasFilter = false;

  /// A list of op filters that determine whether an op should be processed or
  /// ignored by the bufferization. If `hasFilter`, only ops that are not
  /// DENY-filtered and have at least one matching ALLOW filter are processed.
  SmallVector<OpFilterEntry> opFilter;

  /// Initializer functions for analysis state. These can be used to
  /// initialize dialect-specific analysis state.
  SmallVector<AnalysisStateInitFn> stateInitializers;

  /// Add a analysis state initializer that initializes the specified
  /// dialect-specific analysis state.
  void addDialectStateInitializer(StringRef name, const DialectStateInitFn &fn);

private:
  /// Allow a dialect.
  template <typename DialectT>
  void allowDialectInFilterImpl() {
    allowDialectInFilter(DialectT::getDialectNamespace());
  }

  /// Allow an op.
  template <typename OpTy>
  void allowOperationInFilterImpl() {
    allowOperationInFilter(OpTy::getOperationName());
  }
};

/// Specify fine-grain relationship between buffers to enable more analysis.
enum class BufferRelation {
  None,
  // TODO: ResultContainsOperand,
  // TODO: OperandContainsResult,
  Equivalent
};

/// Return `true` if the given value is a BlockArgument of a FuncOp.
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

  /// Return true if the given tensor (or an aliasing tensor) is yielded from
  /// the containing block. Also include all aliasing tensors in the same block.
  ///
  /// Note: In the absence of an analysis, an implementation may return true for
  /// any given tensor.
  virtual bool isTensorYielded(Value tensor) const = 0;

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
    if (!dialectState.count(name))
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

/// This a "no analysis, always copy" AnalysisState. In the absence of an
/// analysis, a buffer must be copied each time it is written to. Therefore, all
/// OpOperands that bufferize to a memory write must bufferize out-of-place.
class AlwaysCopyAnalysisState : public AnalysisState {
public:
  explicit AlwaysCopyAnalysisState(const BufferizationOptions &options);

  AlwaysCopyAnalysisState(const AlwaysCopyAnalysisState &) = delete;

  virtual ~AlwaysCopyAnalysisState() = default;

  /// Return `true` if the given OpResult has been decided to bufferize inplace.
  bool isInPlace(OpOperand &opOperand) const override;

  /// Return true if `v1` and `v2` bufferize to equivalent buffers.
  bool areEquivalentBufferizedValues(Value v1, Value v2) const override;

  /// Return true if the given tensor (or an aliasing tensor) is yielded from
  /// the containing block. Also include all aliasing tensors in the same block.
  bool isTensorYielded(Value tensor) const override;
};

/// BufferizationState provides helper functions for performing bufferization
/// rewrites and handling memref buffers.
struct BufferizationState {
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
  FailureOr<Value>
  getBuffer(RewriterBase &rewriter, OpOperand &opOperand,
            bool forceInPlace = false,
            Optional<Operation *> customCopyInsertionPoint = None);

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

/// Return a contiguous MemRefType (i.e. with canonical/empty layout map)
/// with the same shape as `shapedType` and specified `addressSpace`.
MemRefType getContiguousMemRefType(ShapedType shapedType,
                                   Attribute memorySpace = {});

/// Return a MemRefType to which the `tensorType` can be bufferized in a
/// composable fashion. The layout must be the most dynamic possible and
/// canonicalize away once bufferization is finished.
BaseMemRefType getMemRefType(TensorType tensorType,
                             const BufferizationOptions &options,
                             MemRefLayoutAttrInterface layout = {},
                             Attribute memorySpace = {});

/// Creates a memref deallocation. The given memref buffer must have been
/// allocated using `createAlloc`.
LogicalResult createDealloc(OpBuilder &b, Location loc, Value allocatedBuffer,
                            const BufferizationOptions &options);

/// Creates a memcpy between two given buffers.
LogicalResult createMemCpy(OpBuilder &b, Location loc, Value from, Value to,
                           const BufferizationOptions &options);

/// Try to hoist all new buffer allocations until the next hoisting barrier.
LogicalResult hoistBufferAllocations(Operation *op,
                                     const BufferizationOptions &options);

/// Create alloc/dealloc ops as specified in the bufferization options. If
/// `onlyLeakingAlloc`, only those buffer allocations are processed for which no
/// buffer deallocation can be created. `changed` is set to `true` if the IR was
/// modified.
LogicalResult createAllocDeallocOps(Operation *op,
                                    const BufferizationOptions &options,
                                    bool onlyLeakingAllocs = false,
                                    bool *changed = nullptr);

} // namespace bufferization
} // namespace mlir

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h.inc"

#endif // MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZABLEOPINTERFACE_H_
