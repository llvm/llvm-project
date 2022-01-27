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
class FuncOp;

namespace bufferization {

// TODO: from some HW description.
static constexpr int64_t kBufferAlignments = 128;

class BufferizableOpInterface;
struct BufferizationOptions;
class BufferizationState;

/// Options for ComprehensiveBufferize.
struct BufferizationOptions {
  using AllocationFn = std::function<FailureOr<Value>(OpBuilder &, Location,
                                                      MemRefType, ValueRange)>;
  using DeallocationFn =
      std::function<LogicalResult(OpBuilder &, Location, Value)>;
  using MemCpyFn =
      std::function<LogicalResult(OpBuilder &, Location, Value, Value)>;

  BufferizationOptions();

  // BufferizationOptions cannot be copied.
  BufferizationOptions(const BufferizationOptions &other) = delete;

  /// Return `true` if the op is allowed to be bufferized.
  bool isOpAllowed(Operation *op) const {
    if (!dialectFilter.hasValue())
      return true;
    return dialectFilter->contains(op->getDialect()->getNamespace());
  }

  /// Allow-list the given dialects in the dialect filter. Only ops from
  /// allow-listed dialects will be bufferized. If no dialect is added, ops from
  /// any dialect will be bufferized.
  template <typename... DialectTs>
  void addToDialectFilter() {
    // The following expands a call to addToDialectFilterImpl for each dialect
    // in 'DialectTs'. This magic is necessary due to a limitation in the places
    // that a parameter pack can be expanded in c++11.
    // FIXME: In c++17 this can be simplified by using 'fold expressions'.
    (void)std::initializer_list<int>{
        0, (addToDialectFilterImpl<DialectTs>(), 0)...};
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

  /// Specifies whether returning newly allocated memrefs should be allowed.
  /// Otherwise, a pass failure is triggered.
  bool allowReturnMemref = false;

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

  /// Only bufferize ops from dialects that are allowed-listed by the filter.
  /// All other ops are ignored. This option controls the scope of partial
  /// bufferization.
  ///
  /// Note: If no filter is specified, all ops are bufferized (as long as they
  /// implement BufferizableOpInterface). If a filter is specified,
  /// `allowUnknownOps` should be enabled. Otherwise, bufferization would fail
  /// when encountering an op that is forbidden by the filter.
  Optional<DenseSet<StringRef>> dialectFilter;

private:
  /// Allow-list a dialect in the dialect filter.
  template <typename DialectT>
  void addToDialectFilterImpl() {
    if (!dialectFilter.hasValue())
      dialectFilter.emplace();
    dialectFilter->insert(DialectT::getDialectNamespace());
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

/// Dialect-specific bufferization state. Analysis/bufferization information
/// that is specific to ops from a certain dialect can be stored in derived
/// variants of this struct.
struct DialectBufferizationState {
  DialectBufferizationState() = default;

  virtual ~DialectBufferizationState() = default;

  // Copying state is forbidden. Always pass as reference.
  DialectBufferizationState(const DialectBufferizationState &) = delete;
};

/// BufferizationState provides a variety of helper functions for dealing with
/// tensor values and memref buffers.
class BufferizationState {
public:
  /// Determine which OpOperand* will alias with `result` if the op is
  /// bufferized in place. Return an empty vector if the op is not bufferizable.
  SmallVector<OpOperand *> getAliasingOpOperand(OpResult result) const;

  /// Determine which OpResult will alias with `opOperand` if the op is
  /// bufferized in place. Return an empty OpResult if the op is not
  /// bufferizable.
  OpResult getAliasingOpResult(OpOperand &opOperand) const;

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

  /// Return the buffer (memref) for a given OpOperand (tensor). Allocate
  /// a new buffer and copy over data from the existing buffer if out-of-place
  /// bufferization was decided.
  FailureOr<Value>
  getBuffer(RewriterBase &rewriter, OpOperand &opOperand,
            bool forceInPlace = false,
            Optional<Operation *> customCopyInsertionPoint = None) const;

  /// Return dialect-specific bufferization state.
  template <typename StateT>
  Optional<const StateT *> getDialectState(StringRef name) const {
    auto it = dialectState.find(name);
    if (it == dialectState.end())
      return None;
    return static_cast<const StateT *>(it->getSecond().get());
  }

  /// Return dialect-specific bufferization state or create one if none exists.
  template <typename StateT>
  StateT &getOrCreateDialectState(StringRef name) {
    // Create state if it does not exist yet.
    if (!dialectState.count(name))
      dialectState[name] = std::make_unique<StateT>();
    return static_cast<StateT &>(*dialectState[name]);
  }

  /// Return a reference to the BufferizationOptions.
  const BufferizationOptions &getOptions() const { return options; }

protected:
  explicit BufferizationState(const BufferizationOptions &options);

  // BufferizationState should be passed as a reference.
  BufferizationState(const BufferizationState &) = delete;

  ~BufferizationState() = default;

private:
  /// Dialect-specific bufferization state.
  DenseMap<StringRef, std::unique_ptr<DialectBufferizationState>> dialectState;

  /// A reference to current bufferization options.
  const BufferizationOptions &options;
};

/// This a "no analysis, always copy" BufferizationState. In the absence of an
/// analysis, a buffer must be copied each time it is written to. Therefore, all
/// OpOperands that bufferize to a memory write must bufferize out-of-place.
class AlwaysCopyBufferizationState : public BufferizationState {
public:
  explicit AlwaysCopyBufferizationState(const BufferizationOptions &options);

  AlwaysCopyBufferizationState(const AlwaysCopyBufferizationState &) = delete;

  virtual ~AlwaysCopyBufferizationState() = default;

  /// Return `true` if the given OpResult has been decided to bufferize inplace.
  bool isInPlace(OpOperand &opOperand) const override;

  /// Return true if `v1` and `v2` bufferize to equivalent buffers.
  bool areEquivalentBufferizedValues(Value v1, Value v2) const override;
};

/// Replace an op with replacement values. The op is deleted. Tensor OpResults
/// must be replaced with memref values.
void replaceOpWithBufferizedValues(RewriterBase &rewriter, Operation *op,
                                   ValueRange values);

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

/// Creates a memref allocation with the given type and dynamic extents.
FailureOr<Value> createAlloc(OpBuilder &b, Location loc, MemRefType type,
                             ValueRange dynShape,
                             const BufferizationOptions &options);

/// Creates a memref allocation with the given type and dynamic extents. If
/// `createDealloc`, a deallocation op is inserted at the point where the
/// allocation goes out of scope.
FailureOr<Value> createAlloc(OpBuilder &b, Location loc, MemRefType type,
                             ValueRange dynShape, bool deallocMemref,
                             const BufferizationOptions &options);

/// Creates a memref allocation for the given shaped value. This function may
/// perform additional optimizations such as buffer allocation hoisting. If
/// `createDealloc`, a deallocation op is inserted at the point where the
/// allocation goes out of scope.
// TODO: Allocation hoisting should be a cleanup pass.
FailureOr<Value> createAlloc(OpBuilder &b, Location loc, Value shapedValue,
                             bool deallocMemref,
                             const BufferizationOptions &options);

/// Creates a memref deallocation. The given memref buffer must have been
/// allocated using `createAlloc`.
LogicalResult createDealloc(OpBuilder &b, Location loc, Value allocatedBuffer,
                            const BufferizationOptions &options);

/// Creates a memcpy between two given buffers.
LogicalResult createMemCpy(OpBuilder &b, Location loc, Value from, Value to,
                           const BufferizationOptions &options);

} // namespace bufferization
} // namespace mlir

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h.inc"

namespace mlir {
namespace bufferization {

/// AllocationHoistingBarrierOnly is an external implementation of
/// BufferizableOpInterface for ops that are (not yet) bufferizable, but are
/// known to be allocation hoisting barriers. All interface methods (except for
/// `isAllocationHoistingBarrier`) are implemented conservatively.
template <typename OpTy>
struct AllocationHoistingBarrierOnly
    : public BufferizableOpInterface::ExternalModel<
          AllocationHoistingBarrierOnly<OpTy>, OpTy> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return true;
  }

  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const BufferizationState &state) const {
    return {};
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return OpResult();
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationState &state) const {
    return BufferRelation::None;
  }

  bool isWritable(Operation *op, Value value,
                  const BufferizationState &state) const {
    return false;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    return failure();
  }

  bool isAllocationHoistingBarrier(Operation *op) const { return true; }
};

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZABLEOPINTERFACE_H_
