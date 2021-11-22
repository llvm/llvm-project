//===- BufferizableOpInterface.h - Comprehensive Bufferize ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_BUFFERIZABLEOPINTERFACE_H_
#define MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_BUFFERIZABLEOPINTERFACE_H_

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
class BlockAndValueMapping;
class DominanceInfo;
class FuncOp;

namespace linalg {
namespace comprehensive_bufferize {

class BufferizationAliasInfo;

/// Specify fine-grain relationship between buffers to enable more analysis.
enum class BufferRelation {
  None,
  // TODO: ResultContainsOperand,
  // TODO: OperandContainsResult,
  Equivalent
};

/// The BufferizationAliasInfo class maintains a list of buffer aliases and
/// equivalence classes to support bufferization.
class BufferizationAliasInfo {
public:
  explicit BufferizationAliasInfo(Operation *rootOp);

  // BufferizationAliasInfo should be passed as a reference.
  BufferizationAliasInfo(const BufferizationAliasInfo &) = delete;

  /// Add a new entry for `v` in the `aliasInfo` and `equivalentInfo`. In the
  /// beginning the alias and equivalence sets only contain `v` itself.
  void createAliasInfoEntry(Value v);

  /// Insert an info entry for `newValue` and merge its alias set with that of
  /// `alias`.
  void insertNewBufferAlias(Value newValue, Value alias);

  /// Insert an info entry for `newValue` and merge its alias set with that of
  /// `alias`. Additionally, merge their equivalence classes.
  void insertNewBufferEquivalence(Value newValue, Value alias);

  /// Set the inPlace bufferization spec to true.
  /// Merge result's and operand's aliasing sets and iterate to a fixed point.
  void bufferizeInPlace(OpResult result, OpOperand &operand);

  /// Set the inPlace bufferization spec to false.
  void bufferizeOutOfPlace(OpResult result);

  /// Return true if `v1` and `v2` bufferize to equivalent buffers.
  bool areEquivalentBufferizedValues(Value v1, Value v2) const {
    return equivalentInfo.isEquivalent(v1, v2);
  }

  /// Return true if `v1` and `v2` bufferize to aliasing buffers.
  bool areAliasingBufferizedValues(Value v1, Value v2) const {
    return aliasInfo.isEquivalent(v1, v2);
  }

  /// Union the alias sets of `v1` and `v2`.
  void unionAliasSets(Value v1, Value v2) { aliasInfo.unionSets(v1, v2); }

  /// Union the equivalence classes of `v1` and `v2`.
  void unionEquivalenceClasses(Value v1, Value v2) {
    equivalentInfo.unionSets(v1, v2);
  }

  /// Apply `fun` to all the members of the equivalence class of `v`.
  void applyOnEquivalenceClass(Value v, function_ref<void(Value)> fun) const;

  /// Apply `fun` to all aliases of `v`.
  void applyOnAliases(Value v, function_ref<void(Value)> fun) const;

  // TODO: Move these out of BufferizationAliasInfo.
  /// Return true if the value is known to bufferize to writable memory.
  bool bufferizesToWritableMemory(Value v) const;

  /// Specify that the value is known to bufferize to writable memory.
  void setBufferizesToWritableMemory(Value v);

  /// Mark a value as in-place bufferized.
  void markInPlace(OpResult v) { inplaceBufferized.insert(v); }

  /// Return `true` if a value was marked as in-place bufferized.
  bool isInPlace(OpResult opResult) const;

private:
  /// llvm::EquivalenceClasses wants comparable elements. This comparator uses
  /// uses pointer comparison on the defining op. This is a poor man's
  /// comparison but it's not like UnionFind needs ordering anyway.
  struct ValueComparator {
    bool operator()(const Value &lhs, const Value &rhs) const {
      return lhs.getImpl() < rhs.getImpl();
    }
  };

  using EquivalenceClassRangeType = llvm::iterator_range<
      llvm::EquivalenceClasses<Value, ValueComparator>::member_iterator>;
  /// Check that aliasInfo for `v` exists and return a reference to it.
  EquivalenceClassRangeType getAliases(Value v) const;

  /// Set of tensors that are known to bufferize to writable memory.
  llvm::DenseSet<Value> bufferizeToWritableMemory;

  /// Set of all OpResults that were decided to bufferize in-place.
  llvm::DenseSet<OpResult> inplaceBufferized;

  /// Auxiliary structure to store all the values a given value may alias with.
  /// Alias information is "may be" conservative: In the presence of branches, a
  /// value may alias with one of multiple other values. The concrete aliasing
  /// value may not even be known at compile time. All such values are
  /// considered to be aliases.
  llvm::EquivalenceClasses<Value, ValueComparator> aliasInfo;

  /// Auxiliary structure to store all the equivalent buffer classes. Equivalent
  /// buffer information is "must be" conservative: Only if two values are
  /// guaranteed to be equivalent at runtime, they said to be equivalent. It is
  /// possible that, in the presence of branches, it cannot be determined
  /// statically if two values are equivalent. In that case, the values are
  /// considered to be not equivalent.
  llvm::EquivalenceClasses<Value, ValueComparator> equivalentInfo;
};

/// Determine which OpOperand* will alias with `result` if the op is bufferized
/// in place. Return an empty vector if the op is not bufferizable.
SmallVector<OpOperand *> getAliasingOpOperand(OpResult result);

/// Determine which OpResult will alias with `opOperand` if the op is bufferized
/// in place. Return an empty OpResult if the op is not bufferizable.
OpResult getAliasingOpResult(OpOperand &opOperand);

/// Return true if `opOperand` bufferizes to a memory read. Return `true` if the
/// op is not bufferizable.
bool bufferizesToMemoryRead(OpOperand &opOperand);

/// Return true if `opOperand` bufferizes to a memory write. Return
/// `true` if the op is not bufferizable.
bool bufferizesToMemoryWrite(OpOperand &opOperand);

/// Return true if `opOperand` does neither read nor write but bufferizes to an
/// alias. Return false if the op is not bufferizable.
bool bufferizesToAliasOnly(OpOperand &opOperand);

/// Return true if the given value is read by an op that bufferizes to a memory
/// read. Also takes into account ops that create an alias but do not read by
/// themselves (e.g., ExtractSliceOp).
bool isValueRead(Value value);

/// Return the relationship between the operand and the its corresponding
/// OpResult that it may alias with. Return None if the op is not bufferizable.
BufferRelation bufferRelation(OpOperand &opOperand);

/// Starting from `value`, follow the use-def chain in reverse, always selecting
/// the aliasing OpOperands. Find and return Values for which `condition`
/// evaluates to true. OpOperands of such matching Values are not traversed any
/// further.
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
llvm::SetVector<Value>
findValueInReverseUseDefChain(Value value,
                              std::function<bool(Value)> condition);

/// Find the Value of the last preceding write of a given Value.
///
/// Note: Unknown ops are handled conservatively and assumed to be writes.
/// Furthermore, BlockArguments are also assumed to be writes. There is no
/// analysis across block boundaries.
///
/// Note: When reaching an end of the reverse SSA use-def chain, that value
/// is returned regardless of whether it is a memory write or not.
Value findLastPrecedingWrite(Value value);

struct BufferizationState;

/// Callback functions that are used to allocate/deallocate/copy memory buffers.
/// Comprehensive Bufferize provides default implementations of these functions.
// TODO: Could be replaced with a "bufferization strategy" object with virtual
// functions in the future.
struct AllocationCallbacks {
  using AllocationFn = std::function<Optional<Value>(
      OpBuilder &, Location, MemRefType, const SmallVector<Value> &)>;
  using DeallocationFn = std::function<void(OpBuilder &, Location, Value)>;
  using MemCpyFn = std::function<void(OpBuilder &, Location, Value, Value)>;
  using CreateAllocDeallocFn =
      std::function<Value(OpBuilder &, Location, Value, BufferizationState &)>;

  AllocationCallbacks(AllocationFn allocFn, DeallocationFn deallocFn,
                      MemCpyFn copyFn, CreateAllocDeallocFn allocDeallocFn)
      : allocationFn(allocFn), deallocationFn(deallocFn), memCpyFn(copyFn),
        createAllocDeallocFn(allocDeallocFn) {}

  /// A function that allocates memory.
  AllocationFn allocationFn;

  /// A function that deallocated memory. Must be allocated by `allocationFn`.
  DeallocationFn deallocationFn;

  /// A function that copies memory between two allocations.
  MemCpyFn memCpyFn;

  /// A function that creates an alloc-dealloc pair. This function may perform
  /// additional optimizations such as buffer allocation hoisting. This function
  /// calls `allocationFn` and `deallocationFn` to create (de)allocations.
  CreateAllocDeallocFn createAllocDeallocFn;
};

/// BufferizationState keeps track of bufferization state and provides access to
/// the results of the analysis.
struct BufferizationState {
  BufferizationState(BufferizationAliasInfo &aliasInfo,
                     AllocationCallbacks &allocationFns)
      : aliasInfo(aliasInfo), allocationFns(allocationFns) {}

  // BufferizationState should be passed as a reference.
  BufferizationState(const BufferizationState &) = delete;

  /// Map tensor values to memref buffers.
  void mapBuffer(ValueRange tensors, ValueRange buffers);

  /// Map a value to another value.
  void mapValue(Value from, Value to);

  /// Map a tensor value to a memref buffer.
  void mapBuffer(Value tensor, Value buffer);

  /// Lookup the memref buffer that is associated to the given tensor value.
  /// Asserts if no buffer is associated.
  Value lookupBuffer(Value tensor) const;

  /// Lookup the value that is associated to the given value. Asserts if no
  /// value is associated.
  Value lookupValue(Value value) const;

  /// Return `true` if the given value is mapped.
  bool isMapped(Value value) const;

  /// Mark `op` as obsolete, so that it is deleted after bufferization.
  void markOpObsolete(Operation *op);

  /// `aliasInfo` keeps track of aliasing and equivalent values.
  BufferizationAliasInfo &aliasInfo;

  /// `allocationFns` contains helper functions for creating alloc ops, dealloc
  /// ops and memcpy ops.
  AllocationCallbacks &allocationFns;

  /// The mapping of tensors to buffers. May also contain mappings of non-tensor
  /// values.
  BlockAndValueMapping mapping;

  /// Obsolete ops that should be deleted after bufferization.
  SmallVector<Operation *> obsoleteOps;
};

/// Return the result buffer (memref) for a given OpResult (tensor). Allocate
/// a new buffer and copy over data from the existing buffer if out-of-place
/// bufferization is necessary.
Value getResultBuffer(OpBuilder &b, OpResult result, BufferizationState &state);

/// PostAnalysisSteps can be registered with `BufferizationOptions` and are
/// executed after the analysis, but before bufferization. They can be used
/// implement custom dialect-specific optimizations.
struct PostAnalysisStep {
  virtual ~PostAnalysisStep() {}

  /// Run the post analysis step. This function may modify the IR, but must keep
  /// `aliasInfo` consistent. Newly created operations and operations that
  /// should be re-analyzed must be stored in `newOps`.
  virtual LogicalResult run(FuncOp funcOp, BufferizationAliasInfo &aliasInfo,
                            DominanceInfo &domInfo,
                            SmallVector<Operation *> &newOps) = 0;
};

} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h.inc"

namespace mlir {
namespace linalg {
namespace comprehensive_bufferize {

/// AllocationHoistingBarrierOnly is an external implementation of
/// BufferizableOpInterface for ops that are (not yet) bufferizable, but are
/// known to be allocation hoisting barriers. All interface methods (except for
/// `isAllocationHoistingBarrier`) are implemented conservatively.
template <typename OpTy>
struct AllocationHoistingBarrierOnly
    : public BufferizableOpInterface::ExternalModel<
          AllocationHoistingBarrierOnly<OpTy>, OpTy> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    return false;
  }

  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    return {};
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    return OpResult();
  }

  BufferRelation bufferRelation(Operation *op, OpOperand &opOperand) const {
    return BufferRelation::None;
  }

  bool isWritable(Operation *op, Value value) const { return false; }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto isaTensor = [](Type t) { return t.isa<TensorType>(); };
    if (any_of(op->getOperandTypes(), isaTensor) ||
        any_of(op->getResultTypes(), isaTensor))
      return op->emitError() << "unsupported op with tensors";
    return success();
  }

  bool isAllocationHoistingBarrier(Operation *op) const { return true; }
};

} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_BUFFERIZABLEOPINTERFACE_H_
