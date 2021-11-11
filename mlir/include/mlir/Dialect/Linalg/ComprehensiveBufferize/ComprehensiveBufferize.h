//===- ComprehensiveBufferize.h - Linalg bufferization pass -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_COMPREHENSIVE_BUFFERIZE_H
#define MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_COMPREHENSIVE_BUFFERIZE_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SetOperations.h"

namespace mlir {

class DominanceInfo;
class FuncOp;
class GlobalCreator;
class ModuleOp;

namespace linalg {
namespace comprehensive_bufferize {

// TODO: from some HW description.
static constexpr int64_t kBufferAlignments = 128;

/// The BufferizationAliasInfo class maintains a list of buffer aliases and
/// equivalence classes to support bufferization.
/// ExtractSliceOps have special behavior, they act as a level of indirection
/// for bufferization. They don't create reads or writes themselves and analysis
/// needs to look through their uses.
/// ExtractSliceOp + InsertSliceOp have special joint behavior: they may
/// bufferize to the same buffer (i.e. subview), which is what introduces the
/// need for bufferization classes.
/// Some of these functionalities could be refactored in a Bufferizer class that
/// uses BufferizationAliasInfo.
class BufferizationAliasInfo {
public:
  explicit BufferizationAliasInfo(Operation *rootOp);

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
    // Return `false` if we have no information about `v1` or `v2`.
    if (equivalentInfo.findValue(v1) == equivalentInfo.end() ||
        equivalentInfo.findValue(v2) == equivalentInfo.end())
      return false;

    return equivalentInfo.getLeaderValue(v1) ==
           equivalentInfo.getLeaderValue(v2);
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

  /// Print to `os`.
  void printAliases(raw_ostream &os) const;
  void printEquivalences(raw_ostream &os) const;

  /// Print to `errs()`.
  void dumpAliases() const;
  void dumpEquivalences() const;

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

/// Analyze the `ops` to determine which OpResults are inplaceable.
LogicalResult inPlaceAnalysis(SmallVector<Operation *> &ops,
                              BufferizationAliasInfo &aliasInfo,
                              const DominanceInfo &domInfo,
                              unsigned analysisFuzzerSeed = 0);

// TODO: Do not expose those functions in the header file.
/// Default allocation function that is used by the comprehensive bufferization
/// pass. The default currently creates a ranked memref using `memref.alloc`.
Optional<Value> defaultAllocationFn(OpBuilder &b, Location loc, MemRefType type,
                                    const SmallVector<Value> &dynShape);

/// Default deallocation function that is used by the comprehensive
/// bufferization pass. It expects to recieve back the value called from the
/// `defaultAllocationFn`.
void defaultDeallocationFn(OpBuilder &b, Location loc, Value allocatedBuffer);

/// Default memory copy function that is used by the comprehensive bufferization
/// pass. Creates a `linalg.copy` op.
void defaultMemCpyFn(OpBuilder &b, Location loc, Value from, Value to);

/// Callback functions that are used by the comprehensive bufferization pass to
/// allocate/deallocate memory. These default to use the
/// `defaultAllocationFn`/`defaultDeallocationFn`, but can be overridden by the
/// caller. The `deallocationFn` is gauranteed to recieve the `Value` returned
/// by the `allocationFn`.
struct AllocationCallbacks {
  using AllocationFn = std::function<Optional<Value>(
      OpBuilder &, Location, MemRefType, const SmallVector<Value> &)>;
  using DeallocationFn = std::function<void(OpBuilder &, Location, Value)>;
  using MemCpyFn = std::function<void(OpBuilder &, Location, Value, Value)>;

  AllocationCallbacks(AllocationFn allocFn, DeallocationFn deallocFn,
                      MemCpyFn copyFn)
      : allocationFn(allocFn), deallocationFn(deallocFn), memCpyFn(copyFn) {}

  AllocationCallbacks()
      : allocationFn(defaultAllocationFn),
        deallocationFn(defaultDeallocationFn), memCpyFn(defaultMemCpyFn) {}

  AllocationFn allocationFn;
  DeallocationFn deallocationFn;
  MemCpyFn memCpyFn;
};

/// Bufferize one particular op.
/// `bufferizedFunctionTypes` (resp. `globalCreator`) are expected to be
/// non-null if `op` is a CallOpInterface (resp. GlobalCreator).
LogicalResult
bufferizeOp(Operation *op, BlockAndValueMapping &bvm,
            BufferizationAliasInfo &aliasInfo,
            AllocationCallbacks allocationFns,
            DenseMap<FuncOp, FunctionType> *bufferizedFunctionTypes = nullptr);

/// Register external models implemented for the `BufferizableOpInterface`.
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

/// Try to eliminate InitTensorOps inside `funcOp`.
///
/// * `rewriteFunc` generates the replacement for the InitTensorOp.
/// * Only InitTensorOps that are anchored on a matching OpOperand as per
///   `anchorMatchFunc` are considered. "Anchored" means that there is a path on
///   the reverse SSA use-def chain, starting from the OpOperand and always
///   following the aliasing  OpOperand, that eventually ends at a single
///   InitTensorOp.
/// * The result of `rewriteFunc` must usually be analyzed for inplacability.
///   This analysis can be skipped with `skipAnalysis`.
LogicalResult initTensorElimination(
    FuncOp funcOp, BufferizationAliasInfo &aliasInfo, DominanceInfo &domInfo,
    std::function<bool(OpOperand &)> anchorMatchFunc,
    std::function<Value(OpBuilder &, Location, OpOperand &)> rewriteFunc,
    bool skipAnalysis = false);

/// Try to eliminate InitTensorOps inside funcOp that are anchored on an
/// InsertSliceOp, i.e., if it is eventually inserted into another tensor
/// (and some other conditions are met).
LogicalResult eliminateInsertSliceAnchoredInitTensorOps(
    FuncOp funcOp, BufferizationAliasInfo &aliasInfo, DominanceInfo &domInfo);

struct BufferizationOptions {
  BufferizationOptions()
      : allocationFns(std::make_unique<AllocationCallbacks>()) {}

  std::unique_ptr<AllocationCallbacks> allocationFns;
  bool allowReturnMemref = false;
  unsigned analysisFuzzerSeed = 0;
  bool testAnalysisOnly = false;
};

LogicalResult runComprehensiveBufferize(ModuleOp moduleOp,
                                        const BufferizationOptions &options);

} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_COMPREHENSIVE_BUFFERIZE_H
