//===- OneShotAnalysis.h - One-Shot (Single Pass) Analysis ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ONESHOTANALYSIS_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ONESHOTANALYSIS_H

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/EquivalenceClasses.h"

namespace mlir {
namespace bufferization {

class AnalysisBufferizationState;
class BufferizationAliasInfo;
struct AnalysisBufferizationOptions;

/// PostAnalysisStepFns can be registered with `BufferizationOptions` and are
/// executed after the analysis, but before bufferization. They can be used to
/// implement custom dialect-specific optimizations. They may modify the IR, but
/// must keep `aliasInfo` consistent. Newly created operations and operations
/// that should be re-analyzed must be added to `newOps`.
using PostAnalysisStepFn = std::function<LogicalResult(
    Operation *, BufferizationState &, BufferizationAliasInfo &,
    SmallVector<Operation *> &)>;

using PostAnalysisStepList = SmallVector<PostAnalysisStepFn>;

/// Options for analysis-enabled bufferization.
struct AnalysisBufferizationOptions : public BufferizationOptions {
  AnalysisBufferizationOptions() = default;

  /// Register a "post analysis" step. Such steps are executed after the
  /// analysis, but before bufferization.
  void addPostAnalysisStep(PostAnalysisStepFn fn) {
    postAnalysisSteps.push_back(fn);
  }

  /// Registered post analysis steps.
  PostAnalysisStepList postAnalysisSteps;
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
  void bufferizeInPlace(OpOperand &operand, BufferizationState &state);

  /// Set the inPlace bufferization spec to false.
  void bufferizeOutOfPlace(OpOperand &operand);

  /// Return true if `v1` and `v2` bufferize to equivalent buffers.
  bool areEquivalentBufferizedValues(Value v1, Value v2) const {
    return equivalentInfo.isEquivalent(v1, v2);
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

  /// Mark a value as in-place bufferized.
  void markInPlace(OpOperand &o) { inplaceBufferized.insert(&o); }

  /// Return `true` if a value was marked as in-place bufferized.
  bool isInPlace(OpOperand &opOperand) const;

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

  /// Set of all OpResults that were decided to bufferize in-place.
  llvm::DenseSet<OpOperand *> inplaceBufferized;

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

/// State for analysis-enabled bufferization. This class keeps track of alias
/// (via BufferizationAliasInfo) to decide if tensor OpOperands should bufferize
/// in-place.
class AnalysisBufferizationState : public BufferizationState {
public:
  AnalysisBufferizationState(Operation *op,
                             const AnalysisBufferizationOptions &options);

  AnalysisBufferizationState(const AnalysisBufferizationState &) = delete;

  virtual ~AnalysisBufferizationState() = default;

  /// Return a reference to the BufferizationAliasInfo.
  BufferizationAliasInfo &getAliasInfo() { return aliasInfo; }

  /// Return `true` if the given OpResult has been decided to bufferize inplace.
  bool isInPlace(OpOperand &opOperand) const override;

  /// Return true if `v1` and `v2` bufferize to equivalent buffers.
  bool areEquivalentBufferizedValues(Value v1, Value v2) const override;

private:
  /// `aliasInfo` keeps track of aliasing and equivalent values. Only internal
  /// functions and `runOneShotBufferize` may access this object.
  BufferizationAliasInfo aliasInfo;
};

/// Analyze `op` and its nested ops. Bufferization decisions are stored in
/// `state`.
LogicalResult analyzeOp(Operation *op, AnalysisBufferizationState &state);

/// Run One-Shot Bufferize on the given op: Analysis + Bufferization
LogicalResult
runOneShotBufferize(Operation *op,
                    std::unique_ptr<AnalysisBufferizationOptions> options);

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ONESHOTANALYSIS_H
