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
#include "llvm/ADT/EquivalenceClasses.h"

namespace mlir {
namespace bufferization {

struct OneShotBufferizationOptions;
class BufferizationAliasInfo;
class OneShotAnalysisState;

/// Options for analysis-enabled bufferization.
struct OneShotBufferizationOptions : public BufferizationOptions {
  enum class AnalysisHeuristic { BottomUp, TopDown };

  OneShotBufferizationOptions() = default;

  /// Specifies whether returning newly allocated memrefs should be allowed.
  /// Otherwise, a pass failure is triggered.
  bool allowReturnAllocs = false;

  /// The heuristic controls the order in which ops are traversed during the
  /// analysis.
  AnalysisHeuristic analysisHeuristic = AnalysisHeuristic::BottomUp;
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
  void bufferizeInPlace(OpOperand &operand, AnalysisState &state);

  /// Set the inPlace bufferization spec to false.
  void bufferizeOutOfPlace(OpOperand &operand);

  /// Return true if `v1` and `v2` may bufferize to aliasing buffers.
  bool areAliasingBufferizedValues(Value v1, Value v2) const {
    return aliasInfo.isEquivalent(v1, v2);
  }

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
class OneShotAnalysisState : public AnalysisState {
public:
  OneShotAnalysisState(Operation *op,
                       const OneShotBufferizationOptions &options);

  OneShotAnalysisState(const OneShotAnalysisState &) = delete;

  ~OneShotAnalysisState() override = default;

  static bool classof(const AnalysisState *base) {
    return base->getType() == TypeID::get<OneShotAnalysisState>();
  }

  /// Return a reference to the BufferizationOptions.
  const OneShotBufferizationOptions &getOptions() const {
    return static_cast<const OneShotBufferizationOptions &>(
        AnalysisState::getOptions());
  }

  /// Return a reference to the BufferizationAliasInfo.
  BufferizationAliasInfo &getAliasInfo() { return aliasInfo; }

  /// Return `true` if the given OpResult has been decided to bufferize inplace.
  bool isInPlace(OpOperand &opOperand) const override;

  /// Return true if `v1` and `v2` bufferize to equivalent buffers.
  bool areEquivalentBufferizedValues(Value v1, Value v2) const override;

  /// Return true if `v1` and `v2` may bufferize to aliasing buffers.
  bool areAliasingBufferizedValues(Value v1, Value v2) const override;

  /// Return `true` if the given tensor has undefined contents.
  bool hasUndefinedContents(OpOperand *opOperand) const override;

  /// Return true if the given tensor (or an aliasing tensor) is yielded from
  /// the containing block. Also include all aliasing tensors in the same block.
  bool isTensorYielded(Value tensor) const override;

  /// Find all tensor values in the given operation that have undefined contents
  /// and store them in `undefinedTensorUses`.
  void gatherUndefinedTensorUses(Operation *op);

  /// Find all tensors that are yielded/returned from a block and store them in
  /// `yieldedTensors`. Also include all aliasing tensors in the same block.
  void gatherYieldedTensors(Operation *op);

  /// Return true if the buffer of the given tensor value is written to. Must
  /// not be called for values inside not yet analyzed functions.
  bool isValueWritten(Value value) const;

  /// Return true if the buffer of the given tensor value is writable.
  bool isWritable(Value value) const;

  /// Base class for OneShotAnalysisState extensions that allow
  /// OneShotAnalysisState to contain user-specified information in the state
  /// object. Clients are expected to derive this class, add the desired fields,
  /// and make the derived class compatible with the MLIR TypeID mechanism.
  ///
  /// ```mlir
  /// class MyExtension final : public OneShotAnalysisState::Extension {
  /// public:
  ///   MyExtension(OneShotAnalysisState &state, int myData)
  ///       : Extension(state) {...}
  /// private:
  ///   int mySupplementaryData;
  /// };
  /// ```
  ///
  /// Instances of this and derived classes are not expected to be created by
  /// the user, instead they are directly constructed within a
  /// OneShotAnalysisState. A OneShotAnalysisState can only contain one
  /// extension with the given TypeID. Extensions can be obtained from a
  /// OneShotAnalysisState instance.
  ///
  /// ```mlir
  /// state.addExtension<MyExtension>(/*myData=*/42);
  /// MyExtension *ext = state.getExtension<MyExtension>();
  /// ext->doSomething();
  /// ```
  class Extension {
    // Allow OneShotAnalysisState to allocate Extensions.
    friend class OneShotAnalysisState;

  public:
    /// Base virtual destructor.
    // Out-of-line definition ensures symbols are emitted in a single object
    // file.
    virtual ~Extension();

  protected:
    /// Constructs an extension of the given state object.
    Extension(OneShotAnalysisState &state) : state(state) {}

    /// Provides read-only access to the parent OneShotAnalysisState object.
    const OneShotAnalysisState &getAnalysisState() const { return state; }

  private:
    /// Back-reference to the state that is being extended.
    OneShotAnalysisState &state;
  };

  /// Adds a new Extension of the type specified as template parameter,
  /// constructing it with the arguments provided. The extension is owned by the
  /// OneShotAnalysisState. It is expected that the state does not already have
  /// an extension of the same type. Extension constructors are expected to take
  /// a reference to OneShotAnalysisState as first argument, automatically
  /// supplied by this call.
  template <typename Ty, typename... Args>
  Ty &addExtension(Args &&...args) {
    static_assert(
        std::is_base_of<Extension, Ty>::value,
        "only a class derived from OneShotAnalysisState::Extension is allowed");
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
        "only a class derived from OneShotAnalysisState::Extension is allowed");
    auto iter = extensions.find(TypeID::get<Ty>());
    if (iter == extensions.end())
      return nullptr;
    return static_cast<Ty *>(iter->second.get());
  }

  /// Returns the extension of the specified type.
  template <typename Ty>
  const Ty *getExtension() const {
    return const_cast<OneShotAnalysisState *>(this)->getExtension<Ty>();
  }

private:
  /// `aliasInfo` keeps track of aliasing and equivalent values. Only internal
  /// functions and `runOneShotBufferize` may access this object.
  BufferizationAliasInfo aliasInfo;

  /// A set of all tensors (and maybe aliasing tensors) that yielded from a
  /// block.
  DenseSet<Value> yieldedTensors;

  /// A set of uses of tensors that have undefined contents.
  DenseSet<OpOperand *> undefinedTensorUses;

  /// Extensions attached to the state, identified by the TypeID of their type.
  /// Only one extension of any given type is allowed.
  DenseMap<TypeID, std::unique_ptr<Extension>> extensions;
};

/// Analyze `op` and its nested ops. Bufferization decisions are stored in
/// `state`.
LogicalResult analyzeOp(Operation *op, OneShotAnalysisState &state);

/// Run One-Shot Bufferize on the given op: Analysis + Bufferization
LogicalResult runOneShotBufferize(Operation *op,
                                  const OneShotBufferizationOptions &options);

} // namespace bufferization
} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::bufferization::OneShotAnalysisState)

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ONESHOTANALYSIS_H
