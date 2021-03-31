//===- PatternMatch.h - PatternMatcher classes -------==---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PATTERNMATCHER_H
#define MLIR_PATTERNMATCHER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/FunctionExtras.h"

namespace mlir {

class PatternRewriter;

//===----------------------------------------------------------------------===//
// PatternBenefit class
//===----------------------------------------------------------------------===//

/// This class represents the benefit of a pattern match in a unitless scheme
/// that ranges from 0 (very little benefit) to 65K.  The most common unit to
/// use here is the "number of operations matched" by the pattern.
///
/// This also has a sentinel representation that can be used for patterns that
/// fail to match.
///
class PatternBenefit {
  enum { ImpossibleToMatchSentinel = 65535 };

public:
  PatternBenefit() : representation(ImpossibleToMatchSentinel) {}
  PatternBenefit(unsigned benefit);
  PatternBenefit(const PatternBenefit &) = default;
  PatternBenefit &operator=(const PatternBenefit &) = default;

  static PatternBenefit impossibleToMatch() { return PatternBenefit(); }
  bool isImpossibleToMatch() const { return *this == impossibleToMatch(); }

  /// If the corresponding pattern can match, return its benefit.  If the
  // corresponding pattern isImpossibleToMatch() then this aborts.
  unsigned short getBenefit() const;

  bool operator==(const PatternBenefit &rhs) const {
    return representation == rhs.representation;
  }
  bool operator!=(const PatternBenefit &rhs) const { return !(*this == rhs); }
  bool operator<(const PatternBenefit &rhs) const {
    return representation < rhs.representation;
  }
  bool operator>(const PatternBenefit &rhs) const { return rhs < *this; }
  bool operator<=(const PatternBenefit &rhs) const { return !(*this > rhs); }
  bool operator>=(const PatternBenefit &rhs) const { return !(*this < rhs); }

private:
  unsigned short representation;
};

//===----------------------------------------------------------------------===//
// Pattern
//===----------------------------------------------------------------------===//

/// This class contains all of the data related to a pattern, but does not
/// contain any methods or logic for the actual matching. This class is solely
/// used to interface with the metadata of a pattern, such as the benefit or
/// root operation.
class Pattern {
  /// This enum represents the kind of value used to select the root operations
  /// that match this pattern.
  enum class RootKind {
    /// The pattern root matches "any" operation.
    Any,
    /// The pattern root is matched using a concrete operation name.
    OperationName,
    /// The pattern root is matched using an interface ID.
    InterfaceID,
    /// The patter root is matched using a trait ID.
    TraitID
  };

public:
  /// Return a list of operations that may be generated when rewriting an
  /// operation instance with this pattern.
  ArrayRef<OperationName> getGeneratedOps() const { return generatedOps; }

  /// Return the root node that this pattern matches. Patterns that can match
  /// multiple root types return None.
  Optional<OperationName> getRootKind() const {
    if (rootKind == RootKind::OperationName)
      return OperationName::getFromOpaquePointer(rootValue);
    return llvm::None;
  }

  /// Return the interface ID used to match the root operation of this pattern.
  /// If the pattern does not use an interface ID for deciding the root match,
  /// this returns None.
  Optional<TypeID> getRootInterfaceID() const {
    if (rootKind == RootKind::InterfaceID)
      return TypeID::getFromOpaquePointer(rootValue);
    return llvm::None;
  }

  /// Return the trait ID used to match the root operation of this pattern.
  /// If the pattern does not use a trait ID for deciding the root match, this
  /// returns None.
  Optional<TypeID> getRootTraitID() const {
    if (rootKind == RootKind::TraitID)
      return TypeID::getFromOpaquePointer(rootValue);
    return llvm::None;
  }

  /// Return the benefit (the inverse of "cost") of matching this pattern.  The
  /// benefit of a Pattern is always static - rewrites that may have dynamic
  /// benefit can be instantiated multiple times (different Pattern instances)
  /// for each benefit that they may return, and be guarded by different match
  /// condition predicates.
  PatternBenefit getBenefit() const { return benefit; }

  /// Returns true if this pattern is known to result in recursive application,
  /// i.e. this pattern may generate IR that also matches this pattern, but is
  /// known to bound the recursion. This signals to a rewrite driver that it is
  /// safe to apply this pattern recursively to generated IR.
  bool hasBoundedRewriteRecursion() const {
    return contextAndHasBoundedRecursion.getInt();
  }

  /// Return the MLIRContext used to create this pattern.
  MLIRContext *getContext() const {
    return contextAndHasBoundedRecursion.getPointer();
  }

protected:
  /// This class acts as a special tag that makes the desire to match "any"
  /// operation type explicit. This helps to avoid unnecessary usages of this
  /// feature, and ensures that the user is making a conscious decision.
  struct MatchAnyOpTypeTag {};
  /// This class acts as a special tag that makes the desire to match any
  /// operation that implements a given interface explicit. This helps to avoid
  /// unnecessary usages of this feature, and ensures that the user is making a
  /// conscious decision.
  struct MatchInterfaceOpTypeTag {};
  /// This class acts as a special tag that makes the desire to match any
  /// operation that implements a given trait explicit. This helps to avoid
  /// unnecessary usages of this feature, and ensures that the user is making a
  /// conscious decision.
  struct MatchTraitOpTypeTag {};

  /// Construct a pattern with a certain benefit that matches the operation
  /// with the given root name.
  Pattern(StringRef rootName, PatternBenefit benefit, MLIRContext *context,
          ArrayRef<StringRef> generatedNames = {});
  /// Construct a pattern that may match any operation type. `generatedNames`
  /// contains the names of operations that may be generated during a successful
  /// rewrite. `MatchAnyOpTypeTag` is just a tag to ensure that the "match any"
  /// behavior is what the user actually desired, `MatchAnyOpTypeTag()` should
  /// always be supplied here.
  Pattern(MatchAnyOpTypeTag tag, PatternBenefit benefit, MLIRContext *context,
          ArrayRef<StringRef> generatedNames = {});
  /// Construct a pattern that may match any operation that implements the
  /// interface defined by the provided `interfaceID`. `generatedNames` contains
  /// the names of operations that may be generated during a successful rewrite.
  /// `MatchInterfaceOpTypeTag` is just a tag to ensure that the "match
  /// interface" behavior is what the user actually desired,
  /// `MatchInterfaceOpTypeTag()` should always be supplied here.
  Pattern(MatchInterfaceOpTypeTag tag, TypeID interfaceID,
          PatternBenefit benefit, MLIRContext *context,
          ArrayRef<StringRef> generatedNames = {});
  /// Construct a pattern that may match any operation that implements the
  /// trait defined by the provided `traitID`. `generatedNames` contains the
  /// names of operations that may be generated during a successful rewrite.
  /// `MatchTraitOpTypeTag` is just a tag to ensure that the "match trait"
  /// behavior is what the user actually desired, `MatchTraitOpTypeTag()` should
  /// always be supplied here.
  Pattern(MatchTraitOpTypeTag tag, TypeID traitID, PatternBenefit benefit,
          MLIRContext *context, ArrayRef<StringRef> generatedNames = {});

  /// Set the flag detailing if this pattern has bounded rewrite recursion or
  /// not.
  void setHasBoundedRewriteRecursion(bool hasBoundedRecursionArg = true) {
    contextAndHasBoundedRecursion.setInt(hasBoundedRecursionArg);
  }

private:
  Pattern(const void *rootValue, RootKind rootKind,
          ArrayRef<StringRef> generatedNames, PatternBenefit benefit,
          MLIRContext *context);

  /// The value used to match the root operation of the pattern.
  const void *rootValue;
  RootKind rootKind;

  /// The expected benefit of matching this pattern.
  const PatternBenefit benefit;

  /// The context this pattern was created from, and a boolean flag indicating
  /// whether this pattern has bounded recursion or not.
  llvm::PointerIntPair<MLIRContext *, 1, bool> contextAndHasBoundedRecursion;

  /// A list of the potential operations that may be generated when rewriting
  /// an op with this pattern.
  SmallVector<OperationName, 2> generatedOps;
};

//===----------------------------------------------------------------------===//
// RewritePattern
//===----------------------------------------------------------------------===//

/// RewritePattern is the common base class for all DAG to DAG replacements.
/// There are two possible usages of this class:
///   * Multi-step RewritePattern with "match" and "rewrite"
///     - By overloading the "match" and "rewrite" functions, the user can
///       separate the concerns of matching and rewriting.
///   * Single-step RewritePattern with "matchAndRewrite"
///     - By overloading the "matchAndRewrite" function, the user can perform
///       the rewrite in the same call as the match.
///
class RewritePattern : public Pattern {
public:
  virtual ~RewritePattern() {}

  /// Rewrite the IR rooted at the specified operation with the result of
  /// this pattern, generating any new operations with the specified
  /// builder.  If an unexpected error is encountered (an internal
  /// compiler error), it is emitted through the normal MLIR diagnostic
  /// hooks and the IR is left in a valid state.
  virtual void rewrite(Operation *op, PatternRewriter &rewriter) const;

  /// Attempt to match against code rooted at the specified operation,
  /// which is the same operation code as getRootKind().
  virtual LogicalResult match(Operation *op) const;

  /// Attempt to match against code rooted at the specified operation,
  /// which is the same operation code as getRootKind(). If successful, this
  /// function will automatically perform the rewrite.
  virtual LogicalResult matchAndRewrite(Operation *op,
                                        PatternRewriter &rewriter) const {
    if (succeeded(match(op))) {
      rewrite(op, rewriter);
      return success();
    }
    return failure();
  }

protected:
  /// Inherit the base constructors from `Pattern`.
  using Pattern::Pattern;

  /// An anchor for the virtual table.
  virtual void anchor();
};

namespace detail {
/// OpOrInterfaceRewritePatternBase is a wrapper around RewritePattern that
/// allows for matching and rewriting against an instance of a derived operation
/// class or Interface.
template <typename SourceOp>
struct OpOrInterfaceRewritePatternBase : public RewritePattern {
  using RewritePattern::RewritePattern;

  /// Wrappers around the RewritePattern methods that pass the derived op type.
  void rewrite(Operation *op, PatternRewriter &rewriter) const final {
    rewrite(cast<SourceOp>(op), rewriter);
  }
  LogicalResult match(Operation *op) const final {
    return match(cast<SourceOp>(op));
  }
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    return matchAndRewrite(cast<SourceOp>(op), rewriter);
  }

  /// Rewrite and Match methods that operate on the SourceOp type. These must be
  /// overridden by the derived pattern class.
  virtual void rewrite(SourceOp op, PatternRewriter &rewriter) const {
    llvm_unreachable("must override rewrite or matchAndRewrite");
  }
  virtual LogicalResult match(SourceOp op) const {
    llvm_unreachable("must override match or matchAndRewrite");
  }
  virtual LogicalResult matchAndRewrite(SourceOp op,
                                        PatternRewriter &rewriter) const {
    if (succeeded(match(op))) {
      rewrite(op, rewriter);
      return success();
    }
    return failure();
  }
};
} // namespace detail

/// OpRewritePattern is a wrapper around RewritePattern that allows for
/// matching and rewriting against an instance of a derived operation class as
/// opposed to a raw Operation.
template <typename SourceOp>
struct OpRewritePattern
    : public detail::OpOrInterfaceRewritePatternBase<SourceOp> {
  /// Patterns must specify the root operation name they match against, and can
  /// also specify the benefit of the pattern matching.
  OpRewritePattern(MLIRContext *context, PatternBenefit benefit = 1)
      : detail::OpOrInterfaceRewritePatternBase<SourceOp>(
            SourceOp::getOperationName(), benefit, context) {}
};

/// OpInterfaceRewritePattern is a wrapper around RewritePattern that allows for
/// matching and rewriting against an instance of an operation interface instead
/// of a raw Operation.
template <typename SourceOp>
struct OpInterfaceRewritePattern
    : public detail::OpOrInterfaceRewritePatternBase<SourceOp> {
  OpInterfaceRewritePattern(MLIRContext *context, PatternBenefit benefit = 1)
      : detail::OpOrInterfaceRewritePatternBase<SourceOp>(
            Pattern::MatchInterfaceOpTypeTag(), SourceOp::getInterfaceID(),
            benefit, context) {}
};

/// OpTraitRewritePattern is a wrapper around RewritePattern that allows for
/// matching and rewriting against instances of an operation that possess a
/// given trait.
template <template <typename> class TraitType>
class OpTraitRewritePattern : public RewritePattern {
public:
  OpTraitRewritePattern(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(Pattern::MatchTraitOpTypeTag(), TypeID::get<TraitType>(),
                       benefit, context) {}
};

//===----------------------------------------------------------------------===//
// PDLPatternModule
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PDLValue

/// Storage type of byte-code interpreter values. These are passed to constraint
/// functions as arguments.
class PDLValue {
public:
  /// The underlying kind of a PDL value.
  enum class Kind { Attribute, Operation, Type, TypeRange, Value, ValueRange };

  /// Construct a new PDL value.
  PDLValue(const PDLValue &other) = default;
  PDLValue(std::nullptr_t = nullptr) : value(nullptr), kind(Kind::Attribute) {}
  PDLValue(Attribute value)
      : value(value.getAsOpaquePointer()), kind(Kind::Attribute) {}
  PDLValue(Operation *value) : value(value), kind(Kind::Operation) {}
  PDLValue(Type value) : value(value.getAsOpaquePointer()), kind(Kind::Type) {}
  PDLValue(TypeRange *value) : value(value), kind(Kind::TypeRange) {}
  PDLValue(Value value)
      : value(value.getAsOpaquePointer()), kind(Kind::Value) {}
  PDLValue(ValueRange *value) : value(value), kind(Kind::ValueRange) {}

  /// Returns true if the type of the held value is `T`.
  template <typename T>
  bool isa() const {
    assert(value && "isa<> used on a null value");
    return kind == getKindOf<T>();
  }

  /// Attempt to dynamically cast this value to type `T`, returns null if this
  /// value is not an instance of `T`.
  template <typename T,
            typename ResultT = std::conditional_t<
                std::is_convertible<T, bool>::value, T, Optional<T>>>
  ResultT dyn_cast() const {
    return isa<T>() ? castImpl<T>() : ResultT();
  }

  /// Cast this value to type `T`, asserts if this value is not an instance of
  /// `T`.
  template <typename T>
  T cast() const {
    assert(isa<T>() && "expected value to be of type `T`");
    return castImpl<T>();
  }

  /// Get an opaque pointer to the value.
  const void *getAsOpaquePointer() const { return value; }

  /// Return if this value is null or not.
  explicit operator bool() const { return value; }

  /// Return the kind of this value.
  Kind getKind() const { return kind; }

  /// Print this value to the provided output stream.
  void print(raw_ostream &os) const;

private:
  /// Find the index of a given type in a range of other types.
  template <typename...>
  struct index_of_t;
  template <typename T, typename... R>
  struct index_of_t<T, T, R...> : std::integral_constant<size_t, 0> {};
  template <typename T, typename F, typename... R>
  struct index_of_t<T, F, R...>
      : std::integral_constant<size_t, 1 + index_of_t<T, R...>::value> {};

  /// Return the kind used for the given T.
  template <typename T>
  static Kind getKindOf() {
    return static_cast<Kind>(index_of_t<T, Attribute, Operation *, Type,
                                        TypeRange, Value, ValueRange>::value);
  }

  /// The internal implementation of `cast`, that returns the underlying value
  /// as the given type `T`.
  template <typename T>
  std::enable_if_t<llvm::is_one_of<T, Attribute, Type, Value>::value, T>
  castImpl() const {
    return T::getFromOpaquePointer(value);
  }
  template <typename T>
  std::enable_if_t<llvm::is_one_of<T, TypeRange, ValueRange>::value, T>
  castImpl() const {
    return *reinterpret_cast<T *>(const_cast<void *>(value));
  }
  template <typename T>
  std::enable_if_t<std::is_pointer<T>::value, T> castImpl() const {
    return reinterpret_cast<T>(const_cast<void *>(value));
  }

  /// The internal opaque representation of a PDLValue.
  const void *value;
  /// The kind of the opaque value.
  Kind kind;
};

inline raw_ostream &operator<<(raw_ostream &os, PDLValue value) {
  value.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// PDLResultList

/// The class represents a list of PDL results, returned by a native rewrite
/// method. It provides the mechanism with which to pass PDLValues back to the
/// PDL bytecode.
class PDLResultList {
public:
  /// Push a new Attribute value onto the result list.
  void push_back(Attribute value) { results.push_back(value); }

  /// Push a new Operation onto the result list.
  void push_back(Operation *value) { results.push_back(value); }

  /// Push a new Type onto the result list.
  void push_back(Type value) { results.push_back(value); }

  /// Push a new TypeRange onto the result list.
  void push_back(TypeRange value) {
    // The lifetime of a TypeRange can't be guaranteed, so we'll need to
    // allocate a storage for it.
    llvm::OwningArrayRef<Type> storage(value.size());
    llvm::copy(value, storage.begin());
    allocatedTypeRanges.emplace_back(std::move(storage));
    typeRanges.push_back(allocatedTypeRanges.back());
    results.push_back(&typeRanges.back());
  }
  void push_back(ValueTypeRange<OperandRange> value) {
    typeRanges.push_back(value);
    results.push_back(&typeRanges.back());
  }
  void push_back(ValueTypeRange<ResultRange> value) {
    typeRanges.push_back(value);
    results.push_back(&typeRanges.back());
  }

  /// Push a new Value onto the result list.
  void push_back(Value value) { results.push_back(value); }

  /// Push a new ValueRange onto the result list.
  void push_back(ValueRange value) {
    // The lifetime of a ValueRange can't be guaranteed, so we'll need to
    // allocate a storage for it.
    llvm::OwningArrayRef<Value> storage(value.size());
    llvm::copy(value, storage.begin());
    allocatedValueRanges.emplace_back(std::move(storage));
    valueRanges.push_back(allocatedValueRanges.back());
    results.push_back(&valueRanges.back());
  }
  void push_back(OperandRange value) {
    valueRanges.push_back(value);
    results.push_back(&valueRanges.back());
  }
  void push_back(ResultRange value) {
    valueRanges.push_back(value);
    results.push_back(&valueRanges.back());
  }

protected:
  /// Create a new result list with the expected number of results.
  PDLResultList(unsigned maxNumResults) {
    // For now just reserve enough space for all of the results. We could do
    // separate counts per range type, but it isn't really worth it unless there
    // are a "large" number of results.
    typeRanges.reserve(maxNumResults);
    valueRanges.reserve(maxNumResults);
  }

  /// The PDL results held by this list.
  SmallVector<PDLValue> results;
  /// Memory used to store ranges held by the list.
  SmallVector<TypeRange> typeRanges;
  SmallVector<ValueRange> valueRanges;
  /// Memory allocated to store ranges in the result list whose lifetime was
  /// generated in the native function.
  SmallVector<llvm::OwningArrayRef<Type>> allocatedTypeRanges;
  SmallVector<llvm::OwningArrayRef<Value>> allocatedValueRanges;
};

//===----------------------------------------------------------------------===//
// PDLPatternModule

/// A generic PDL pattern constraint function. This function applies a
/// constraint to a given set of opaque PDLValue entities. The second parameter
/// is a set of constant value parameters specified in Attribute form. Returns
/// success if the constraint successfully held, failure otherwise.
using PDLConstraintFunction = std::function<LogicalResult(
    ArrayRef<PDLValue>, ArrayAttr, PatternRewriter &)>;
/// A native PDL rewrite function. This function performs a rewrite on the
/// given set of values and constant parameters. Any results from this rewrite
/// that should be passed back to PDL should be added to the provided result
/// list. This method is only invoked when the corresponding match was
/// successful.
using PDLRewriteFunction = std::function<void(
    ArrayRef<PDLValue>, ArrayAttr, PatternRewriter &, PDLResultList &)>;
/// A generic PDL pattern constraint function. This function applies a
/// constraint to a given opaque PDLValue entity. The second parameter is a set
/// of constant value parameters specified in Attribute form. Returns success if
/// the constraint successfully held, failure otherwise.
using PDLSingleEntityConstraintFunction =
    std::function<LogicalResult(PDLValue, ArrayAttr, PatternRewriter &)>;

/// This class contains all of the necessary data for a set of PDL patterns, or
/// pattern rewrites specified in the form of the PDL dialect. This PDL module
/// contained by this pattern may contain any number of `pdl.pattern`
/// operations.
class PDLPatternModule {
public:
  PDLPatternModule() = default;

  /// Construct a PDL pattern with the given module.
  PDLPatternModule(OwningModuleRef pdlModule)
      : pdlModule(std::move(pdlModule)) {}

  /// Merge the state in `other` into this pattern module.
  void mergeIn(PDLPatternModule &&other);

  /// Return the internal PDL module of this pattern.
  ModuleOp getModule() { return pdlModule.get(); }

  //===--------------------------------------------------------------------===//
  // Function Registry

  /// Register a constraint function.
  void registerConstraintFunction(StringRef name,
                                  PDLConstraintFunction constraintFn);
  /// Register a single entity constraint function.
  template <typename SingleEntityFn>
  std::enable_if_t<!llvm::is_invocable<SingleEntityFn, ArrayRef<PDLValue>,
                                       ArrayAttr, PatternRewriter &>::value>
  registerConstraintFunction(StringRef name, SingleEntityFn &&constraintFn) {
    registerConstraintFunction(
        name, [constraintFn = std::forward<SingleEntityFn>(constraintFn)](
                  ArrayRef<PDLValue> values, ArrayAttr constantParams,
                  PatternRewriter &rewriter) {
          assert(values.size() == 1 &&
                 "expected values to have a single entity");
          return constraintFn(values[0], constantParams, rewriter);
        });
  }

  /// Register a rewrite function.
  void registerRewriteFunction(StringRef name, PDLRewriteFunction rewriteFn);

  /// Return the set of the registered constraint functions.
  const llvm::StringMap<PDLConstraintFunction> &getConstraintFunctions() const {
    return constraintFunctions;
  }
  llvm::StringMap<PDLConstraintFunction> takeConstraintFunctions() {
    return constraintFunctions;
  }
  /// Return the set of the registered rewrite functions.
  const llvm::StringMap<PDLRewriteFunction> &getRewriteFunctions() const {
    return rewriteFunctions;
  }
  llvm::StringMap<PDLRewriteFunction> takeRewriteFunctions() {
    return rewriteFunctions;
  }

  /// Clear out the patterns and functions within this module.
  void clear() {
    pdlModule = nullptr;
    constraintFunctions.clear();
    rewriteFunctions.clear();
  }

private:
  /// The module containing the `pdl.pattern` operations.
  OwningModuleRef pdlModule;

  /// The external functions referenced from within the PDL module.
  llvm::StringMap<PDLConstraintFunction> constraintFunctions;
  llvm::StringMap<PDLRewriteFunction> rewriteFunctions;
};

//===----------------------------------------------------------------------===//
// RewriterBase
//===----------------------------------------------------------------------===//

/// This class coordinates the application of a rewrite on a set of IR,
/// providing a way for clients to track mutations and create new operations.
/// This class serves as a common API for IR mutation between pattern rewrites
/// and non-pattern rewrites, and facilitates the development of shared
/// IR transformation utilities.
class RewriterBase : public OpBuilder, public OpBuilder::Listener {
public:
  /// Move the blocks that belong to "region" before the given position in
  /// another region "parent". The two regions must be different. The caller
  /// is responsible for creating or updating the operation transferring flow
  /// of control to the region and passing it the correct block arguments.
  virtual void inlineRegionBefore(Region &region, Region &parent,
                                  Region::iterator before);
  void inlineRegionBefore(Region &region, Block *before);

  /// Clone the blocks that belong to "region" before the given position in
  /// another region "parent". The two regions must be different. The caller is
  /// responsible for creating or updating the operation transferring flow of
  /// control to the region and passing it the correct block arguments.
  virtual void cloneRegionBefore(Region &region, Region &parent,
                                 Region::iterator before,
                                 BlockAndValueMapping &mapping);
  void cloneRegionBefore(Region &region, Region &parent,
                         Region::iterator before);
  void cloneRegionBefore(Region &region, Block *before);

  /// This method replaces the uses of the results of `op` with the values in
  /// `newValues` when the provided `functor` returns true for a specific use.
  /// The number of values in `newValues` is required to match the number of
  /// results of `op`. `allUsesReplaced`, if non-null, is set to true if all of
  /// the uses of `op` were replaced. Note that in some rewriters, the given
  /// 'functor' may be stored beyond the lifetime of the rewrite being applied.
  /// As such, the function should not capture by reference and instead use
  /// value capture as necessary.
  virtual void
  replaceOpWithIf(Operation *op, ValueRange newValues, bool *allUsesReplaced,
                  llvm::unique_function<bool(OpOperand &) const> functor);
  void replaceOpWithIf(Operation *op, ValueRange newValues,
                       llvm::unique_function<bool(OpOperand &) const> functor) {
    replaceOpWithIf(op, newValues, /*allUsesReplaced=*/nullptr,
                    std::move(functor));
  }

  /// This method replaces the uses of the results of `op` with the values in
  /// `newValues` when a use is nested within the given `block`. The number of
  /// values in `newValues` is required to match the number of results of `op`.
  /// If all uses of this operation are replaced, the operation is erased.
  void replaceOpWithinBlock(Operation *op, ValueRange newValues, Block *block,
                            bool *allUsesReplaced = nullptr);

  /// This method replaces the results of the operation with the specified list
  /// of values. The number of provided values must match the number of results
  /// of the operation.
  virtual void replaceOp(Operation *op, ValueRange newValues);

  /// Replaces the result op with a new op that is created without verification.
  /// The result values of the two ops must be the same types.
  template <typename OpTy, typename... Args>
  OpTy replaceOpWithNewOp(Operation *op, Args &&... args) {
    auto newOp = create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
    replaceOpWithResultsOfAnotherOp(op, newOp.getOperation());
    return newOp;
  }

  /// This method erases an operation that is known to have no uses.
  virtual void eraseOp(Operation *op);

  /// This method erases all operations in a block.
  virtual void eraseBlock(Block *block);

  /// Merge the operations of block 'source' into the end of block 'dest'.
  /// 'source's predecessors must either be empty or only contain 'dest`.
  /// 'argValues' is used to replace the block arguments of 'source' after
  /// merging.
  virtual void mergeBlocks(Block *source, Block *dest,
                           ValueRange argValues = llvm::None);

  // Merge the operations of block 'source' before the operation 'op'. Source
  // block should not have existing predecessors or successors.
  void mergeBlockBefore(Block *source, Operation *op,
                        ValueRange argValues = llvm::None);

  /// Split the operations starting at "before" (inclusive) out of the given
  /// block into a new block, and return it.
  virtual Block *splitBlock(Block *block, Block::iterator before);

  /// This method is used to notify the rewriter that an in-place operation
  /// modification is about to happen. A call to this function *must* be
  /// followed by a call to either `finalizeRootUpdate` or `cancelRootUpdate`.
  /// This is a minor efficiency win (it avoids creating a new operation and
  /// removing the old one) but also often allows simpler code in the client.
  virtual void startRootUpdate(Operation *op) {}

  /// This method is used to signal the end of a root update on the given
  /// operation. This can only be called on operations that were provided to a
  /// call to `startRootUpdate`.
  virtual void finalizeRootUpdate(Operation *op) {}

  /// This method cancels a pending root update. This can only be called on
  /// operations that were provided to a call to `startRootUpdate`.
  virtual void cancelRootUpdate(Operation *op) {}

  /// This method is a utility wrapper around a root update of an operation. It
  /// wraps calls to `startRootUpdate` and `finalizeRootUpdate` around the given
  /// callable.
  template <typename CallableT>
  void updateRootInPlace(Operation *root, CallableT &&callable) {
    startRootUpdate(root);
    callable();
    finalizeRootUpdate(root);
  }

  /// Used to notify the rewriter that the IR failed to be rewritten because of
  /// a match failure, and provide a callback to populate a diagnostic with the
  /// reason why the failure occurred. This method allows for derived rewriters
  /// to optionally hook into the reason why a rewrite failed, and display it to
  /// users.
  template <typename CallbackT>
  std::enable_if_t<!std::is_convertible<CallbackT, Twine>::value, LogicalResult>
  notifyMatchFailure(Operation *op, CallbackT &&reasonCallback) {
#ifndef NDEBUG
    return notifyMatchFailure(op,
                              function_ref<void(Diagnostic &)>(reasonCallback));
#else
    return failure();
#endif
  }
  LogicalResult notifyMatchFailure(Operation *op, const Twine &msg) {
    return notifyMatchFailure(op, [&](Diagnostic &diag) { diag << msg; });
  }
  LogicalResult notifyMatchFailure(Operation *op, const char *msg) {
    return notifyMatchFailure(op, Twine(msg));
  }

protected:
  /// Initialize the builder with this rewriter as the listener.
  explicit RewriterBase(MLIRContext *ctx) : OpBuilder(ctx, /*listener=*/this) {}
  explicit RewriterBase(const OpBuilder &otherBuilder)
      : OpBuilder(otherBuilder) {
    setListener(this);
  }
  ~RewriterBase() override;

  /// These are the callback methods that subclasses can choose to implement if
  /// they would like to be notified about certain types of mutations.

  /// Notify the rewriter that the specified operation is about to be replaced
  /// with another set of operations. This is called before the uses of the
  /// operation have been changed.
  virtual void notifyRootReplaced(Operation *op) {}

  /// This is called on an operation that a rewrite is removing, right before
  /// the operation is deleted. At this point, the operation has zero uses.
  virtual void notifyOperationRemoved(Operation *op) {}

  /// Notify the rewriter that the pattern failed to match the given operation,
  /// and provide a callback to populate a diagnostic with the reason why the
  /// failure occurred. This method allows for derived rewriters to optionally
  /// hook into the reason why a rewrite failed, and display it to users.
  virtual LogicalResult
  notifyMatchFailure(Operation *op,
                     function_ref<void(Diagnostic &)> reasonCallback) {
    return failure();
  }

private:
  /// 'op' and 'newOp' are known to have the same number of results, replace the
  /// uses of op with uses of newOp.
  void replaceOpWithResultsOfAnotherOp(Operation *op, Operation *newOp);
};

//===----------------------------------------------------------------------===//
// IRRewriter
//===----------------------------------------------------------------------===//

/// This class coordinates rewriting a piece of IR outside of a pattern rewrite,
/// providing a way to keep track of the mutations made to the IR. This class
/// should only be used in situations where another `RewriterBase` instance,
/// such as a `PatternRewriter`, is not available.
class IRRewriter : public RewriterBase {
public:
  explicit IRRewriter(MLIRContext *ctx) : RewriterBase(ctx) {}
  explicit IRRewriter(const OpBuilder &builder) : RewriterBase(builder) {}
};

//===----------------------------------------------------------------------===//
// PatternRewriter
//===----------------------------------------------------------------------===//

/// A special type of `RewriterBase` that coordinates the application of a
/// rewrite pattern on the current IR being matched, providing a way to keep
/// track of any mutations made. This class should be used to perform all
/// necessary IR mutations within a rewrite pattern, as the pattern driver may
/// be tracking various state that would be invalidated when a mutation takes
/// place.
class PatternRewriter : public RewriterBase {
public:
  using RewriterBase::RewriterBase;
};

//===----------------------------------------------------------------------===//
// RewritePatternSet
//===----------------------------------------------------------------------===//

class RewritePatternSet {
  using NativePatternListT = std::vector<std::unique_ptr<RewritePattern>>;

public:
  RewritePatternSet(MLIRContext *context) : context(context) {}

  /// Construct a RewritePatternSet populated with the given pattern.
  RewritePatternSet(MLIRContext *context,
                    std::unique_ptr<RewritePattern> pattern)
      : context(context) {
    nativePatterns.emplace_back(std::move(pattern));
  }
  RewritePatternSet(PDLPatternModule &&pattern)
      : context(pattern.getModule()->getContext()),
        pdlPatterns(std::move(pattern)) {}

  MLIRContext *getContext() const { return context; }

  /// Return the native patterns held in this list.
  NativePatternListT &getNativePatterns() { return nativePatterns; }

  /// Return the PDL patterns held in this list.
  PDLPatternModule &getPDLPatterns() { return pdlPatterns; }

  /// Clear out all of the held patterns in this list.
  void clear() {
    nativePatterns.clear();
    pdlPatterns.clear();
  }

  //===--------------------------------------------------------------------===//
  // 'add' methods for adding patterns to the set.
  //===--------------------------------------------------------------------===//

  /// Add an instance of each of the pattern types 'Ts' to the pattern list with
  /// the given arguments. Return a reference to `this` for chaining insertions.
  /// Note: ConstructorArg is necessary here to separate the two variadic lists.
  template <typename... Ts, typename ConstructorArg,
            typename... ConstructorArgs,
            typename = std::enable_if_t<sizeof...(Ts) != 0>>
  RewritePatternSet &add(ConstructorArg &&arg, ConstructorArgs &&... args) {
    // The following expands a call to emplace_back for each of the pattern
    // types 'Ts'. This magic is necessary due to a limitation in the places
    // that a parameter pack can be expanded in c++11.
    // FIXME: In c++17 this can be simplified by using 'fold expressions'.
    (void)std::initializer_list<int>{0, (addImpl<Ts>(arg, args...), 0)...};
    return *this;
  }

  /// Add an instance of each of the pattern types 'Ts'. Return a reference to
  /// `this` for chaining insertions.
  template <typename... Ts>
  RewritePatternSet &add() {
    (void)std::initializer_list<int>{0, (addImpl<Ts>(), 0)...};
    return *this;
  }

  /// Add the given native pattern to the pattern list. Return a reference to
  /// `this` for chaining insertions.
  RewritePatternSet &add(std::unique_ptr<RewritePattern> pattern) {
    nativePatterns.emplace_back(std::move(pattern));
    return *this;
  }

  /// Add the given PDL pattern to the pattern list. Return a reference to
  /// `this` for chaining insertions.
  RewritePatternSet &add(PDLPatternModule &&pattern) {
    pdlPatterns.mergeIn(std::move(pattern));
    return *this;
  }

  // Add a matchAndRewrite style pattern represented as a C function pointer.
  template <typename OpType>
  RewritePatternSet &add(LogicalResult (*implFn)(OpType,
                                                 PatternRewriter &rewriter)) {
    struct FnPattern final : public OpRewritePattern<OpType> {
      FnPattern(LogicalResult (*implFn)(OpType, PatternRewriter &rewriter),
                MLIRContext *context)
          : OpRewritePattern<OpType>(context), implFn(implFn) {}

      LogicalResult matchAndRewrite(OpType op,
                                    PatternRewriter &rewriter) const override {
        return implFn(op, rewriter);
      }

    private:
      LogicalResult (*implFn)(OpType, PatternRewriter &rewriter);
    };
    add(std::make_unique<FnPattern>(std::move(implFn), getContext()));
    return *this;
  }

  //===--------------------------------------------------------------------===//
  // Pattern Insertion
  //===--------------------------------------------------------------------===//

  // TODO: These are soft deprecated in favor of the 'add' methods above.

  /// Add an instance of each of the pattern types 'Ts' to the pattern list with
  /// the given arguments. Return a reference to `this` for chaining insertions.
  /// Note: ConstructorArg is necessary here to separate the two variadic lists.
  template <typename... Ts, typename ConstructorArg,
            typename... ConstructorArgs,
            typename = std::enable_if_t<sizeof...(Ts) != 0>>
  RewritePatternSet &insert(ConstructorArg &&arg, ConstructorArgs &&... args) {
    // The following expands a call to emplace_back for each of the pattern
    // types 'Ts'. This magic is necessary due to a limitation in the places
    // that a parameter pack can be expanded in c++11.
    // FIXME: In c++17 this can be simplified by using 'fold expressions'.
    (void)std::initializer_list<int>{0, (addImpl<Ts>(arg, args...), 0)...};
    return *this;
  }

  /// Add an instance of each of the pattern types 'Ts'. Return a reference to
  /// `this` for chaining insertions.
  template <typename... Ts>
  RewritePatternSet &insert() {
    (void)std::initializer_list<int>{0, (addImpl<Ts>(), 0)...};
    return *this;
  }

  /// Add the given native pattern to the pattern list. Return a reference to
  /// `this` for chaining insertions.
  RewritePatternSet &insert(std::unique_ptr<RewritePattern> pattern) {
    nativePatterns.emplace_back(std::move(pattern));
    return *this;
  }

  /// Add the given PDL pattern to the pattern list. Return a reference to
  /// `this` for chaining insertions.
  RewritePatternSet &insert(PDLPatternModule &&pattern) {
    pdlPatterns.mergeIn(std::move(pattern));
    return *this;
  }

  // Add a matchAndRewrite style pattern represented as a C function pointer.
  template <typename OpType>
  RewritePatternSet &
  insert(LogicalResult (*implFn)(OpType, PatternRewriter &rewriter)) {
    struct FnPattern final : public OpRewritePattern<OpType> {
      FnPattern(LogicalResult (*implFn)(OpType, PatternRewriter &rewriter),
                MLIRContext *context)
          : OpRewritePattern<OpType>(context), implFn(implFn) {}

      LogicalResult matchAndRewrite(OpType op,
                                    PatternRewriter &rewriter) const override {
        return implFn(op, rewriter);
      }

    private:
      LogicalResult (*implFn)(OpType, PatternRewriter &rewriter);
    };
    insert(std::make_unique<FnPattern>(std::move(implFn), getContext()));
    return *this;
  }

private:
  /// Add an instance of the pattern type 'T'. Return a reference to `this` for
  /// chaining insertions.
  template <typename T, typename... Args>
  std::enable_if_t<std::is_base_of<RewritePattern, T>::value>
  addImpl(Args &&... args) {
    nativePatterns.emplace_back(
        std::make_unique<T>(std::forward<Args>(args)...));
  }
  template <typename T, typename... Args>
  std::enable_if_t<std::is_base_of<PDLPatternModule, T>::value>
  addImpl(Args &&... args) {
    pdlPatterns.mergeIn(T(std::forward<Args>(args)...));
  }

  MLIRContext *const context;
  NativePatternListT nativePatterns;
  PDLPatternModule pdlPatterns;
};

// TODO: OwningRewritePatternList is soft-deprecated and will be removed in the
// future.
using OwningRewritePatternList = RewritePatternSet;

} // end namespace mlir

#endif // MLIR_PATTERN_MATCH_H
