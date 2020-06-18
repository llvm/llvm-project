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
  /*implicit*/ PatternBenefit(unsigned benefit);
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

private:
  PatternBenefit() : representation(ImpossibleToMatchSentinel) {}
  unsigned short representation;
};

//===----------------------------------------------------------------------===//
// Pattern class
//===----------------------------------------------------------------------===//

/// Instances of Pattern can be matched against SSA IR.  These matches get used
/// in ways dependent on their subclasses and the driver doing the matching.
/// For example, RewritePatterns implement a rewrite from one matched pattern
/// to a replacement DAG tile.
class Pattern {
public:
  /// Return the benefit (the inverse of "cost") of matching this pattern.  The
  /// benefit of a Pattern is always static - rewrites that may have dynamic
  /// benefit can be instantiated multiple times (different Pattern instances)
  /// for each benefit that they may return, and be guarded by different match
  /// condition predicates.
  PatternBenefit getBenefit() const { return benefit; }

  /// Return the root node that this pattern matches.  Patterns that can
  /// match multiple root types are instantiated once per root.
  OperationName getRootKind() const { return rootKind; }

  //===--------------------------------------------------------------------===//
  // Implementation hooks for patterns to implement.
  //===--------------------------------------------------------------------===//

  /// Attempt to match against code rooted at the specified operation,
  /// which is the same operation code as getRootKind().
  virtual LogicalResult match(Operation *op) const = 0;

  virtual ~Pattern() {}

protected:
  /// Patterns must specify the root operation name they match against, and can
  /// also specify the benefit of the pattern matching.
  Pattern(StringRef rootName, PatternBenefit benefit, MLIRContext *context);

private:
  const OperationName rootKind;
  const PatternBenefit benefit;

  virtual void anchor();
};

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
  /// Rewrite the IR rooted at the specified operation with the result of
  /// this pattern, generating any new operations with the specified
  /// builder.  If an unexpected error is encountered (an internal
  /// compiler error), it is emitted through the normal MLIR diagnostic
  /// hooks and the IR is left in a valid state.
  virtual void rewrite(Operation *op, PatternRewriter &rewriter) const;

  /// Attempt to match against code rooted at the specified operation,
  /// which is the same operation code as getRootKind().
  LogicalResult match(Operation *op) const override;

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

  /// Returns true if this pattern is known to result in recursive application,
  /// i.e. this pattern may generate IR that also matches this pattern, but is
  /// known to bound the recursion. This signals to a rewriter that it is safe
  /// to apply this pattern recursively to generated IR.
  virtual bool hasBoundedRewriteRecursion() const { return false; }

  /// Return a list of operations that may be generated when rewriting an
  /// operation instance with this pattern.
  ArrayRef<OperationName> getGeneratedOps() const { return generatedOps; }

protected:
  /// Patterns must specify the root operation name they match against, and can
  /// also specify the benefit of the pattern matching.
  RewritePattern(StringRef rootName, PatternBenefit benefit,
                 MLIRContext *context)
      : Pattern(rootName, benefit, context) {}
  /// Patterns must specify the root operation name they match against, and can
  /// also specify the benefit of the pattern matching. They can also specify
  /// the names of operations that may be generated during a successful rewrite.
  RewritePattern(StringRef rootName, ArrayRef<StringRef> generatedNames,
                 PatternBenefit benefit, MLIRContext *context);

  /// A list of the potential operations that may be generated when rewriting
  /// an op with this pattern.
  SmallVector<OperationName, 2> generatedOps;
};

/// OpRewritePattern is a wrapper around RewritePattern that allows for
/// matching and rewriting against an instance of a derived operation class as
/// opposed to a raw Operation.
template <typename SourceOp> struct OpRewritePattern : public RewritePattern {
  /// Patterns must specify the root operation name they match against, and can
  /// also specify the benefit of the pattern matching.
  OpRewritePattern(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(SourceOp::getOperationName(), benefit, context) {}

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

//===----------------------------------------------------------------------===//
// PatternRewriter class
//===----------------------------------------------------------------------===//

/// This class coordinates the application of a pattern to the current function,
/// providing a way to create operations and keep track of what gets deleted.
///
/// These class serves two purposes:
///  1) it is the interface that patterns interact with to make mutations to the
///     IR they are being applied to.
///  2) It is a base class that clients of the PatternMatcher use when they want
///     to apply patterns and observe their effects (e.g. to keep worklists or
///     other data structures up to date).
///
class PatternRewriter : public OpBuilder, public OpBuilder::Listener {
public:
  /// Create operation of specific op type at the current insertion point
  /// without verifying to see if it is valid.
  template <typename OpTy, typename... Args>
  OpTy create(Location location, Args... args) {
    OperationState state(location, OpTy::getOperationName());
    OpTy::build(*this, state, args...);
    auto *op = createOperation(state);
    auto result = dyn_cast<OpTy>(op);
    assert(result && "Builder didn't return the right type");
    return result;
  }

  /// Creates an operation of specific op type at the current insertion point.
  /// If the result is an invalid op (the verifier hook fails), emit an error
  /// and return null.
  template <typename OpTy, typename... Args>
  OpTy createChecked(Location location, Args... args) {
    OperationState state(location, OpTy::getOperationName());
    OpTy::build(*this, state, args...);
    auto *op = createOperation(state);

    // If the Operation we produce is valid, return it.
    if (!OpTy::verifyInvariants(op)) {
      auto result = dyn_cast<OpTy>(op);
      assert(result && "Builder didn't return the right type");
      return result;
    }

    // Otherwise, the error message got emitted.  Just remove the operation
    // we made.
    op->erase();
    return OpTy();
  }

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

  /// This method performs the final replacement for a pattern, where the
  /// results of the operation are updated to use the specified list of SSA
  /// values.
  virtual void replaceOp(Operation *op, ValueRange newValues);

  /// Replaces the result op with a new op that is created without verification.
  /// The result values of the two ops must be the same types.
  template <typename OpTy, typename... Args>
  void replaceOpWithNewOp(Operation *op, Args &&... args) {
    auto newOp = create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
    replaceOpWithResultsOfAnotherOp(op, newOp.getOperation());
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

  /// Notify the pattern rewriter that the pattern is failing to match the given
  /// operation, and provide a callback to populate a diagnostic with the reason
  /// why the failure occurred. This method allows for derived rewriters to
  /// optionally hook into the reason why a pattern failed, and display it to
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
  explicit PatternRewriter(MLIRContext *ctx)
      : OpBuilder(ctx, /*listener=*/this) {}
  ~PatternRewriter() override;

  /// These are the callback methods that subclasses can choose to implement if
  /// they would like to be notified about certain types of mutations.

  /// Notify the pattern rewriter that the specified operation is about to be
  /// replaced with another set of operations.  This is called before the uses
  /// of the operation have been changed.
  virtual void notifyRootReplaced(Operation *op) {}

  /// This is called on an operation that a pattern match is removing, right
  /// before the operation is deleted.  At this point, the operation has zero
  /// uses.
  virtual void notifyOperationRemoved(Operation *op) {}

  /// Notify the pattern rewriter that the pattern is failing to match the given
  /// operation, and provide a callback to populate a diagnostic with the reason
  /// why the failure occurred. This method allows for derived rewriters to
  /// optionally hook into the reason why a pattern failed, and display it to
  /// users.
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
// Pattern-driven rewriters
//===----------------------------------------------------------------------===//

class OwningRewritePatternList {
  using PatternListT = std::vector<std::unique_ptr<RewritePattern>>;

public:
  OwningRewritePatternList() = default;

  /// Construct a OwningRewritePatternList populated with the pattern `t` of
  /// type `T`.
  template <typename T>
  OwningRewritePatternList(T &&t) {
    patterns.emplace_back(std::make_unique<T>(std::forward<T>(t)));
  }

  PatternListT::iterator begin() { return patterns.begin(); }
  PatternListT::iterator end() { return patterns.end(); }
  PatternListT::const_iterator begin() const { return patterns.begin(); }
  PatternListT::const_iterator end() const { return patterns.end(); }
  void clear() { patterns.clear(); }

  //===--------------------------------------------------------------------===//
  // Pattern Insertion
  //===--------------------------------------------------------------------===//

  /// Add an instance of each of the pattern types 'Ts' to the pattern list with
  /// the given arguments. Return a reference to `this` for chaining insertions.
  /// Note: ConstructorArg is necessary here to separate the two variadic lists.
  template <typename... Ts, typename ConstructorArg,
            typename... ConstructorArgs,
            typename = std::enable_if_t<sizeof...(Ts) != 0>>
  OwningRewritePatternList &insert(ConstructorArg &&arg,
                                   ConstructorArgs &&... args) {
    // The following expands a call to emplace_back for each of the pattern
    // types 'Ts'. This magic is necessary due to a limitation in the places
    // that a parameter pack can be expanded in c++11.
    // FIXME: In c++17 this can be simplified by using 'fold expressions'.
    using dummy = int[];
    (void)dummy{
        0, (patterns.emplace_back(std::make_unique<Ts>(arg, args...)), 0)...};
    return *this;
  }

private:
  PatternListT patterns;
};

/// This class manages optimization and execution of a group of rewrite
/// patterns, providing an API for finding and applying, the best match against
/// a given node.
///
class RewritePatternMatcher {
public:
  /// Create a RewritePatternMatcher with the specified set of patterns.
  explicit RewritePatternMatcher(const OwningRewritePatternList &patterns);

  /// Try to match the given operation to a pattern and rewrite it. Return
  /// true if any pattern matches.
  bool matchAndRewrite(Operation *op, PatternRewriter &rewriter);

private:
  RewritePatternMatcher(const RewritePatternMatcher &) = delete;
  void operator=(const RewritePatternMatcher &) = delete;

  /// The group of patterns that are matched for optimization through this
  /// matcher.
  std::vector<RewritePattern *> patterns;
};

/// Rewrite the regions of the specified operation, which must be isolated from
/// above, by repeatedly applying the highest benefit patterns in a greedy
/// work-list driven manner. Return true if no more patterns can be matched in
/// the result operation regions.
/// Note: This does not apply patterns to the top-level operation itself.
/// Note: These methods also perform folding and simple dead-code elimination
///       before attempting to match any of the provided patterns.
///
bool applyPatternsAndFoldGreedily(Operation *op,
                                  const OwningRewritePatternList &patterns);
/// Rewrite the given regions, which must be isolated from above.
bool applyPatternsAndFoldGreedily(MutableArrayRef<Region> regions,
                                  const OwningRewritePatternList &patterns);

/// Applies the specified patterns on `op` alone while also trying to fold it,
/// by selecting the highest benefits patterns in a greedy manner. Returns true
/// if no more patterns can be matched. `erased` is set to true if `op` was
/// folded away or erased as a result of becoming dead. Note: This does not
/// apply any patterns recursively to the regions of `op`.
bool applyOpPatternsAndFold(Operation *op,
                            const OwningRewritePatternList &patterns,
                            bool *erased = nullptr);
} // end namespace mlir

#endif // MLIR_PATTERN_MATCH_H
