# Pattern Rewriting : Generic DAG-to-DAG Rewriting

[TOC]

This document details the design and API of the pattern rewriting infrastructure
present in MLIR, a general DAG-to-DAG transformation framework. This framework
is widely used throughout MLIR for canonicalization, conversion, and general
transformation.

For an introduction to DAG-to-DAG transformation, and the rationale behind this
framework please take a look at the
[Generic DAG Rewriter Rationale](Rationale/RationaleGenericDAGRewriter.md).

## Introduction

The pattern rewriting framework can largely be decomposed into two parts:
Pattern Definition and Pattern Application.

## Defining Patterns

Patterns are defined by inheriting from the `RewritePattern` class. This class
represents the base class of all rewrite patterns within MLIR, and is comprised
of the following components:

### Benefit

This is the expected benefit of applying a given pattern. This benefit is static
upon construction of the pattern, but may be computed dynamically at pattern
initialization time, e.g. allowing the benefit to be derived from domain
specific information (like the target architecture). This limitation allows for
performing pattern fusion and compiling patterns into an efficient state
machine, and
[Thier, Ertl, and Krall](https://dl.acm.org/citation.cfm?id=3179501) have shown
that match predicates eliminate the need for dynamically computed costs in
almost all cases: you can simply instantiate the same pattern one time for each
possible cost and use the predicate to guard the match.

### Root Operation Name (Optional)

The name of the root operation that this pattern matches against. If specified,
only operations with the given root name will be provided to the `match` and
`rewrite` implementation. If not specified, any operation type may be provided.
The root operation name should be provided whenever possible, because it
simplifies the analysis of patterns when applying a cost model. To match any
operation type, a special tag must be provided to make the intent explicit:
`MatchAnyOpTypeTag`.

### `match` and `rewrite` implementation

This is the chunk of code that matches a given root `Operation` and performs a
rewrite of the IR. A `RewritePattern` can specify this implementation either via
separate `match` and `rewrite` methods, or via a combined `matchAndRewrite`
method. When using the combined `matchAndRewrite` method, no IR mutation should
take place before the match is deemed successful. The combined `matchAndRewrite`
is useful when non-trivially recomputable information is required by the
matching and rewriting phase. See below for examples:

```c++
class MyPattern : public RewritePattern {
public:
  /// This overload constructs a pattern that only matches operations with the
  /// root name of `MyOp`.
  MyPattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(MyOp::getOperationName(), benefit, context) {}
  /// This overload constructs a pattern that matches any operation type.
  MyPattern(PatternBenefit benefit)
      : RewritePattern(benefit, MatchAnyOpTypeTag()) {}

  /// In this section, the `match` and `rewrite` implementation is specified
  /// using the separate hooks.
  LogicalResult match(Operation *op) const override {
    // The `match` method returns `success()` if the pattern is a match, failure
    // otherwise.
    // ...
  }
  void rewrite(Operation *op, PatternRewriter &rewriter) {
    // The `rewrite` method performs mutations on the IR rooted at `op` using
    // the provided rewriter. All mutations must go through the provided
    // rewriter.
  }

  /// In this section, the `match` and `rewrite` implementation is specified
  /// using a single hook.
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) {
    // The `matchAndRewrite` method performs both the matching and the mutation.
    // Note that the match must reach a successful point before IR mutation may
    // take place.
  }
};
```

#### Restrictions

Within the `match` section of a pattern, the following constraints apply:

*   No mutation of the IR is allowed.

Within the `rewrite` section of a pattern, the following constraints apply:

*   All IR mutations, including creation, *must* be performed by the given
    `PatternRewriter`. This class provides hooks for performing all of the
    possible mutations that may take place within a pattern. For example, this
    means that an operation should not be erased via its `erase` method. To
    erase an operation, the appropriate `PatternRewriter` hook (in this case
    `eraseOp`) should be used instead.
*   The root operation is required to either be: updated in-place, replaced, or
    erased.

### Application Recursion

Recursion is an important topic in the context of pattern rewrites, as a pattern
may often be applicable to its own result. For example, imagine a pattern that
peels a single iteration from a loop operation. If the loop has multiple
peelable iterations, this pattern may apply multiple times during the
application process. By looking at the implementation of this pattern, the bound
for recursive application may be obvious, e.g. there are no peelable iterations
within the loop, but from the perspective of the pattern driver this recursion
is potentially dangerous. Often times the recursive application of a pattern
indicates a bug in the matching logic. These types of bugs generally do not
cause crashes, but create infinite loops within the application process. Given
this, the pattern rewriting infrastructure conservatively assumes that no
patterns have a proper bounded recursion, and will fail if recursion is
detected. A pattern that is known to have proper support for handling recursion
can signal this by calling `setHasBoundedRewriteRecursion` when initializing the
pattern. This will signal to the pattern driver that recursive application of
this pattern may happen, and the pattern is equipped to safely handle it.

### Debug Names and Labels

To aid in debugging, patterns may specify: a debug name (via `setDebugName`),
which should correspond to an identifier that uniquely identifies the specific
pattern; and a set of debug labels (via `addDebugLabels`), which correspond to
identifiers that uniquely identify groups of patterns. This information is used
by various utilities to aid in the debugging of pattern rewrites, e.g. in debug
logs, to provide pattern filtering, etc. A simple code example is shown below:

```c++
class MyPattern : public RewritePattern {
public:
  /// Inherit constructors from RewritePattern.
  using RewritePattern::RewritePattern;

  void initialize() {
    setDebugName("MyPattern");
    addDebugLabels("MyRewritePass");
  }

  // ...
};

void populateMyPatterns(RewritePatternSet &patterns, MLIRContext *ctx) {
  // Debug labels may also be attached to patterns during insertion. This allows
  // for easily attaching common labels to groups of patterns.
  patterns.addWithLabel<MyPattern, ...>("MyRewritePatterns", ctx);
}
```

### Initialization

Several pieces of pattern state require explicit initialization by the pattern,
for example setting `setHasBoundedRewriteRecursion` if a pattern safely handles
recursive application. This pattern state can be initialized either in the
constructor of the pattern or via the utility `initialize` hook. Using the
`initialize` hook removes the need to redefine pattern constructors just to
inject additional pattern state initialization. An example is shown below:

```c++
class MyPattern : public RewritePattern {
public:
  /// Inherit the constructors from RewritePattern.
  using RewritePattern::RewritePattern;

  /// Initialize the pattern.
  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    setHasBoundedRewriteRecursion();
  }

  // ...
};
```

### Construction

Constructing a RewritePattern should be performed by using the static
`RewritePattern::create<T>` utility method. This method ensures that the pattern
is properly initialized and prepared for insertion into a `RewritePatternSet`.

## Pattern Rewriter

A `PatternRewriter` is a special class that allows for a pattern to communicate
with the driver of pattern application. As noted above, *all* IR mutations,
including creations, are required to be performed via the `PatternRewriter`
class. This is required because the underlying pattern driver may have state
that would be invalidated when a mutation takes place. Examples of some of the
more prevalent `PatternRewriter` API is shown below, please refer to the
[class documentation](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/PatternMatch.h#L235)
for a more up-to-date listing of the available API:

*   Erase an Operation : `eraseOp`

This method erases an operation that either has no results, or whose results are
all known to have no uses.

*   Notify why a `match` failed : `notifyMatchFailure`

This method allows for providing a diagnostic message within a `matchAndRewrite`
as to why a pattern failed to match. How this message is displayed back to the
user is determined by the specific pattern driver.

*   Replace an Operation : `replaceOp`/`replaceOpWithNewOp`

This method replaces an operation's results with a set of provided values, and
erases the operation.

*   Update an Operation in-place : `(start|cancel|finalize)RootUpdate`

This is a collection of methods that provide a transaction-like API for updating
the attributes, location, operands, or successors of an operation in-place
within a pattern. An in-place update transaction is started with
`startRootUpdate`, and may either be canceled or finalized with
`cancelRootUpdate` and `finalizeRootUpdate` respectively. A convenience wrapper,
`updateRootInPlace`, is provided that wraps a `start` and `finalize` around a
callback.

*   OpBuilder API

The `PatternRewriter` inherits from the `OpBuilder` class, and thus provides all
of the same functionality present within an `OpBuilder`. This includes operation
creation, as well as many useful attribute and type construction methods.

## Pattern Application

After a set of patterns have been defined, they are collected and provided to a
specific driver for application. A driver consists of several high levels parts:

*   Input `RewritePatternSet`

The input patterns to a driver are provided in the form of an
`RewritePatternSet`. This class provides a simplified API for building a
list of patterns.

*   Driver-specific `PatternRewriter`

To ensure that the driver state does not become invalidated by IR mutations
within the pattern rewriters, a driver must provide a `PatternRewriter` instance
with the necessary hooks overridden. If a driver does not need to hook into
certain mutations, a default implementation is provided that will perform the
mutation directly.

*   Pattern Application and Cost Model

Each driver is responsible for defining its own operation visitation order as
well as pattern cost model, but the final application is performed via a
`PatternApplicator` class. This class takes as input the
`RewritePatternSet` and transforms the patterns based upon a provided
cost model. This cost model computes a final benefit for a given pattern, using
whatever driver specific information necessary. After a cost model has been
computed, the driver may begin to match patterns against operations using
`PatternApplicator::matchAndRewrite`.

An example is shown below:

```c++
class MyPattern : public RewritePattern {
public:
  MyPattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(MyOp::getOperationName(), benefit, context) {}
};

/// Populate the pattern list.
void collectMyPatterns(RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<MyPattern>(/*benefit=*/1, ctx);
}

/// Define a custom PatternRewriter for use by the driver.
class MyPatternRewriter : public PatternRewriter {
public:
  MyPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}

  /// Override the necessary PatternRewriter hooks here.
};

/// Apply the custom driver to `op`.
void applyMyPatternDriver(Operation *op,
                          const RewritePatternSet &patterns) {
  // Initialize the custom PatternRewriter.
  MyPatternRewriter rewriter(op->getContext());

  // Create the applicator and apply our cost model.
  PatternApplicator applicator(patterns);
  applicator.applyCostModel([](const Pattern &pattern) {
    // Apply a default cost model.
    // Note: This is just for demonstration, if the default cost model is truly
    //       desired `applicator.applyDefaultCostModel()` should be used
    //       instead.
    return pattern.getBenefit();
  });

  // Try to match and apply a pattern.
  LogicalResult result = applicator.matchAndRewrite(op, rewriter);
  if (failed(result)) {
    // ... No patterns were applied.
  }
  // ... A pattern was successfully applied.
}
```

## Common Pattern Drivers

MLIR provides several common pattern drivers that serve a variety of different
use cases.

### Dialect Conversion Driver

This driver provides a framework in which to perform operation conversions
between, and within dialects using a concept of "legality". This framework
allows for transforming illegal operations to those supported by a provided
conversion target, via a set of pattern-based operation rewriting patterns. This
framework also provides support for type conversions. More information on this
driver can be found [here](DialectConversion.md).

### Greedy Pattern Rewrite Driver

This driver walks the provided operations and greedily applies the patterns that
locally have the most benefit. The benefit of
a pattern is decided solely by the benefit specified on the pattern, and the
relative order of the pattern within the pattern list (when two patterns have
the same local benefit). Patterns are iteratively applied to operations until a
fixed point is reached, at which point the driver finishes. This driver may be
used via the following: `applyPatternsAndFoldGreedily` and
`applyOpPatternsAndFold`. The latter of which only applies patterns to the
provided operation, and will not traverse the IR.

The driver is configurable and supports two modes: 1) you may opt-in to a
"top-down" traversal, which seeds the worklist with each operation top down and
in a pre-order over the region tree.  This is generally more efficient in
compile time.  2) the default is a "bottom up" traversal, which builds the
initial worklist with a postorder traversal of the region tree.  This may
match larger patterns with ambiguous pattern sets.

Note: This driver is the one used by the [canonicalization](Canonicalization.md)
[pass](Passes.md/#-canonicalize-canonicalize-operations) in MLIR.

### Debugging

To debug the execution of the greedy pattern rewrite driver,
`-debug-only=greedy-rewriter` may be used. This command line flag activates
LLVM's debug logging infrastructure solely for the greedy pattern rewriter. The
output is formatted as a tree structure, mirroring the structure of the pattern
application process. This output contains all of the actions performed by the
rewriter, how operations get processed and patterns are applied, and why they
fail.

Example output is shown below:

```
//===-------------------------------------------===//
Processing operation : 'std.cond_br'(0x60f000001120) {
  "std.cond_br"(%arg0)[^bb2, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (i1) -> ()

  * Pattern SimplifyConstCondBranchPred : 'std.cond_br -> ()' {
  } -> failure : pattern failed to match

  * Pattern SimplifyCondBranchIdenticalSuccessors : 'std.cond_br -> ()' {
    ** Insert  : 'std.br'(0x60b000003690)
    ** Replace : 'std.cond_br'(0x60f000001120)
  } -> success : pattern applied successfully
} -> success : pattern matched
//===-------------------------------------------===//
```

This output is describing the processing of a `std.cond_br` operation. We first
try to apply the `SimplifyConstCondBranchPred`, which fails. From there, another
pattern (`SimplifyCondBranchIdenticalSuccessors`) is applied that matches the
`std.cond_br` and replaces it with a `std.br`.

## Debugging

### Pattern Filtering

To simplify test case definition and reduction, the `FrozenRewritePatternSet`
class provides built-in support for filtering which patterns should be provided
to the pattern driver for application. Filtering behavior is specified by
providing a `disabledPatterns` and `enabledPatterns` list when constructing the
`FrozenRewritePatternSet`. The `disabledPatterns` list should contain a set of
debug names or labels for patterns that are disabled during pattern application,
i.e. which patterns should be filtered out. The `enabledPatterns` list should
contain a set of debug names or labels for patterns that are enabled during
pattern application, patterns that do not satisfy this constraint are filtered
out. Note that patterns specified by the `disabledPatterns` list will be
filtered out even if they match criteria in the `enabledPatterns` list. An
example is shown below:

```c++
void MyPass::initialize(MLIRContext *context) {
  // No patterns are explicitly disabled.
  SmallVector<std::string> disabledPatterns;
  // Enable only patterns with a debug name or label of `MyRewritePatterns`.
  SmallVector<std::string> enabledPatterns(1, "MyRewritePatterns");

  RewritePatternSet rewritePatterns(context);
  // ...
  frozenPatterns = FrozenRewritePatternSet(rewritePatterns, disabledPatterns,
                                           enabledPatterns);
}
```

### Common Pass Utilities

Passes that utilize rewrite patterns should aim to provide a common set of
options and toggles to simplify the debugging experience when switching between
different passes/projects/etc. To aid in this endeavor, MLIR provides a common
set of utilities that can be easily included when defining a custom pass. These
are defined in `mlir/RewritePassUtil.td`; an example usage is shown below:

```tablegen
def MyRewritePass : Pass<"..."> {
  let summary = "...";
  let constructor = "createMyRewritePass()";

  // Inherit the common pattern rewrite options from `RewritePassUtils`.
  let options = RewritePassUtils.options;
}
```

#### Rewrite Pass Options

This section documents common pass options that are useful for controlling the
behavior of rewrite pattern application.

##### Pattern Filtering

Two common pattern filtering options are exposed, `disable-patterns` and
`enable-patterns`, matching the behavior of the `disabledPatterns` and
`enabledPatterns` lists described in the [Pattern Filtering](#pattern-filtering)
section above. A snippet of the tablegen definition of these options is shown
below:

```tablegen
ListOption<"disabledPatterns", "disable-patterns", "std::string",
           "Labels of patterns that should be filtered out during application",
           "llvm::cl::MiscFlags::CommaSeparated">,
ListOption<"enabledPatterns", "enable-patterns", "std::string",
           "Labels of patterns that should be used during application, all "
           "other patterns are filtered out",
           "llvm::cl::MiscFlags::CommaSeparated">,
```

These options may be used to provide filtering behavior when constructing any
`FrozenRewritePatternSet`s within the pass:

```c++
void MyRewritePass::initialize(MLIRContext *context) {
  RewritePatternSet rewritePatterns(context);
  // ...

  // When constructing the `FrozenRewritePatternSet`, we provide the filter
  // list options.
  frozenPatterns = FrozenRewritePatternSet(rewritePatterns, disabledPatterns,
                                           enabledPatterns);
}
```
