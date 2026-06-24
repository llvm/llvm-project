# Rewriting MLIR

[TOC]

This document details the design and API of the general IR rewriting 
infrastructure present in MLIR. There are specific implemenations
of this infrastructure, for example [Pattern Rewriting](PatternRewriter.md),
which is widely used throughout MLIR for canonicalization, conversion, and
general transformation.

## Rewriters

### `RewriterBase`

All rewriters extend from [`RewriterBase`](https://mlir.llvm.org/doxygen/classmlir_1_1RewriterBase.html), which inherits from [`OpBuilder`](https://mlir.llvm.org/doxygen/classmlir_1_1OpBuilder.html).
`RewriterBase` provides common API functions for any general types of rewriting
IR. It additionally provides a [`Listener`](#listeners) mechanism to keep track of IR
modifications.

#### Rewriter Implementations

Currently, there are only two implementations of `RewriterBase`:
[`PatternRewriter`](https://mlir.llvm.org/doxygen/classmlir_1_1PatternRewriter.html) and [`IRRewriter`](https://mlir.llvm.org/doxygen/classmlir_1_1IRRewriter.html).
The `RewriterBase` class is more designed towards being a base class of
`PatternRewriter`, which is described in more detail in [Pattern Rewriting](PatternRewriter.md/#pattern-rewriter).
However, the `IRRewriter` class is provided as a thin wrapper to use when
`PatternRewriter` is not available for use.

#### Rewriter Functions

Here is a list of commonly used functions that are provided to all 
rewriter implementations. For a complete list, see the
[list of public member functions in the class documentation](https://mlir.llvm.org/doxygen/classmlir_1_1RewriterBase.html).

*   Erase an Operation : `eraseOp`

This method erases an operation that either has no results, or whose results are
all known to have no uses.

*   Replace an Operation : `replaceOp`/`replaceOpWithNewOp`

This method replaces an operation's results with a set of provided values, and
erases the operation.

*   Replace all uses of a Value(s) or Operation : `replaceAllUsesWith`/`replaceAllOpUsesWith`

This method replaces a value, values, or an operation's results with a set of
provided values, but does NOT erase the operation.

*   Notify why a `match` failed : `notifyMatchFailure`

This method allows for providing a diagnostic message within a `matchAndRewrite`
as to why a pattern failed to match. How this message is displayed back to the
user is determined by the specific pattern driver.

### Listeners

The `RewriteBase::Listener` struct extends the [`OpBuilder::Listener`](https://mlir.llvm.org/doxygen/structmlir_1_1OpBuilder_1_1Listener.html)
class to add additional functions to handle the modification and erasure of IR.
These functions are called to notify the listener of IR changes.

#### Listener Functions

Here is a list of commonly used functions that are provided to all 
listener implementations. For a complete list, see the
[list of public member functions in the struct documentation](https://mlir.llvm.org/doxygen/structmlir_1_1RewriterBase_1_1Listener.html).

Since the listener functions are specialized to a specific usage, one
insertion, modification, or deletion _may_ lead to multipler listener functions
being called.
These listener notification functions map closely (almost 1:1) to the rewriter
replacement functions.

*   Inserting/Moving Ops/Blocks: `notifyOperationInserted`/`notifyBlockInserted`

These are provided from the parent `OpBuilder::Listener` struct, and provide
the op/block and the previous location of the op/block (if present).

*   Erasing Ops/Blocks: `notifyOperationErased`/`notifyBlockErased`

When this is called, the op/block already has zero uses. This is not called
when an op is unlinked.

*   Replacing Ops: `notifyOperationReplaced`

When an op is replaced with a single op or a range of values, one of the
overloaded versions of this function is called.

*   Notification of pattern application: `notifyPatternBegin`/`notifyPatternEnd`

These specify the pattern and either the root op to apply on or the 
success/failure status of the pattern application.

#### Listener Implementations

Currently, there are two implementations of `RewriterBase::Listener`:
[`RewriterBase::ForwardingListener`](https://mlir.llvm.org/doxygen/structmlir_1_1RewriterBase_1_1ForwardingListener.html)
and [`RewriterBase::MatchFailureEmittingListener`](https://mlir.llvm.org/doxygen/structmlir_1_1RewriterBase_1_1MatchFailureEmittingListener.html).

`ForwardingListener` can be extended to forward one notification to multiple listeners.
For an example, see [`NewOpsListener`](https://github.com/llvm/llvm-project/blob/0310f7f2d0c56a5697710251cec9803cbf7b4d56/mlir/lib/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp#L281-L287) in `LinalgTransformOps.cpp` or [`ExpensiveChecks`](https://github.com/llvm/llvm-project/blob/0310f7f2d0c56a5697710251cec9803cbf7b4d56/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L55-L57) in `GreedyPatternRewriteDriver.cpp`.

`MatchFailureEmittingListener` can be used to emit the `Diagnostic` from
`notifyMatchFailure` calls with a given [`DiagnosticSeverity`](https://llvm.org/doxygen/namespacellvm.html#abfcab32516704f11d146c757f402ad5c).
By default, the severity is `Error`, and the location associated with the
diagnostic will be the location provided to `notifyMatchFailure`.

#### Using Listeners

`OpBuilder` (from which `RewriterBase` extends) provides the `setListener`
function to set the listener. However, some frameworks do not directly
expose the rewriter before starting rewriting, so they usually provide a
config struct which can be used to set the listener.

For example, here is an example from `TransformOps.cpp` which uses the dialect
conversion framework:

```cpp
// Attach a tracking listener if handles should be preserved. We configure the
// listener to allow op replacements with different names, as conversion
// patterns typically replace ops with replacement ops that have a different
// name.
TrackingListenerConfig trackingConfig;
trackingConfig.requireMatchingReplacementOpName = false;
ErrorCheckingTrackingListener trackingListener(state, *this, trackingConfig);
ConversionConfig conversionConfig;
if (getPreserveHandles())
  conversionConfig.listener = &trackingListener;
```