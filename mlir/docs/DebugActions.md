# Debug Actions

This file documents the infrastructure for `Debug Actions`. This is a DEBUG only
API that allows for external entities to control various aspects of compiler
execution. This is conceptually similar to something like `DebugCounters` in
LLVM, but at a lower level. This framework doesn't make any assumptions about
how the higher level driver is controlling the execution, it merely provides a
framework for connecting the two together. A high level overview of the workflow
surrounding debug actions is shown below:

*   Compiler developer defines an [`action`](#debug-action) that is taken by the
    a pass, transformation, utility that they are developing.
*   Depending on the needs, the developer dispatches various queries, pertaining
    to this action, to an [`action manager`](#debug-action-manager) that will
    provide an answer as to what behavior the action should take.
*   An external entity registers an [`action handler`](#debug-action-handler)
    with the action manager, and provides the logic to resolve queries on
    actions.

The exact definition of an `external entity` is left opaque, to allow for more
interesting handlers. The set of possible action queries is detailed in the
[`action manager`](#debug-action-manager) section below.

[TOC]

## Debug Action

A `debug action` is essentially a marker for a type of action that may be
performed within the compiler. There are no constraints on the granularity of an
“action”, it can be as simple as “perform this fold” and as complex as “run this
pass pipeline”. An action is comprised of the following:

*   Tag:

    -   A unique string identifier, similar to a command line flag or
        DEBUG_TYPE.

*   Description:

    -   A short description of what the action represents.

*   Parameter Types:

    -   The types of values that are passed to queries related to this action,
        to help guide decisions.

Below is an example action that may be provided by the
[pattern rewriter](PatternRewriter.md) framework to control the application of
rewrite patterns.

```c++
/// A debug action that allows for controlling the application of patterns.
/// A new action type can be defined by inheriting from `DebugAction`.
/// * The Tag is specified via a static `StringRef getTag()` method.
/// * The Description is specified via a static `StringRef getDescription()`
///   method.
/// * The parameters for the action are provided via template parameters when
///   inheriting from `DebugAction`.
struct ApplyPatternAction
    : public DebugAction<Operation *, const Pattern &> {
  static StringRef getTag() { return "apply-pattern"; }
  static StringRef getDescription() {
    return "Control the application of rewrite patterns";
  }
};
```

## Debug Action Manager

The `DebugActionManager` orchestrates the various different queries relating to
debug actions, and is accessible via the `MLIRContext`. These queries allow for
external entities to control various aspects of the compiler via
[action handlers](#debug-action-handler). When resolving a query for an action,
the result from the most recently registered handler is used.

TODO: It may be interesting to support merging results from multiple action
handlers, but this is left for future work when driven by a real use case.

The set of available queries are shown below:

```c++
class DebugActionManager {
public:
  /// Returns true if the given action type should be executed, false otherwise.
  /// `Params` correspond to any action specific parameters that may be used to
  /// guide the decision.
  template <typename ActionType, typename... Params>
  bool shouldExecute(Params &&... params);
};
```

Building on the example from the [previous section](#debug-action), an example
usage of the `shouldExecute` query is shown below:

```c++
/// A debug action that allows for controlling the application of patterns.
struct ApplyPatternAction
    : public DebugAction<Operation *, const Pattern &> {
  static StringRef getTag() { return "apply-pattern"; }
  static StringRef getDescription() {
    return "Control the application of rewrite patterns";
  }
};

// ...

bool shouldApplyPattern(Operation *currentOp, const Pattern &currentPattern) {
  MLIRContext *context = currentOp->getContext();
  DebugActionManager &manager = context->getDebugActionManager();

  // Query the action manager to see if `currentPattern` should be applied to
  // `currentOp`.
  return manager.shouldExecute<ApplyPatternAction>(currentOp, currentPattern);
}
```

## Debug Action Handler

A debug action handler provides the internal implementation for the various
action related queries within the [`DebugActionManager`](#debug-action-manager).
Action handlers allow for external entities to control and inject external
information into the compiler. Handlers can be registered with the
`DebugActionManager` using `registerActionHandler`. There are two types of
handlers; action-specific handlers and generic handlers.

### Action Specific Handlers

Action specific handlers handle a specific debug action type, with the
parameters to its query methods mapping 1-1 to the parameter types of the action
type. An action specific handler can be defined by inheriting from the handler
base class defined at `ActionType::Handler` where `ActionType` is the specific
action that should be handled. An example using our running pattern example is
shown below:

```c++
struct MyPatternHandler : public ApplyPatternAction::Handler {
  /// A variant of `shouldExecute` shown in the `DebugActionManager` class
  /// above.
  /// This method returns a FailureOr<bool>, where failure signifies that the
  /// action was not handled (allowing for other handlers to process it), or the
  /// boolean true/false signifying if the action should execute or not.
  FailureOr<bool> shouldExecute(Operation *op,
                                const RewritePattern &pattern) final;
};
```

### Generic Handlers

A generic handler allows for handling queries on any action type. These types of
handlers are useful for implementing general functionality that doesn’t
necessarily need to interpret the exact action parameters, or can rely on an
external interpreter (such as the user). As these handlers are generic, they
take a set of opaque parameters that try to map the context of the action type
in a generic way. A generic handler can be defined by inheriting from
`DebugActionManager::GenericHandler`. An example is shown below:

```c++
struct MyPatternHandler : public DebugActionManager::GenericHandler {
  /// The return type of this method is the same as the action-specific handler.
  /// The arguments to this method map the concepts of an action type in an
  /// opaque way. The arguments are provided in such a way so that the context
  /// of the action is still somewhat user readable, or at least loggable as
  /// such.
  /// - actionTag: The tag specified by the action type.
  /// - actionDesc: The description specified by the action type.
  virtual FailureOr<bool> shouldExecute(StringRef actionTag,
                                        StringRef actionDesc);
};
```

### Common Action Handlers

MLIR provides several common debug action handlers for immediate use that have
proven useful in general.

#### DebugCounter

When debugging a compiler issue,
["bisection"](https://en.wikipedia.org/wiki/Bisection_\(software_engineering\))
is a useful technique for locating the root cause of the issue. `Debug Counters`
enable using this technique for debug actions by attaching a counter value to a
specific debug action and enabling/disabling execution of this action based on
the value of the counter. The counter controls the execution of the action with
a "skip" and "count" value. The "skip" value is used to skip a certain number of
initial executions of a debug action. The "count" value is used to prevent a
debug action from executing after it has executed for a set number of times (not
including any executions that have been skipped). If the "skip" value is
negative, the action will always execute. If the "count" value is negative, the
action will always execute after the "skip" value has been reached. For example,
a counter for a debug action with `skip=47` and `count=2`, would skip the first
47 executions, then execute twice, and finally prevent any further executions.
With a bit of tooling, the values to use for the counter can be automatically
selected; allowing for finding the exact execution of a debug action that
potentially causes the bug being investigated.

Note: The DebugCounter action handler does not support multi-threaded execution,
and should only be used in MLIRContexts where multi-threading is disabled (e.g.
via `-mlir-disable-threading`).

##### CommandLine Configuration

The `DebugCounter` handler provides several that allow for configuring counters.
The main option is `mlir-debug-counter`, which accepts a comma separated list of
`<count-name>=<counter-value>`. A `<counter-name>` is the debug action tag to
attach the counter, suffixed with either `-skip` or `-count`. A `-skip` suffix
will set the "skip" value of the counter. A `-count` suffix will set the "count"
value of the counter. The `<counter-value>` component is a numeric value to use
for the counter. An example is shown below using `ApplyPatternAction` defined
above:

```shell
$ mlir-opt foo.mlir -mlir-debug-counter=apply-pattern-skip=47,apply-pattern-count=2
```

The above configuration would skip the first 47 executions of
`ApplyPatternAction`, then execute twice, and finally prevent any further
executions.

Note: Each counter currently only has one `skip` and one `count` value, meaning
that sequences of `skip`/`count` will not be chained.

The `mlir-print-debug-counter` option may be used to print out debug counter
information after all counters have been accumulated. The information is printed
in the following format:

```shell
DebugCounter counters:
<action-tag>                   : {<current-count>,<skip>,<count>}
```

For example, using the options above we can see how many times an action is
executed:

```shell
$ mlir-opt foo.mlir -mlir-debug-counter=apply-pattern-skip=-1 -mlir-print-debug-counter

DebugCounter counters:
apply-pattern                   : {370,-1,-1}
```
