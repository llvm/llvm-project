# Debug Actions

[TOC]

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

(TODO: Add connection to existing handlers when they are added)

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
action related queries within the [`DebugActionManager`](debug-action-manager).
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
