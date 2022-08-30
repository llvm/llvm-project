# Pass Infrastructure

[TOC]

Passes represent the basic infrastructure for transformation and optimization.
This document provides an overview of the pass infrastructure in MLIR and how to
use it.

See [MLIR specification](LangRef.md) for more information about MLIR and its
core aspects, such as the IR structure and operations.

See [MLIR Rewrites](Tutorials/QuickstartRewrites.md) for a quick start on graph
rewriting in MLIR. If a transformation involves pattern matching operation DAGs,
this is a great place to start.

## Operation Pass

In MLIR, the main unit of abstraction and transformation is an
[operation](LangRef.md/#operations). As such, the pass manager is designed to
work on instances of operations at different levels of nesting. The structure of
the [pass manager](#pass-manager), and the concept of nesting, is detailed
further below. All passes in MLIR derive from `OperationPass` and adhere to the
following restrictions; any noncompliance will lead to problematic behavior in
multithreaded and other advanced scenarios:

*   Must not modify any state referenced or relied upon outside the current
    operation being operated on. This includes adding or removing operations
    from the parent block, changing the attributes(depending on the contract
    of the current operation)/operands/results/successors of the current operation.
*   Must not modify the state of another operation not nested within the current
    operation being operated on.
    *   Other threads may be operating on these operations simultaneously.
*   Must not inspect the state of sibling operations.
    *   Other threads may be modifying these operations in parallel.
    *   Inspecting the state of ancestor/parent operations is permitted.
*   Must not maintain mutable pass state across invocations of `runOnOperation`.
    A pass may be run on many different operations with no guarantee of
    execution order.
    *   When multithreading, a specific pass instance may not even execute on
        all operations within the IR. As such, a pass should not rely on running
        on all operations.
*   Must not maintain any global mutable state, e.g. static variables within the
    source file. All mutable state should be maintained by an instance of the
    pass.
*   Must be copy-constructible
    *   Multiple instances of the pass may be created by the pass manager to
        process operations in parallel.

### Op-Agnostic Operation Passes

By default, an operation pass is `op-agnostic`, meaning that it operates on the
operation type of the pass manager that it is added to. This means a pass may operate
on many different types of operations. Agnostic passes should be written such that
they do not make assumptions on the operation they run on. Examples of this type of pass are
[Canonicalization](Pass.md/-canonicalize-canonicalize-operations)
[Common Sub-Expression Elimination](Passes.md/#-cse-eliminate-common-sub-expressions).

To create an agnostic operation pass, a derived class must adhere to the following:

*   Inherit from the CRTP class `OperationPass`.
*   Override the virtual `void runOnOperation()` method.

A simple pass may look like:

```c++
/// Here we utilize the CRTP `PassWrapper` utility class to provide some
/// necessary utility hooks. This is only necessary for passes defined directly
/// in C++. Passes defined declaratively use a cleaner mechanism for providing
/// these utilities.
struct MyOperationPass : public PassWrapper<MyOperationPass, OperationPass<>> {
  void runOnOperation() override {
    // Get the current operation being operated on.
    Operation *op = getOperation();
    ...
  }
};
```

### Filtered Operation Pass

If a pass needs to constrain its execution to specific types or classes of operations,
additional filtering may be applied on top. This transforms a once `agnostic` pass into
one more specific to a certain context. There are various ways in which to filter the
execution of a pass, and different contexts in which filtering may apply:

### Operation Pass: Static Schedule Filtering

Static filtering allows for applying additional constraints on the operation types a
pass may be scheduled on. This type of filtering generally allows for building more
constrained passes that can only be scheduled on operations that satisfy the necessary
constraints. For example, this allows for specifying passes that only run on operations
of a certain, those that provide a certain interface, trait, or some other constraint that
applies to all instances of that operation type. Below is an example of a pass that only
permits scheduling on operations that implement `FunctionOpInterface`:

```c++
struct MyFunctionPass : ... {
  /// This method is used to provide additional static filtering, and returns if the
  /// pass may be scheduled on the given operation type.
  bool canScheduleOn(RegisteredOperationName opInfo) const override {
    return opInfo.hasInterface<FunctionOpInterface>();
  }

  void runOnOperation() {
    // Here we can freely cast to FunctionOpInterface, because our `canScheduleOn` ensures
    // that our pass is only executed on operations implementing that interface.
    FunctionOpInterface op = cast<FunctionOpInterface>(getOperation()); 
  }
};
```

When a pass with static filtering is added to an [`op-specific` pass manager](#oppassmanager),
it asserts that the operation type of the pass manager satisfies the static constraints of the
pass. When added to an [`op-agnostic` pass manager](#oppassmanager), that pass manager, and all
passes contained within, inherits the static constraints of the pass. For example, if the pass
filters on `FunctionOpInterface`, as in the `MyFunctionPass` example above, only operations that
implement `FunctionOpInterface` will be considered when executing **any** passes within the pass
manager. This invariant is important to keep in mind, as each pass added to an `op-agnostic` pass
manager further constrains the operations that may be scheduled on it. Consider the following example:

```mlir
func.func @foo() {
  // ...
  return
}

module @someModule {
  // ...
}
```

If we were to apply the op-agnostic pipeline, `any(cse,my-function-pass)`, to the above MLIR snippet
it would only run on the `foo` function operation. This is because the `my-function-pass` has a
static filtering constraint to only schedule on operations implementing `FunctionOpInterface`. Remember
that this constraint is inherited by the entire pass manager, so we never consider `someModule` for
any of the passes, including `cse` which normally can be scheduled on any operation.

#### Operation Pass: Static Filtering By Op Type

In the above section, we detailed a general mechanism for statically filtering the types of operations
that a pass may be scheduled on. Sugar is provided on top of that mechanism to simplify the definition
of passes that are restricted to scheduling on a single operation type. In these cases, a pass simply
needs to provide the type of operation to the `OperationPass` base class. This will automatically
instill filtering on that operation type:

```c++
/// Here we utilize the CRTP `PassWrapper` utility class to provide some
/// necessary utility hooks. This is only necessary for passes defined directly
/// in C++. Passes defined declaratively use a cleaner mechanism for providing
/// these utilities.
struct MyFunctionPass : public PassWrapper<MyOperationPass, OperationPass<func::FuncOp>> {
  void runOnOperation() {
    // Get the current operation being operated on.
    func::FuncOp op = getOperation();
  }
};
```

#### Operation Pass: Static Filtering By Interface

In the above section, we detailed a general mechanism for statically filtering the types of operations
that a pass may be scheduled on. Sugar is provided on top of that mechanism to simplify the definition
of passes that are restricted to scheduling on a specific operation interface. In these cases, a pass
simply needs to inherit from the `InterfacePass` base class. This class is similar to `OperationPass`,
but expects the type of interface to operate on. This will automatically instill filtering on that
interface type:

```c++
/// Here we utilize the CRTP `PassWrapper` utility class to provide some
/// necessary utility hooks. This is only necessary for passes defined directly
/// in C++. Passes defined declaratively use a cleaner mechanism for providing
/// these utilities.
struct MyFunctionPass : public PassWrapper<MyOperationPass, InterfacePass<FunctionOpInterface>> {
  void runOnOperation() {
    // Get the current operation being operated on.
    FunctionOpInterface op = getOperation();
  }
};
```

### Dependent Dialects

Dialects must be loaded in the MLIRContext before entities from these dialects
(operations, types, attributes, ...) can be created. Dialects must also be
loaded before starting the execution of a multi-threaded pass pipeline. To this
end, a pass that may create an entity from a dialect that isn't guaranteed to
already be loaded must express this by overriding the `getDependentDialects()`
method and declare this list of Dialects explicitly.
See also the `dependentDialects` field in the
[TableGen Specification](#tablegen-specification).

### Initialization

In certain situations, a Pass may contain state that is constructed dynamically,
but is potentially expensive to recompute in successive runs of the Pass. One
such example is when using [`PDL`-based](Dialects/PDLOps.md)
[patterns](PatternRewriter.md), which are compiled into a bytecode during
runtime. In these situations, a pass may override the following hook to
initialize this heavy state:

*   `LogicalResult initialize(MLIRContext *context)`

This hook is executed once per run of a full pass pipeline, meaning that it does
not have access to the state available during a `runOnOperation` call. More
concretely, all necessary accesses to an `MLIRContext` should be driven via the
provided `context` parameter, and methods that utilize "per-run" state such as
`getContext`/`getOperation`/`getAnalysis`/etc. must not be used.
In case of an error during initialization, the pass is expected to emit an error
diagnostic and return a `failure()` which will abort the pass pipeline execution.

## Analysis Management

An important concept, along with transformation passes, are analyses. These are
conceptually similar to transformation passes, except that they compute
information on a specific operation without modifying it. In MLIR, analyses are
not passes but free-standing classes that are computed lazily on-demand and
cached to avoid unnecessary recomputation. An analysis in MLIR must adhere to
the following:

*   Provide a valid constructor taking either an `Operation*` or `Operation*`
    and `AnalysisManager &`.
    *   The provided `AnalysisManager &` should be used to query any necessary
        analysis dependencies.
*   Must not modify the given operation.

An analysis may provide additional hooks to control various behavior:

*   `bool isInvalidated(const AnalysisManager::PreservedAnalyses &)`

Given a preserved analysis set, the analysis returns true if it should truly be
invalidated. This allows for more fine-tuned invalidation in cases where an
analysis wasn't explicitly marked preserved, but may be preserved (or
invalidated) based upon other properties such as analyses sets. If the analysis
uses any other analysis as a dependency, it must also check if the dependency
was invalidated.

### Querying Analyses

The base `OperationPass` class provides utilities for querying and preserving
analyses for the current operation being processed.

*   OperationPass automatically provides the following utilities for querying
    analyses:
    *   `getAnalysis<>`
        -   Get an analysis for the current operation, constructing it if
            necessary.
    *   `getCachedAnalysis<>`
        -   Get an analysis for the current operation, if it already exists.
    *   `getCachedParentAnalysis<>`
        -   Get an analysis for a given parent operation, if it exists.
    *   `getCachedChildAnalysis<>`
        -   Get an analysis for a given child operation, if it exists.
    *   `getChildAnalysis<>`
        -   Get an analysis for a given child operation, constructing it if
            necessary.

Using the example passes defined above, let's see some examples:

```c++
/// An interesting analysis.
struct MyOperationAnalysis {
  // Compute this analysis with the provided operation.
  MyOperationAnalysis(Operation *op);
};

struct MyOperationAnalysisWithDependency {
  MyOperationAnalysisWithDependency(Operation *op, AnalysisManager &am) {
    // Request other analysis as dependency
    MyOperationAnalysis &otherAnalysis = am.getAnalysis<MyOperationAnalysis>();
    ...
  }

  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa) {
    // Check if analysis or its dependency were invalidated
    return !pa.isPreserved<MyOperationAnalysisWithDependency>() ||
           !pa.isPreserved<MyOperationAnalysis>();
  }
};

void MyOperationPass::runOnOperation() {
  // Query MyOperationAnalysis for the current operation.
  MyOperationAnalysis &myAnalysis = getAnalysis<MyOperationAnalysis>();

  // Query a cached instance of MyOperationAnalysis for the current operation.
  // It will not be computed if it doesn't exist.
  auto optionalAnalysis = getCachedAnalysis<MyOperationAnalysis>();
  if (optionalAnalysis)
    ...

  // Query a cached instance of MyOperationAnalysis for the parent operation of
  // the current operation. It will not be computed if it doesn't exist.
  auto optionalAnalysis = getCachedParentAnalysis<MyOperationAnalysis>();
  if (optionalAnalysis)
    ...
}
```

### Preserving Analyses

Analyses that are constructed after being queried by a pass are cached to avoid
unnecessary computation if they are requested again later. To avoid stale
analyses, all analyses are assumed to be invalidated by a pass. To avoid
invalidation, a pass must specifically mark analyses that are known to be
preserved.

*   All Pass classes automatically provide the following utilities for
    preserving analyses:
    *   `markAllAnalysesPreserved`
    *   `markAnalysesPreserved<>`

```c++
void MyOperationPass::runOnOperation() {
  // Mark all analyses as preserved. This is useful if a pass can guarantee
  // that no transformation was performed.
  markAllAnalysesPreserved();

  // Mark specific analyses as preserved. This is used if some transformation
  // was performed, but some analyses were either unaffected or explicitly
  // preserved.
  markAnalysesPreserved<MyAnalysis, MyAnalyses...>();
}
```

## Pass Failure

Passes in MLIR are allowed to gracefully fail. This may happen if some invariant
of the pass was broken, potentially leaving the IR in some invalid state. If
such a situation occurs, the pass can directly signal a failure to the pass
manager via the `signalPassFailure` method. If a pass signaled a failure when
executing, no other passes in the pipeline will execute and the top-level call
to `PassManager::run` will return `failure`.

```c++
void MyOperationPass::runOnOperation() {
  // Signal failure on a broken invariant.
  if (some_broken_invariant)
    return signalPassFailure();
}
```

## Pass Manager

The above sections introduced the different types of passes and their
invariants. This section introduces the concept of a PassManager, and how it can
be used to configure and schedule a pass pipeline. There are two main classes
related to pass management, the `PassManager` and the `OpPassManager`. The
`PassManager` class acts as the top-level entry point, and contains various
configurations used for the entire pass pipeline. The `OpPassManager` class is
used to schedule passes to run at a specific level of nesting. The top-level
`PassManager` also functions as an `OpPassManager`.

### OpPassManager

An `OpPassManager` is essentially a collection of passes anchored to execute on
operations at a given level of nesting. A pass manager may be `op-specific`
(anchored on a specific operation type), or `op-agnostic` (not restricted to any
specific operation, and executed on any viable operation type). Operation types that
anchor pass managers must adhere to the following requirement:

*   Must be registered and marked
    [`IsolatedFromAbove`](Traits.md/#isolatedfromabove).

    *   Passes are expected not to modify operations at or above the current
        operation being processed. If the operation is not isolated, it may
        inadvertently modify or traverse the SSA use-list of an operation it is
        not supposed to.

Passes can be added to a pass manager via `addPass`.

An `OpPassManager` is generally created by explicitly nesting a pipeline within
another existing `OpPassManager` via the `nest<OpT>` or `nestAny` methods. The
former method takes the operation type that the nested pass manager will operate on.
The latter method nests an `op-agnostic` pass manager, that may run on any viable
operation type. Nesting in this sense, corresponds to the
[structural](Tutorials/UnderstandingTheIRStructure.md) nesting within
[Regions](LangRef.md/#regions) of the IR.

For example, the following `.mlir`:

```mlir
module {
  spv.module "Logical" "GLSL450" {
    func @foo() {
      ...
    }
  }
}
```

Has the nesting structure of:

```
`builtin.module`
  `spv.module`
    `spv.func`
```

Below is an example of constructing a pipeline that operates on the above
structure:

```c++
// Create a top-level `PassManager` class. If an operation type is not
// explicitly specific, the default is the builtin `module` operation.
PassManager pm(ctx);
// Note: We could also create the above `PassManager` this way.
PassManager pm(ctx, /*operationName=*/"builtin.module");

// Add a pass on the top-level module operation.
pm.addPass(std::make_unique<MyModulePass>());

// Nest a pass manager that operates on `spirv.module` operations nested
// directly under the top-level module.
OpPassManager &nestedModulePM = pm.nest<spirv::ModuleOp>();
nestedModulePM.addPass(std::make_unique<MySPIRVModulePass>());

// Nest a pass manager that operates on functions within the nested SPIRV
// module.
OpPassManager &nestedFunctionPM = nestedModulePM.nest<func::FuncOp>();
nestedFunctionPM.addPass(std::make_unique<MyFunctionPass>());

// Nest an op-agnostic pass manager. This will operate on any viable
// operation, e.g. func.func, spv.func, spv.module, builtin.module, etc.
OpPassManager &nestedAnyPM = nestedModulePM.nestAny();
nestedAnyPM.addPass(createCanonicalizePass());
nestedAnyPM.addPass(createCSEPass());

// Run the pass manager on the top-level module.
ModuleOp m = ...;
if (failed(pm.run(m)))
    ... // One of the passes signaled a failure.
```

The above pass manager contains the following pipeline structure:

```
OpPassManager<ModuleOp>
  MyModulePass
  OpPassManager<spirv::ModuleOp>
    MySPIRVModulePass
    OpPassManager<func::FuncOp>
      MyFunctionPass
    OpPassManager<>
      Canonicalizer
      CSE
```

These pipelines are then run over a single operation at a time. This means that,
for example, given a series of consecutive passes on func::FuncOp, it will execute all
on the first function, then all on the second function, etc. until the entire
program has been run through the passes. This provides several benefits:

*   This improves the cache behavior of the compiler, because it is only
    touching a single function at a time, instead of traversing the entire
    program.
*   This improves multi-threading performance by reducing the number of jobs
    that need to be scheduled, as well as increasing the efficiency of each job.
    An entire function pipeline can be run on each function asynchronously.

## Dynamic Pass Pipelines

In some situations it may be useful to run a pass pipeline within another pass,
to allow configuring or filtering based on some invariants of the current
operation being operated on. For example, the
[Inliner Pass](Passes.md/#-inline-inline-function-calls) may want to run
intraprocedural simplification passes while it is inlining to produce a better
cost model, and provide more optimal inlining. To enable this, passes may run an
arbitrary `OpPassManager` on the current operation being operated on or any
operation nested within the current operation via the `LogicalResult
Pass::runPipeline(OpPassManager &, Operation *)` method. This method returns
whether the dynamic pipeline succeeded or failed, similarly to the result of the
top-level `PassManager::run` method. A simple example is shown below:

```c++
void MyModulePass::runOnOperation() {
  ModuleOp module = getOperation();
  if (hasSomeSpecificProperty(module)) {
    OpPassManager dynamicPM("builtin.module");
    ...; // Build the dynamic pipeline.
    if (failed(runPipeline(dynamicPM, module)))
      return signalPassFailure();
  }
}
```

Note: though above the dynamic pipeline was constructed within the
`runOnOperation` method, this is not necessary and pipelines should be cached
when possible as the `OpPassManager` class can be safely copy constructed.

The mechanism described in this section should be used whenever a pass pipeline
should run in a nested fashion, i.e. when the nested pipeline cannot be
scheduled statically along with the rest of the main pass pipeline. More
specifically, a `PassManager` should generally never need to be constructed
within a `Pass`. Using `runPipeline` also ensures that all analyses,
[instrumentations](#pass-instrumentation), and other pass manager related
components are integrated with the dynamic pipeline being executed.

## Instance Specific Pass Options

MLIR provides a builtin mechanism for passes to specify options that configure
its behavior. These options are parsed at pass construction time independently
for each instance of the pass. Options are defined using the `Option<>` and
`ListOption<>` classes, and generally follow the
[LLVM command line](https://llvm.org/docs/CommandLine.html) flag definition
rules. One major distinction from the LLVM command line functionality is that
all `ListOption`s are comma-separated, and delimited sub-ranges within individual
elements of the list may contain commas that are not treated as separators for the
top-level list.

```c++
struct MyPass ... {
  /// Make sure that we have a valid default constructor and copy constructor to
  /// ensure that the options are initialized properly.
  MyPass() = default;
  MyPass(const MyPass& pass) {}

  /// Any parameters after the description are forwarded to llvm::cl::list and
  /// llvm::cl::opt respectively.
  Option<int> exampleOption{*this, "flag-name", llvm::cl::desc("...")};
  ListOption<int> exampleListOption{*this, "list-flag-name", llvm::cl::desc("...")};
};
```

For pass pipelines, the `PassPipelineRegistration` templates take an additional
template parameter for an optional `Option` struct definition. This struct
should inherit from `mlir::PassPipelineOptions` and contain the desired pipeline
options. When using `PassPipelineRegistration`, the constructor now takes a
function with the signature `void (OpPassManager &pm, const MyPipelineOptions&)`
which should construct the passes from the options and pass them to the pm:

```c++
struct MyPipelineOptions : public PassPipelineOptions {
  // The structure of these options is the same as those for pass options.
  Option<int> exampleOption{*this, "flag-name", llvm::cl::desc("...")};
  ListOption<int> exampleListOption{*this, "list-flag-name",
                                    llvm::cl::desc("...")};
};

void registerMyPasses() {
  PassPipelineRegistration<MyPipelineOptions>(
    "example-pipeline", "Run an example pipeline.",
    [](OpPassManager &pm, const MyPipelineOptions &pipelineOptions) {
      // Initialize the pass manager.
    });
}
```

## Pass Statistics

Statistics are a way to keep track of what the compiler is doing and how
effective various transformations are. It is often useful to see what effect
specific transformations have on a particular input, and how often they trigger.
Pass statistics are specific to each pass instance, which allow for seeing the
effect of placing a particular transformation at specific places within the pass
pipeline. For example, they help answer questions like "What happens if I run
CSE again here?".

Statistics can be added to a pass by using the 'Pass::Statistic' class. This
class takes as a constructor arguments: the parent pass, a name, and a
description. This class acts like an atomic unsigned integer, and may be
incremented and updated accordingly. These statistics rely on the same
infrastructure as
[`llvm::Statistic`](http://llvm.org/docs/ProgrammersManual.html#the-statistic-class-stats-option)
and thus have similar usage constraints. Collected statistics can be dumped by
the [pass manager](#pass-manager) programmatically via
`PassManager::enableStatistics`; or via `-mlir-pass-statistics` and
`-mlir-pass-statistics-display` on the command line.

An example is shown below:

```c++
struct MyPass ... {
  /// Make sure that we have a valid default constructor and copy constructor to
  /// ensure that the options are initialized properly.
  MyPass() = default;
  MyPass(const MyPass& pass) {}
  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "argument";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return  "description";
  }
  /// Define the statistic to track during the execution of MyPass.
  Statistic exampleStat{this, "exampleStat", "An example statistic"};

  void runOnOperation() {
    ...

    // Update the statistic after some invariant was hit.
    ++exampleStat;

    ...
  }
};
```

The collected statistics may be aggregated in two types of views:

A pipeline view that models the structure of the pass manager, this is the
default view:

```shell
$ mlir-opt -pass-pipeline='func.func(my-pass,my-pass)' foo.mlir -mlir-pass-statistics

===-------------------------------------------------------------------------===
                         ... Pass statistics report ...
===-------------------------------------------------------------------------===
'func.func' Pipeline
  MyPass
    (S) 15 exampleStat - An example statistic
  VerifierPass
  MyPass
    (S)  6 exampleStat - An example statistic
  VerifierPass
VerifierPass
```

A list view that aggregates the statistics of all instances of a specific pass
together:

```shell
$ mlir-opt -pass-pipeline='func.func(my-pass, my-pass)' foo.mlir -mlir-pass-statistics -mlir-pass-statistics-display=list

===-------------------------------------------------------------------------===
                         ... Pass statistics report ...
===-------------------------------------------------------------------------===
MyPass
  (S) 21 exampleStat - An example statistic
```

## Pass Registration

Briefly shown in the example definitions of the various pass types is the
`PassRegistration` class. This mechanism allows for registering pass classes so
that they may be created within a
[textual pass pipeline description](#textual-pass-pipeline-specification). An
example registration is shown below:

```c++
void registerMyPass() {
  PassRegistration<MyPass>();
}
```

*   `MyPass` is the name of the derived pass class.
*   The pass `getArgument()` method is used to get the identifier that will be
    used to refer to the pass.
*   The pass `getDescription()` method provides a short summary describing the
    pass.

For passes that cannot be default-constructed, `PassRegistration` accepts an
optional argument that takes a callback to create the pass:

```c++
void registerMyPass() {
  PassRegistration<MyParametricPass>(
    []() -> std::unique_ptr<Pass> {
      std::unique_ptr<Pass> p = std::make_unique<MyParametricPass>(/*options*/);
      /*... non-trivial-logic to configure the pass ...*/;
      return p;
    });
}
```

This variant of registration can be used, for example, to accept the
configuration of a pass from command-line arguments and pass it to the pass
constructor.

Note: Make sure that the pass is copy-constructible in a way that does not share
data as the [pass manager](#pass-manager) may create copies of the pass to run
in parallel.

### Pass Pipeline Registration

Described above is the mechanism used for registering a specific derived pass
class. On top of that, MLIR allows for registering custom pass pipelines in a
similar fashion. This allows for custom pipelines to be available to tools like
mlir-opt in the same way that passes are, which is useful for encapsulating
common pipelines like the "-O1" series of passes. Pipelines are registered via a
similar mechanism to passes in the form of `PassPipelineRegistration`. Compared
to `PassRegistration`, this class takes an additional parameter in the form of a
pipeline builder that modifies a provided `OpPassManager`.

```c++
void pipelineBuilder(OpPassManager &pm) {
  pm.addPass(std::make_unique<MyPass>());
  pm.addPass(std::make_unique<MyOtherPass>());
}

void registerMyPasses() {
  // Register an existing pipeline builder function.
  PassPipelineRegistration<>(
    "argument", "description", pipelineBuilder);

  // Register an inline pipeline builder.
  PassPipelineRegistration<>(
    "argument", "description", [](OpPassManager &pm) {
      pm.addPass(std::make_unique<MyPass>());
      pm.addPass(std::make_unique<MyOtherPass>());
    });
}
```

### Textual Pass Pipeline Specification

The previous sections detailed how to register passes and pass pipelines with a
specific argument and description. Once registered, these can be used to
configure a pass manager from a string description. This is especially useful
for tools like `mlir-opt`, that configure pass managers from the command line,
or as options to passes that utilize
[dynamic pass pipelines](#dynamic-pass-pipelines).

To support the ability to describe the full structure of pass pipelines, MLIR
supports a custom textual description of pass pipelines. The textual description
includes the nesting structure, the arguments of the passes and pass pipelines
to run, and any options for those passes and pipelines. A textual pipeline is
defined as a series of names, each of which may in itself recursively contain a
nested pipeline description. The syntax for this specification is as follows:

```ebnf
pipeline          ::= op-anchor `(` pipeline-element (`,` pipeline-element)* `)`
pipeline-element  ::= pipeline | (pass-name | pass-pipeline-name) options?
options           ::= '{' (key ('=' value)?)+ '}'
```

*   `op-anchor`
    *   This corresponds to the mnemonic name that anchors the execution of the
        pass manager. This is either the name of an operation to run passes on,
        e.g. `func.func` or `builtin.module`, or `any`, for op-agnostic pass
        managers that execute on any viable operation (i.e. any operation that
        can be used to anchor a pass manager).
*   `pass-name` | `pass-pipeline-name`
    *   This corresponds to the argument of a registered pass or pass pipeline,
        e.g. `cse` or `canonicalize`.
*   `options`
    *   Options are specific key value pairs representing options defined by a
        pass or pass pipeline, as described in the
        ["Instance Specific Pass Options"](#instance-specific-pass-options)
        section. See this section for an example usage in a textual pipeline.

For example, the following pipeline:

```shell
$ mlir-opt foo.mlir -cse -canonicalize -convert-func-to-llvm='use-bare-ptr-memref-call-conv=1'
```

Can also be specified as (via the `-pass-pipeline` flag):

```shell
# Anchor the cse and canonicalize passes on the `func.func` operation.
$ mlir-opt foo.mlir -pass-pipeline='func.func(cse,canonicalize),convert-func-to-llvm{use-bare-ptr-memref-call-conv=1}'

# Anchor the cse and canonicalize passes on "any" viable root operation.
$ mlir-opt foo.mlir -pass-pipeline='any(cse,canonicalize),convert-func-to-llvm{use-bare-ptr-memref-call-conv=1}'
```

In order to support round-tripping a pass to the textual representation using
`OpPassManager::printAsTextualPipeline(raw_ostream&)`, override `StringRef
Pass::getArgument()` to specify the argument used when registering a pass.

## Declarative Pass Specification

Some aspects of a Pass may be specified declaratively, in a form similar to
[operations](OpDefinitions.md). This specification simplifies several mechanisms
used when defining passes. It can be used for generating pass registration
calls, defining boilerplate pass utilities, and generating pass documentation.

Consider the following pass specified in C++:

```c++
struct MyPass : PassWrapper<MyPass, OperationPass<ModuleOp>> {
  MyPass() = default;
  MyPass(const MyPass &) {}

  ...

  // Specify any options.
  Option<bool> option{
      *this, "example-option",
      llvm::cl::desc("An example option"), llvm::cl::init(true)};
  ListOption<int64_t> listOption{
      *this, "example-list",
      llvm::cl::desc("An example list option")};

  // Specify any statistics.
  Statistic statistic{this, "example-statistic", "An example statistic"};
};

/// Expose this pass to the outside world.
std::unique_ptr<Pass> foo::createMyPass() {
  return std::make_unique<MyPass>();
}

/// Register this pass.
void foo::registerMyPass() {
  PassRegistration<MyPass>();
}
```

This pass may be specified declaratively as so:

```tablegen
def MyPass : Pass<"my-pass", "ModuleOp"> {
  let summary = "My Pass Summary";
  let description = [{
    Here we can now give a much larger description of `MyPass`, including all of
    its various constraints and behavior.
  }];

  // A constructor must be provided to specify how to create a default instance
  // of MyPass. It can be skipped for this specific example, because both the
  // constructor and the registration methods live in the same namespace.
  let constructor = "foo::createMyPass()";

  // Specify any options.
  let options = [
    Option<"option", "example-option", "bool", /*default=*/"true",
           "An example option">,
    ListOption<"listOption", "example-list", "int64_t",
               "An example list option">
  ];

  // Specify any statistics.
  let statistics = [
    Statistic<"statistic", "example-statistic", "An example statistic">
  ];
}
```

Using the `gen-pass-decls` generator, we can generate most of the boilerplate
above automatically. This generator takes as an input a `-name` parameter, that
provides a tag for the group of passes that are being generated. This generator
produces code with multiple purposes:

The first is to register the declared passes with the global registry. For
each pass, the generator produces a `registerPassName` where
`PassName` is the name of the definition specified in tablegen. It also
generates a `registerGroupPasses`, where `Group` is the tag provided via the
`-name` input parameter, that registers all of the passes present.

```c++
// Tablegen options: -gen-pass-decls -name="Example"

// Passes.h

namespace foo {
#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"
} // namespace foo

void registerMyPasses() {
  // Register all of the passes.
  foo::registerExamplePasses();
  
  // Or

  // Register `MyPass` specifically.
  foo::registerMyPass();
}
```

The second is to provide a way to configure the pass options. These classes are
named in the form of `MyPassOptions`, where `MyPass` is the name of the pass
definition in tablegen. The configurable parameters reflect the options
declared in the tablegen file. Differently from the registration hooks, these
classes can be enabled on a per-pass basis by defining the
`GEN_PASS_DECL_PASSNAME` macro, where `PASSNAME` is the uppercase version of
the name specified in tablegen.

```c++
// .h.inc

#ifdef GEN_PASS_DECL_MYPASS

struct MyPassOptions {
    bool option = true;
    ::llvm::ArrayRef<int64_t> listOption;
};

#undef GEN_PASS_DECL_MYPASS
#endif // GEN_PASS_DECL_MYPASS
```

If the `constructor` field has not been specified in the tablegen declaration,
then autogenerated file will also contain the declarations of the default
constructors.

```c++
// .h.inc

#ifdef GEN_PASS_DECL_MYPASS
...

std::unique_ptr<::mlir::Pass> createMyPass();
std::unique_ptr<::mlir::Pass> createMyPass(const MyPassOptions &options);

#undef GEN_PASS_DECL_MYPASS
#endif // GEN_PASS_DECL_MYPASS
```

The last purpose of this generator is to emit a base class for each of the
passes, containing most of the boiler plate related to pass definitions. These
classes are named in the form of `MyPassBase` and are declared inside the
`impl` namespace, where `MyPass` is the name of the pass definition in
tablegen. We can update the original C++ pass definition as so:

```c++
// MyPass.cpp

/// Include the generated base pass class definitions.
namespace foo {
#define GEN_PASS_DEF_MYPASS
#include "Passes.h.inc"
}

/// Define the main class as deriving from the generated base class.
struct MyPass : foo::impl::MyPassBase<MyPass> {
  using MyPassBase::MyPassBase;

  /// The definitions of the options and statistics are now generated within
  /// the base class, but are accessible in the same way.
};
```

Similarly to the previous cases, the definitions can be enabled on a per-pass
basis by defining the appropriate preprocessor `GEN_PASS_DEF_PASSNAME` macro,
with `PASSNAME` equal to the uppercase version of the name of the pass
definition in tablegen.
If the `constructor` field has not been specified in tablegen, then the default
constructors are also defined and expect the name of the actual pass class to
be equal to the name defined in tablegen.

Using the `gen-pass-doc` generator, markdown documentation for each of the
passes can be generated. See [Passes.md](Passes.md) for example output of real
MLIR passes.

### Tablegen Specification

The `Pass` class is used to begin a new pass definition. This class takes as an
argument the registry argument to attribute to the pass, as well as an optional
string corresponding to the operation type that the pass operates on. The class
contains the following fields:

*   `summary`
    -   A short one-line summary of the pass, used as the description when
        registering the pass.
*   `description`
    -   A longer, more detailed description of the pass. This is used when
        generating pass documentation.
*   `dependentDialects`
    -   A list of strings representing the `Dialect` classes this pass may
        introduce entities, Attributes/Operations/Types/etc., of.
*   `constructor`
    -   A code block used to create a default instance of the pass.
*   `options`
    -   A list of pass options used by the pass.
*   `statistics`
    -   A list of pass statistics used by the pass.

#### Options

Options may be specified via the `Option` and `ListOption` classes. The `Option`
class takes the following template parameters:

*   C++ variable name
    -   A name to use for the generated option variable.
*   argument
    -   The argument name of the option.
*   type
    -   The C++ type of the option.
*   default value
    -   The default option value.
*   description
    -   A one-line description of the option.
*   additional option flags
    -   A string containing any additional options necessary to construct the
        option.

```tablegen
def MyPass : Pass<"my-pass"> {
  let options = [
    Option<"option", "example-option", "bool", /*default=*/"true",
           "An example option">,
  ];
}
```

The `ListOption` class takes the following fields:

*   C++ variable name
    -   A name to use for the generated option variable.
*   argument
    -   The argument name of the option.
*   element type
    -   The C++ type of the list element.
*   description
    -   A one-line description of the option.
*   additional option flags
    -   A string containing any additional options necessary to construct the
        option.

```tablegen
def MyPass : Pass<"my-pass"> {
  let options = [
    ListOption<"listOption", "example-list", "int64_t",
               "An example list option">
  ];
}
```

#### Statistic

Statistics may be specified via the `Statistic`, which takes the following
template parameters:

*   C++ variable name
    -   A name to use for the generated statistic variable.
*   display name
    -   The name used when displaying the statistic.
*   description
    -   A one-line description of the statistic.

```tablegen
def MyPass : Pass<"my-pass"> {
  let statistics = [
    Statistic<"statistic", "example-statistic", "An example statistic">
  ];
}
```

## Pass Instrumentation

MLIR provides a customizable framework to instrument pass execution and analysis
computation, via the `PassInstrumentation` class. This class provides hooks into
the PassManager that observe various events:

*   `runBeforePipeline`
    *   This callback is run just before a pass pipeline, i.e. pass manager, is
        executed.
*   `runAfterPipeline`
    *   This callback is run right after a pass pipeline has been executed,
        successfully or not.
*   `runBeforePass`
    *   This callback is run just before a pass is executed.
*   `runAfterPass`
    *   This callback is run right after a pass has been successfully executed.
        If this hook is executed, `runAfterPassFailed` will *not* be.
*   `runAfterPassFailed`
    *   This callback is run right after a pass execution fails. If this hook is
        executed, `runAfterPass` will *not* be.
*   `runBeforeAnalysis`
    *   This callback is run just before an analysis is computed.
    *   If the analysis requested another analysis as a dependency, the
        `runBeforeAnalysis`/`runAfterAnalysis` pair for the dependency can be
        called from inside of the current `runBeforeAnalysis`/`runAfterAnalysis`
        pair.
*   `runAfterAnalysis`
    *   This callback is run right after an analysis is computed.

PassInstrumentation instances may be registered directly with a
[PassManager](#pass-manager) instance via the `addInstrumentation` method.
Instrumentations added to the PassManager are run in a stack like fashion, i.e.
the last instrumentation to execute a `runBefore*` hook will be the first to
execute the respective `runAfter*` hook. The hooks of a `PassInstrumentation`
class are guaranteed to be executed in a thread-safe fashion, so additional
synchronization is not necessary. Below in an example instrumentation that
counts the number of times the `DominanceInfo` analysis is computed:

```c++
struct DominanceCounterInstrumentation : public PassInstrumentation {
  /// The cumulative count of how many times dominance has been calculated.
  unsigned &count;

  DominanceCounterInstrumentation(unsigned &count) : count(count) {}
  void runAfterAnalysis(llvm::StringRef, TypeID id, Operation *) override {
    if (id == TypeID::get<DominanceInfo>())
      ++count;
  }
};

MLIRContext *ctx = ...;
PassManager pm(ctx);

// Add the instrumentation to the pass manager.
unsigned domInfoCount;
pm.addInstrumentation(
    std::make_unique<DominanceCounterInstrumentation>(domInfoCount));

// Run the pass manager on a module operation.
ModuleOp m = ...;
if (failed(pm.run(m)))
    ...

llvm::errs() << "DominanceInfo was computed " << domInfoCount << " times!\n";
```

### Standard Instrumentations

MLIR utilizes the pass instrumentation framework to provide a few useful
developer tools and utilities. Each of these instrumentations are directly
available to all users of the MLIR pass framework.

#### Pass Timing

The PassTiming instrumentation provides timing information about the execution
of passes and computation of analyses. This provides a quick glimpse into what
passes are taking the most time to execute, as well as how much of an effect a
pass has on the total execution time of the pipeline. Users can enable this
instrumentation directly on the PassManager via `enableTiming`. This
instrumentation is also made available in mlir-opt via the `-mlir-timing` flag.
The PassTiming instrumentation provides several different display modes for the
timing results, each of which is described below:

##### List Display Mode

In this mode, the results are displayed in a list sorted by total time with each
pass/analysis instance aggregated into one unique result. This view is useful
for getting an overview of what analyses/passes are taking the most time in a
pipeline. This display mode is available in mlir-opt via
`-mlir-timing-display=list`.

```shell
$ mlir-opt foo.mlir -mlir-disable-threading -pass-pipeline='func.func(cse,canonicalize)' -convert-func-to-llvm -mlir-timing -mlir-timing-display=list

===-------------------------------------------------------------------------===
                      ... Pass execution timing report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 0.0203 seconds

   ---Wall Time---  --- Name ---
   0.0047 ( 55.9%)  Canonicalizer
   0.0019 ( 22.2%)  VerifierPass
   0.0016 ( 18.5%)  LLVMLoweringPass
   0.0003 (  3.4%)  CSE
   0.0002 (  1.9%)  (A) DominanceInfo
   0.0084 (100.0%)  Total
```

##### Tree Display Mode

In this mode, the results are displayed in a nested pipeline view that mirrors
the internal pass pipeline that is being executed in the pass manager. This view
is useful for understanding specifically which parts of the pipeline are taking
the most time, and can also be used to identify when analyses are being
invalidated and recomputed. This is the default display mode.

```shell
$ mlir-opt foo.mlir -mlir-disable-threading -pass-pipeline='func.func(cse,canonicalize)' -convert-func-to-llvm -mlir-timing

===-------------------------------------------------------------------------===
                      ... Pass execution timing report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 0.0249 seconds

   ---Wall Time---  --- Name ---
   0.0058 ( 70.8%)  'func.func' Pipeline
   0.0004 (  4.3%)    CSE
   0.0002 (  2.6%)      (A) DominanceInfo
   0.0004 (  4.8%)    VerifierPass
   0.0046 ( 55.4%)    Canonicalizer
   0.0005 (  6.2%)    VerifierPass
   0.0005 (  5.8%)  VerifierPass
   0.0014 ( 17.2%)  LLVMLoweringPass
   0.0005 (  6.2%)  VerifierPass
   0.0082 (100.0%)  Total
```

##### Multi-threaded Pass Timing

When multi-threading is enabled in the pass manager the meaning of the display
slightly changes. First, a new timing column is added, `User Time`, that
displays the total time spent across all threads. Secondly, the `Wall Time`
column displays the longest individual time spent amongst all of the threads.
This means that the `Wall Time` column will continue to give an indicator on the
perceived time, or clock time, whereas the `User Time` will display the total
cpu time.

```shell
$ mlir-opt foo.mlir -pass-pipeline='func.func(cse,canonicalize)' -convert-func-to-llvm -mlir-timing

===-------------------------------------------------------------------------===
                      ... Pass execution timing report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 0.0078 seconds

   ---User Time---   ---Wall Time---  --- Name ---
   0.0177 ( 88.5%)     0.0057 ( 71.3%)  'func.func' Pipeline
   0.0044 ( 22.0%)     0.0015 ( 18.9%)    CSE
   0.0029 ( 14.5%)     0.0012 ( 15.2%)      (A) DominanceInfo
   0.0038 ( 18.9%)     0.0015 ( 18.7%)    VerifierPass
   0.0089 ( 44.6%)     0.0025 ( 31.1%)    Canonicalizer
   0.0006 (  3.0%)     0.0002 (  2.6%)    VerifierPass
   0.0004 (  2.2%)     0.0004 (  5.4%)  VerifierPass
   0.0013 (  6.5%)     0.0013 ( 16.3%)  LLVMLoweringPass
   0.0006 (  2.8%)     0.0006 (  7.0%)  VerifierPass
   0.0200 (100.0%)     0.0081 (100.0%)  Total
```

#### IR Printing

When debugging it is often useful to dump the IR at various stages of a pass
pipeline. This is where the IR printing instrumentation comes into play. This
instrumentation allows for conditionally printing the IR before and after pass
execution by optionally filtering on the pass being executed. This
instrumentation can be added directly to the PassManager via the
`enableIRPrinting` method. `mlir-opt` provides a few useful flags for utilizing
this instrumentation:

*   `mlir-print-ir-before=(comma-separated-pass-list)`
    *   Print the IR before each of the passes provided within the pass list.
*   `mlir-print-ir-before-all`
    *   Print the IR before every pass in the pipeline.

```shell
$ mlir-opt foo.mlir -pass-pipeline='func.func(cse)' -mlir-print-ir-before=cse

*** IR Dump Before CSE ***
func.func @simple_constant() -> (i32, i32) {
  %c1_i32 = arith.constant 1 : i32
  %c1_i32_0 = arith.constant 1 : i32
  return %c1_i32, %c1_i32_0 : i32, i32
}
```

*   `mlir-print-ir-after=(comma-separated-pass-list)`
    *   Print the IR after each of the passes provided within the pass list.
*   `mlir-print-ir-after-all`
    *   Print the IR after every pass in the pipeline.

```shell
$ mlir-opt foo.mlir -pass-pipeline='func.func(cse)' -mlir-print-ir-after=cse

*** IR Dump After CSE ***
func.func @simple_constant() -> (i32, i32) {
  %c1_i32 = arith.constant 1 : i32
  return %c1_i32, %c1_i32 : i32, i32
}
```

*   `mlir-print-ir-after-change`
    *   Only print the IR after a pass if the pass mutated the IR. This helps to
        reduce the number of IR dumps for "uninteresting" passes.
    *   Note: Changes are detected by comparing a hash of the operation before
        and after the pass. This adds additional run-time to compute the hash of
        the IR, and in some rare cases may result in false-positives depending
        on the collision rate of the hash algorithm used.
    *   Note: This option should be used in unison with one of the other
        'mlir-print-ir-after' options above, as this option alone does not enable
        printing.

```shell
$ mlir-opt foo.mlir -pass-pipeline='func.func(cse,cse)' -mlir-print-ir-after=cse -mlir-print-ir-after-change

*** IR Dump After CSE ***
func.func @simple_constant() -> (i32, i32) {
  %c1_i32 = arith.constant 1 : i32
  return %c1_i32, %c1_i32 : i32, i32
}
```

*   `mlir-print-ir-after-failure`
    *   Only print IR after a pass failure.
    *   This option should *not* be used with the other `mlir-print-ir-after` flags
        above.

```shell
$ mlir-opt foo.mlir -pass-pipeline='func.func(cse,bad-pass)' -mlir-print-ir-after-failure

*** IR Dump After BadPass Failed ***
func.func @simple_constant() -> (i32, i32) {
  %c1_i32 = arith.constant 1 : i32
  return %c1_i32, %c1_i32 : i32, i32
}
```

*   `mlir-print-ir-module-scope`
    *   Always print the top-level module operation, regardless of pass type or
        operation nesting level.
    *   Note: Printing at module scope should only be used when multi-threading
        is disabled(`-mlir-disable-threading`)

```shell
$ mlir-opt foo.mlir -mlir-disable-threading -pass-pipeline='func.func(cse)' -mlir-print-ir-after=cse -mlir-print-ir-module-scope

*** IR Dump After CSE ***  ('func.func' operation: @bar)
func.func @bar(%arg0: f32, %arg1: f32) -> f32 {
  ...
}

func.func @simple_constant() -> (i32, i32) {
  %c1_i32 = arith.constant 1 : i32
  %c1_i32_0 = arith.constant 1 : i32
  return %c1_i32, %c1_i32_0 : i32, i32
}

*** IR Dump After CSE ***  ('func.func' operation: @simple_constant)
func.func @bar(%arg0: f32, %arg1: f32) -> f32 {
  ...
}

func.func @simple_constant() -> (i32, i32) {
  %c1_i32 = arith.constant 1 : i32
  return %c1_i32, %c1_i32 : i32, i32
}
```

## Crash and Failure Reproduction

The [pass manager](#pass-manager) in MLIR contains a builtin mechanism to
generate reproducibles in the event of a crash, or a
[pass failure](#pass-failure). This functionality can be enabled via
`PassManager::enableCrashReproducerGeneration` or via the command line flag
`mlir-pass-pipeline-crash-reproducer`. In either case, an argument is provided that
corresponds to the output `.mlir` file name that the reproducible should be
written to. The reproducible contains the configuration of the pass manager that
was executing, as well as the initial IR before any passes were run. The reproducer
is stored within the assembly format as an external resource. A potential reproducible
may have the form:

```mlir
module {
  func.func @foo() {
    ...
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "func.func(cse,canonicalize),inline",
      disable_threading: true,
      verify_each: true
    }
  }
#-}
```

The configuration dumped can be passed to `mlir-opt`. This will result in
parsing the configuration of the reproducer and adjusting the necessary opt
state, e.g. configuring the pass manager, context, etc.

Beyond specifying a filename, one can also register a `ReproducerStreamFactory`
function that would be invoked in the case of a crash and the reproducer written
to its stream.

### Local Reproducer Generation

An additional flag may be passed to
`PassManager::enableCrashReproducerGeneration`, and specified via
`mlir-pass-pipeline-local-reproducer` on the command line, that signals that the pass
manager should attempt to generate a "local" reproducer. This will attempt to
generate a reproducer containing IR right before the pass that fails. This is
useful for situations where the crash is known to be within a specific pass, or
when the original input relies on components (like dialects or passes) that may
not always be available.

Note: Local reproducer generation requires that multi-threading is
disabled(`-mlir-disable-threading`)

For example, if the failure in the previous example came from the `canonicalize` pass,
the following reproducer would be generated:

```mlir
module {
  func.func @foo() {
    ...
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "func.func(canonicalize)",
      disable_threading: true,
      verify_each: true
    }
  }
#-}
```
