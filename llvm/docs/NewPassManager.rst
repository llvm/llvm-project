==========================
Using the New Pass Manager
==========================

.. contents::
    :local:

Overview
========

For an overview of the new pass manager, see the `blog post
<https://blog.llvm.org/posts/2021-03-26-the-new-pass-manager/>`_.

Just Tell Me How To Run The Default Optimization Pipeline With The New Pass Manager
===================================================================================

.. code-block:: c++

  // Create the analysis managers.
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // Create the new pass manager builder.
  // Take a look at the PassBuilder constructor parameters for more
  // customization, e.g. specifying a TargetMachine or various debugging
  // options.
  PassBuilder PB;

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Create the pass manager.
  // This one corresponds to a typical -O2 optimization pipeline.
  ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(OptimizationLevel::O2);

  // Optimize the IR!
  MPM.run(MyModule, MAM);

The C API also supports most of this, see ``llvm-c/Transforms/PassBuilder.h``.

Adding Passes to a Pass Manager
===============================

For how to write a new PM pass, see :doc:`this page <WritingAnLLVMNewPMPass>`.

To add a pass to a new PM pass manager, the important thing is to match the
pass type and the pass manager type. For example, a ``FunctionPassManager``
can only contain function passes:

.. code-block:: c++

  FunctionPassManager FPM;
  // InstSimplifyPass is a function pass
  FPM.addPass(InstSimplifyPass());

If you want to add a loop pass that runs on all loops in a function to a
``FunctionPassManager``, the loop pass must be wrapped in a function pass
adaptor that goes through all the loops in the function and runs the loop
pass on each one.

.. code-block:: c++

  FunctionPassManager FPM;
  // LoopRotatePass is a loop pass
  FPM.addPass(createFunctionToLoopPassAdaptor(LoopRotatePass()));

The IR hierarchy in terms of the new PM is Module -> (CGSCC ->) Function ->
Loop, where going through a CGSCC is optional.

.. code-block:: c++

  FunctionPassManager FPM;
  // loop -> function
  FPM.addPass(createFunctionToLoopPassAdaptor(LoopFooPass()));

  CGSCCPassManager CGPM;
  // loop -> function -> cgscc
  CGPM.addPass(createCGSCCToFunctionPassAdaptor(createFunctionToLoopPassAdaptor(LoopFooPass())));
  // function -> cgscc
  CGPM.addPass(createCGSCCToFunctionPassAdaptor(FunctionFooPass()));

  ModulePassManager MPM;
  // loop -> function -> module
  MPM.addPass(createModuleToFunctionPassAdaptor(createFunctionToLoopPassAdaptor(LoopFooPass())));
  // function -> module
  MPM.addPass(createModuleToFunctionPassAdaptor(FunctionFooPass()));

  // loop -> function -> cgscc -> module
  MPM.addPass(createModuleToCGSCCPassAdaptor(createCGSCCToFunctionPassAdaptor(createFunctionToLoopPassAdaptor(LoopFooPass()))));
  // function -> cgscc -> module
  MPM.addPass(createModuleToCGSCCPassAdaptor(createCGSCCToFunctionPassAdaptor(FunctionFooPass())));


A pass manager of a specific IR unit is also a pass of that kind. For
example, a ``FunctionPassManager`` is a function pass, meaning it can be
added to a ``ModulePassManager``:

.. code-block:: c++

  ModulePassManager MPM;

  FunctionPassManager FPM;
  // InstSimplifyPass is a function pass
  FPM.addPass(InstSimplifyPass());

  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

Generally you want to group CGSCC/function/loop passes together in a pass
manager, as opposed to adding adaptors for each pass to the containing upper
level pass manager. For example,

.. code-block:: c++

  ModulePassManager MPM;
  MPM.addPass(createModuleToFunctionPassAdaptor(FunctionPass1()));
  MPM.addPass(createModuleToFunctionPassAdaptor(FunctionPass2()));
  MPM.run();

will run ``FunctionPass1`` on each function in a module, then run
``FunctionPass2`` on each function in the module. In contrast,

.. code-block:: c++

  ModulePassManager MPM;

  FunctionPassManager FPM;
  FPM.addPass(FunctionPass1());
  FPM.addPass(FunctionPass2());

  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

will run ``FunctionPass1`` and ``FunctionPass2`` on the first function in a
module, then run both passes on the second function in the module, and so on.
This is better for cache locality around LLVM data structures. This similarly
applies for the other IR types, and in some cases can even affect the quality
of optimization. For example, running all loop passes on a loop may cause a
later loop to be able to be optimized more than if each loop pass were run
separately.

Inserting Passes into Default Pipelines
=======================================

Rather than manually adding passes to a pass manager, the typical way of
creating a pass manager is to use a ``PassBuilder`` and call something like
``PassBuilder::buildPerModuleDefaultPipeline()`` which creates a typical
pipeline for a given optimization level.

Sometimes either frontends or backends will want to inject passes into the
pipeline. For example, frontends may want to add instrumentation, and target
backends may want to add passes that lower custom intrinsics. For these
cases, ``PassBuilder`` exposes callbacks that allow injecting passes into
certain parts of the pipeline. For example,

.. code-block:: c++

  PassBuilder PB;
  PB.registerPipelineStartEPCallback([&](ModulePassManager &MPM,
                                         PassBuilder::OptimizationLevel Level) {
      MPM.addPass(FooPass());
  };

will add ``FooPass`` near the very beginning of the pipeline for pass
managers created by that ``PassBuilder``. See the documentation for
``PassBuilder`` for the various places that passes can be added.

If a ``PassBuilder`` has a corresponding ``TargetMachine`` for a backend, it
will call ``TargetMachine::registerPassBuilderCallbacks()`` to allow the
backend to inject passes into the pipeline. This is equivalent to the legacy
PM's ``TargetMachine::adjustPassManager()``.

Clang's ``BackendUtil.cpp`` shows examples of a frontend adding (mostly
sanitizer) passes to various parts of the pipeline.
``AMDGPUTargetMachine::registerPassBuilderCallbacks()`` is an example of a
backend adding passes to various parts of the pipeline.

Using Analyses
==============

LLVM provides many analyses that passes can use, such as a dominator tree.
Calculating these can be expensive, so the new pass manager has
infrastructure to cache analyses and reuse them when possible.

When a pass runs on some IR, it also receives an analysis manager which it can
query for analyses. Querying for an analysis will cause the manager to check if
it has already computed the result for the requested IR. If it already has and
the result is still valid, it will return that. Otherwise it will construct a
new result by calling the analysis's ``run()`` method, cache it, and return it.
You can also ask the analysis manager to only return an analysis if it's
already cached.

The analysis manager only provides analysis results for the same IR type as
what the pass runs on. For example, a function pass receives an analysis
manager that only provides function-level analyses. This works for many
passes which work on a fixed scope. However, some passes want to peek up or
down the IR hierarchy. For example, an SCC pass may want to look at function
analyses for the functions inside the SCC. Or it may want to look at some
immutable global analysis. In these cases, the analysis manager can provide a
proxy to an outer or inner level analysis manager. For example, to get a
``FunctionAnalysisManager`` from a ``CGSCCAnalysisManager``, you can call

.. code-block:: c++

  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerCGSCCProxy>(InitialC, CG)
          .getManager();

and use ``FAM`` as a typical ``FunctionAnalysisManager`` that a function pass
would have access to. To get access to an outer level IR analysis, you can
call

.. code-block:: c++

  const auto &MAMProxy =
      AM.getResult<ModuleAnalysisManagerCGSCCProxy>(InitialC, CG);
  FooAnalysisResult *AR = MAMProxy.getCachedResult<FooAnalysis>(M);

Getting direct access to an outer level IR analysis manager is not allowed.
This is to keep in mind potential future pass concurrency, for example
parallelizing function passes over different functions in a CGSCC or module.
Since passes can ask for a cached analysis result, allowing passes to trigger
outer level analysis computation could result in non-determinism if
concurrency was supported. Therefore a pass running on inner level IR cannot
change the state of outer level IR analyses. Another limitation is that outer
level IR analyses that are used must be immutable, or else they could be
invalidated by changes to inner level IR. Outer analyses unused by inner
passes can and often will be invalidated by changes to inner level IR. These
invalidations happen after the inner pass manager finishes, so accessing
mutable analyses would give invalid results.

The exception to the above is accessing function analyses in loop passes.
Loop passes inherently require modifying the function the loop is in, and
that includes some function analyses the loop analyses depend on. This
discounts future concurrency over separate loops in a function, but that's a
tradeoff due to how tightly a loop and its function are coupled. To make sure
the function analyses loop passes use are valid, they are manually updated in
the loop passes to ensure that invalidation is not necessary. There is a set
of common function analyses that loop passes and analyses have access to
which is passed into loop passes as a ``LoopStandardAnalysisResults``
parameter. Other function analyses are not accessible from loop passes.

As with any caching mechanism, we need some way to tell analysis managers
when results are no longer valid. Much of the analysis manager complexity
comes from trying to invalidate as few analysis results as possible to keep
compile times as low as possible.

There are two ways to deal with potentially invalid analysis results. One is
to simply force clear the results. This should generally only be used when
the IR that the result is keyed on becomes invalid. For example, a function
is deleted, or a CGSCC has become invalid due to call graph changes.

The typical way to invalidate analysis results is for a pass to declare what
types of analyses it preserves and what types it does not. When transforming
IR, a pass either has the option to update analyses alongside the IR
transformation, or tell the analysis manager that analyses are no longer
valid and should be invalidated. If a pass wants to keep some specific
analysis up to date, such as when updating it would be faster than
invalidating and recalculating it, the analysis itself may have methods to
update it for specific transformations, or there may be helper updaters like
``DomTreeUpdater`` for a ``DominatorTree``. Otherwise to mark some analysis
as no longer valid, the pass can return a ``PreservedAnalyses`` with the
proper analyses invalidated.

.. code-block:: c++

  // We've made no transformations that can affect any analyses.
  return PreservedAnalyses::all();

  // We've made transformations and don't want to bother to update any analyses.
  return PreservedAnalyses::none();

  // We've specifically updated the dominator tree alongside any transformations, but other analysis results may be invalid.
  PreservedAnalyses PA;
  PA.preserve<DominatorAnalysis>();
  return PA;

  // We haven't made any control flow changes, any analyses that only care about the control flow are still valid.
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;

The pass manager will call the analysis manager's ``invalidate()`` method
with the pass's returned ``PreservedAnalyses``. This can be also done
manually within the pass:

.. code-block:: c++

  FooModulePass::run(Module& M, ModuleAnalysisManager& AM) {
    auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

    // Invalidate all analysis results for function F
    FAM.invalidate(F, PreservedAnalyses::none());

    // Invalidate all analysis results
    AM.invalidate(M, PreservedAnalyses::none());

    ...
  }

This is especially important when a pass removes then adds a function. The
analysis manager may store a pointer to a function that has been deleted, and
if the pass creates a new function before invalidating analysis results, the
new function may be at the same address as the old one, causing invalid
cached results. This is also useful for being more precise about
invalidation. Selectively invalidating analysis results only for functions
modified in an SCC pass can allow more analysis results to remain. But except
for complex fine-grain invalidation with inner proxies, passes should
typically just return a proper ``PreservedAnalyses`` and let the pass manager
deal with proper invalidation.

Implementing Analysis Invalidation
==================================

By default, an analysis is invalidated if ``PreservedAnalyses`` says that
analyses on the IR unit it runs on are not preserved (see
``AnalysisResultModel::invalidate()``). An analysis can implement
``invalidate()`` to be more conservative when it comes to invalidation. For
example,

.. code-block:: c++

  bool FooAnalysisResult::invalidate(Function &F, const PreservedAnalyses &PA,
                                     FunctionAnalysisManager::Invalidator &) {
    auto PAC = PA.getChecker<FooAnalysis>();
    // the default would be:
    // return !(PAC.preserved() || PAC.preservedSet<AllAnalysesOn<Function>>());
    return !(PAC.preserved() || PAC.preservedSet<AllAnalysesOn<Function>>()
        || PAC.preservedSet<CFGAnalyses>());
  }

says that if the ``PreservedAnalyses`` specifically preserves
``FooAnalysis``, or if ``PreservedAnalyses`` preserves all analyses (implicit
in ``PAC.preserved()``), or if ``PreservedAnalyses`` preserves all function
analyses, or ``PreservedAnalyses`` preserves all analyses that only care
about the CFG, the ``FooAnalysisResult`` should not be invalidated.

If an analysis is stateless and generally shouldn't be invalidated, use the
following:

.. code-block:: c++

  bool FooAnalysisResult::invalidate(Function &F, const PreservedAnalyses &PA,
                                     FunctionAnalysisManager::Invalidator &) {
    // Check whether the analysis has been explicitly invalidated. Otherwise, it's
    // stateless and remains preserved.
    auto PAC = PA.getChecker<FooAnalysis>();
    return !PAC.preservedWhenStateless();
  }

If an analysis depends on other analyses, those analyses also need to be
checked if they are invalidated:

.. code-block:: c++

  bool FooAnalysisResult::invalidate(Function &F, const PreservedAnalyses &PA,
                                     FunctionAnalysisManager::Invalidator &) {
    auto PAC = PA.getChecker<FooAnalysis>();
    if (!PAC.preserved() && !PAC.preservedSet<AllAnalysesOn<Function>>())
      return true;

    // Check transitive dependencies.
    return Inv.invalidate<BarAnalysis>(F, PA) ||
          Inv.invalidate<BazAnalysis>(F, PA);
  }

Combining invalidation and analysis manager proxies results in some
complexity. For example, when we invalidate all analyses in a module pass,
we have to make sure that we also invalidate function analyses accessible via
any existing inner proxies. The inner proxy's ``invalidate()`` first checks
if the proxy itself should be invalidated. If so, that means the proxy may
contain pointers to IR that is no longer valid, meaning that the inner proxy
needs to completely clear all relevant analysis results. Otherwise the proxy
simply forwards the invalidation to the inner analysis manager.

Generally for outer proxies, analysis results from the outer analysis manager
should be immutable, so invalidation shouldn't be a concern. However, it is
possible for some inner analysis to depend on some outer analysis, and when
the outer analysis is invalidated, we need to make sure that dependent inner
analyses are also invalidated. This actually happens with alias analysis
results. Alias analysis is a function-level analysis, but there are
module-level implementations of specific types of alias analysis. Currently
``GlobalsAA`` is the only module-level alias analysis and it generally is not
invalidated so this is not so much of a concern. See
``OuterAnalysisManagerProxy::Result::registerOuterAnalysisInvalidation()``
for more details.

Invoking ``opt``
================

To use the legacy pass manager:

.. code-block:: shell

  $ opt -enable-new-pm=0 -pass1 -pass2 /tmp/a.ll -S

This will be removed once the legacy pass manager is deprecated and removed for
the optimization pipeline.

To use the new PM:

.. code-block:: shell

  $ opt -passes='pass1,pass2' /tmp/a.ll -S

The new PM typically requires explicit pass nesting. For example, to run a
function pass, then a module pass, we need to wrap the function pass in a module
adaptor:

.. code-block:: shell

  $ opt -passes='function(no-op-function),no-op-module' /tmp/a.ll -S

A more complete example, and ``-debug-pass-manager`` to show the execution
order:

.. code-block:: shell

  $ opt -passes='no-op-module,cgscc(no-op-cgscc,function(no-op-function,loop(no-op-loop))),function(no-op-function,loop(no-op-loop))' /tmp/a.ll -S -debug-pass-manager

Improper nesting can lead to error messages such as

.. code-block:: shell

  $ opt -passes='no-op-function,no-op-module' /tmp/a.ll -S
  opt: unknown function pass 'no-op-module'

The nesting is: module (-> cgscc) -> function -> loop, where the CGSCC nesting is optional.

There are a couple of special cases for easier typing:

* If the first pass is not a module pass, a pass manager of the first pass is
  implicitly created

  * For example, the following are equivalent

.. code-block:: shell

  $ opt -passes='no-op-function,no-op-function' /tmp/a.ll -S
  $ opt -passes='function(no-op-function,no-op-function)' /tmp/a.ll -S

* If there is an adaptor for a pass that lets it fit in the previous pass
  manager, that is implicitly created

  * For example, the following are equivalent

.. code-block:: shell

  $ opt -passes='no-op-function,no-op-loop' /tmp/a.ll -S
  $ opt -passes='no-op-function,loop(no-op-loop)' /tmp/a.ll -S

For a list of available passes and analyses, including the IR unit (module,
CGSCC, function, loop) they operate on, run

.. code-block:: shell

  $ opt --print-passes

or take a look at ``PassRegistry.def``.

To make sure an analysis named ``foo`` is available before a pass, add
``require<foo>`` to the pass pipeline. This adds a pass that simply requests
that the analysis is run. This pass is also subject to proper nesting.  For
example, to make sure some function analysis is already computed for all
functions before a module pass:

.. code-block:: shell

  $ opt -passes='function(require<my-function-analysis>),my-module-pass' /tmp/a.ll -S

Status of the New and Legacy Pass Managers
==========================================

LLVM currently contains two pass managers, the legacy PM and the new PM. The
optimization pipeline (aka the middle-end) works with both the legacy PM and
the new PM, whereas the backend target-dependent code generation only works
with the legacy PM.

For the optimization pipeline, the new PM is the default PM. The legacy PM is
available for the optimization pipeline either by setting the CMake flag
``-DLLVM_ENABLE_NEW_PASS_MANAGER=OFF`` when building LLVM, or by
various compiler/linker flags, e.g. ``-flegacy-pass-manager`` for ``clang``.

There will be efforts to deprecate and remove the legacy PM for the
optimization pipeline in the future.

Some IR passes are considered part of the backend codegen pipeline even if
they are LLVM IR passes (whereas all MIR passes are codegen passes). This
includes anything added via ``TargetPassConfig`` hooks, e.g.
``TargetPassConfig::addCodeGenPrepare()``. As mentioned before, passes added
in ``TargetMachine::adjustPassManager()`` are part of the optimization
pipeline, and should have a corresponding line in
``TargetMachine::registerPassBuilderCallbacks()``.

Currently there are efforts to make the codegen pipeline work with the new
PM.
