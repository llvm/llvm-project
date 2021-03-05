==========================
Using the New Pass Manager
==========================

.. contents::
    :local:

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

If you want add a loop pass that runs on all loops in a function to a
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

Status of the New and Legacy Pass Managers
==========================================

LLVM currently contains two pass managers, the legacy PM and the new PM. The
optimization pipeline (aka the middle-end) works with both the legacy PM and
the new PM, whereas the backend target-dependent code generation only works
with the legacy PM.

For the optimization pipeline, the new PM is the default PM. The legacy PM is
available for the optimization pipeline either by setting the CMake flag
``-DENABLE_EXPERIMENTAL_NEW_PASS_MANAGER=OFF`` when building LLVM, or by
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
