=============================================
Building a JIT: Per-function Lazy Compilation
=============================================

.. contents::
   :local:

**This tutorial is under active development. It is incomplete and details may
change frequently.** Nonetheless we invite you to try it out as it stands, and
we welcome any feedback.

Chapter 3 Introduction
======================

Welcome to Chapter 3 of the "Building an ORC-based JIT in LLVM" tutorial. This
chapter discusses lazy JITing and shows you how to enable it by adding an ORC
CompileOnDemand layer the JIT from `Chapter 2 <BuildingAJIT2.html>`_.

Lazy Compilation
================

When we add a module to the KaleidoscopeJIT class described in Chapter 2 it is
immediately optimized, compiled and linked for us by the IRTransformLayer,
IRCompileLayer and ObjectLinkingLayer respectively. This scheme, where all the
work to make a Module executable is done up front, is relatively simple to
understand its performance characteristics are easy to reason about. However,
it will lead to very high startup times if the amount of code to be compiled is
large, and may also do a lot of unnecessary compilation if only a few compiled
functions are ever called at runtime. A truly "just-in-time" compiler should
allow us to defer the compilation of any given function until the moment that
function is first called, improving launch times and eliminating redundant work.
In fact, the ORC APIs provide us with a layer to lazily compile LLVM IR:
*CompileOnDemandLayer*.

The CompileOnDemandLayer conforms to the layer interface described in Chapter 2,
but the addModuleSet method behaves quite differently from the layers we have
seen so far: rather than doing any work up front, it just constructs a *stub*
for each function in the module and arranges for the stub to trigger compilation
of the actual function the first time it is called. Because stub functions are
very cheap to produce CompileOnDemand's addModuleSet method runs very quickly,
reducing the time required to launch the first function to be executed, and
saving us from doing any redundant compilation. By conforming to the layer
interface, CompileOnDemand can be easily added on top of our existing JIT class.
We just need a few changes:

.. code-block:: c++

  ...
  #include "llvm/ExecutionEngine/SectionMemoryManager.h"
  #include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
  #include "llvm/ExecutionEngine/Orc/CompileUtils.h"
  ...

  ...
  class KaleidoscopeJIT {
  private:
    std::unique_ptr<TargetMachine> TM;
    const DataLayout DL;
    std::unique_ptr<JITCompileCallbackManager> CompileCallbackManager;
    ObjectLinkingLayer<> ObjectLayer;
    IRCompileLayer<decltype(ObjectLayer)> CompileLayer;

    typedef std::function<std::unique_ptr<Module>(std::unique_ptr<Module>)>
      OptimizeFunction;

    IRTransformLayer<decltype(CompileLayer), OptimizeFunction> OptimizeLayer;
    CompileOnDemandLayer<decltype(OptimizeLayer)> CODLayer;

  public:
    typedef decltype(CODLayer)::ModuleSetHandleT ModuleHandle;

First we need to include the CompileOnDemandLayer.h header, then add two new
members: a std::unique_ptr<CompileCallbackManager> and a CompileOnDemandLayer,
to our class. The CompileCallbackManager is a utility that enables us to
create re-entry points into the compiler for functions that we want to lazily
compile. In the next chapter we'll be looking at this class in detail, but for
now we'll be treating it as an opaque utility: We just need to pass a reference
to it into our new CompileOnDemandLayer, and the layer will do all the work of
setting up the callbacks using the callback manager we gave it.

.. code-block:: c++

  KaleidoscopeJIT()
      : TM(EngineBuilder().selectTarget()), DL(TM->createDataLayout()),
        CompileLayer(ObjectLayer, SimpleCompiler(*TM)),
        OptimizeLayer(CompileLayer,
                      [this](std::unique_ptr<Module> M) {
                        return optimizeModule(std::move(M));
                      }),
        CompileCallbackManager(
            orc::createLocalCompileCallbackManager(TM->getTargetTriple(), 0)),
        CODLayer(OptimizeLayer,
                 [this](Function &F) { return std::set<Function*>({&F}); },
                 *CompileCallbackManager,
                 orc::createLocalIndirectStubsManagerBuilder(
                   TM->getTargetTriple())) {
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

Next we have to update our constructor to initialize the new members. To create
an appropriate compile callback manager we use the
createLocalCompileCallbackManager function, which takes a TargetMachine and a
TargetAddress to call if it receives a request to compile an unknown function.
In our simple JIT this situation is unlikely to come up, so we'll cheat and
just pass '0' here. In a production quality JIT you could give the address of a
function that throws an exception in order to unwind the JIT'd code stack.

Now we can construct our CompileOnDemandLayer. Following the pattern from
previous layers we start by passing a reference to the next layer down in our
stack -- the OptimizeLayer. Next we need to supply a 'partitioning function':
when a not-yet-compiled function is called, the CompileOnDemandLayer will call
this function to ask us what we would like to compile. At a minimum we need to
compile the function being called (given by the argument to the partitioning
function), but we could also request that the CompileOnDemandLayer compile other
functions that are unconditionally called (or highly likely to be called) from
the function being called. For KaleidoscopeJIT we'll keep it simple and just
request compilation of the function that was called. Next we pass a reference to
our CompileCallbackManager. Finally, we need to supply an "indirect stubs
manager builder". This is a function that constructs IndirectStubManagers, which
are in turn used to build the stubs for each module. The CompileOnDemandLayer
will call the indirect stub manager builder once for each call to addModuleSet,
and use the resulting indirect stubs manager to create stubs for all functions
in all modules added. If/when the module set is removed from the JIT the
indirect stubs manager will be deleted, freeing any memory allocated to the
stubs. We supply this function by using the
createLocalIndirectStubsManagerBuilder utility.

.. code-block:: c++

  // ...
          if (auto Sym = CODLayer.findSymbol(Name, false))
  // ...
  return CODLayer.addModuleSet(std::move(Ms),
                               make_unique<SectionMemoryManager>(),
                               std::move(Resolver));
  // ...

  // ...
  return CODLayer.findSymbol(MangledNameStream.str(), true);
  // ...

  // ...
  CODLayer.removeModuleSet(H);
  // ...

Finally, we need to replace the references to OptimizeLayer in our addModule,
findSymbol, and removeModule methods. With that, we're up and running.

**To be done:**

** Discuss CompileCallbackManagers and IndirectStubManagers in more detail.**

Full Code Listing
=================

Here is the complete code listing for our running example with a CompileOnDemand
layer added to enable lazy function-at-a-time compilation. To build this example, use:

.. code-block:: bash

    # Compile
    clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orc native` -O3 -o toy
    # Run
    ./toy

Here is the code:

.. literalinclude:: ../../examples/Kaleidoscope/BuildingAJIT/Chapter3/KaleidoscopeJIT.h
   :language: c++

`Next: Extreme Laziness -- Using Compile Callbacks to JIT directly from ASTs <BuildingAJIT4.html>`_
