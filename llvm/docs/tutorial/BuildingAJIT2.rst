=====================================================================
Building a JIT: Adding Optimizations -- An introduction to ORC Layers
=====================================================================

.. contents::
   :local:

**This tutorial is under active development. It is incomplete and details may
change frequently.** Nonetheless we invite you to try it out as it stands, and
we welcome any feedback.

Chapter 2 Introduction
======================

Welcome to Chapter 2 of the "Building an ORC-based JIT in LLVM" tutorial. In
`Chapter 1 <BuildingAJIT1.html>`_ of this series we examined a basic JIT
class, KaleidoscopeJIT, that could take LLVM IR modules as input and produce
executable code in memory. KaleidoscopeJIT was able to do this with relatively
little code by composing two off-the-shelf *ORC layers*: IRCompileLayer and
ObjectLinkingLayer, to do much of the heavy lifting.

In this layer we'll learn more about the ORC layer concept by using a new layer,
IRTransformLayer, to add IR optimization support to KaleidoscopeJIT.

Optimizing Modules using the IRTransformLayer
=============================================

In `Chapter 4 <LangImpl04.html>`_ of the "Implementing a language with LLVM"
tutorial series the llvm *FunctionPassManager* is introduced as a means for
optimizing LLVM IR. Interested readers may read that chapter for details, but
in short: to optimize a Module we create an llvm::FunctionPassManager
instance, configure it with a set of optimizations, then run the PassManager on
a Module to mutate it into a (hopefully) more optimized but semantically
equivalent form. In the original tutorial series the FunctionPassManager was
created outside the KaleidoscopeJIT and modules were optimized before being
added to it. In this Chapter we will make optimization a phase of our JIT
instead. For now this will provide us a motivation to learn more about ORC
layers, but in the long term making optimization part of our JIT will yield an
important benefit: When we begin lazily compiling code (i.e. deferring
compilation of each function until the first time it's run), having
optimization managed by our JIT will allow us to optimize lazily too, rather
than having to do all our optimization up-front.

To add optimization support to our JIT we will take the KaleidoscopeJIT from
Chapter 1 and compose an ORC *IRTransformLayer* on top. We will look at how the
IRTransformLayer works in more detail below, but the interface is simple: the
constructor for this layer takes a reference to the layer below (as all layers
do) plus an *IR optimization function* that it will apply to each Module that
is added via addModuleSet:

.. code-block:: c++

  class KaleidoscopeJIT {
  private:
    std::unique_ptr<TargetMachine> TM;
    const DataLayout DL;
    ObjectLinkingLayer<> ObjectLayer;
    IRCompileLayer<decltype(ObjectLayer)> CompileLayer;

    typedef std::function<std::unique_ptr<Module>(std::unique_ptr<Module>)>
      OptimizeFunction;

    IRTransformLayer<decltype(CompileLayer), OptimizeFunction> OptimizeLayer;

  public:
    typedef decltype(OptimizeLayer)::ModuleSetHandleT ModuleHandle;

    KaleidoscopeJIT()
        : TM(EngineBuilder().selectTarget()), DL(TM->createDataLayout()),
          CompileLayer(ObjectLayer, SimpleCompiler(*TM)),
          OptimizeLayer(CompileLayer,
                        [this](std::unique_ptr<Module> M) {
                          return optimizeModule(std::move(M));
                        }) {
      llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
    }

Our extended KaleidoscopeJIT class starts out the same as it did in Chapter 1,
but after the CompileLayer we introduce a typedef for our optimization function.
In this case we use a std::function (a handy wrapper for "function-like" things)
from a single unique_ptr<Module> input to a std::unique_ptr<Module> output. With
our optimization function typedef in place we can declare our OptimizeLayer,
which sits on top of our CompileLayer.

To initialize our OptimizeLayer we pass it a reference to the CompileLayer
below (standard practice for layers), and we initialize the OptimizeFunction
using a lambda that calls out to an "optimizeModule" function that we will
define below.

.. code-block:: c++

  // ...
  auto Resolver = createLambdaResolver(
      [&](const std::string &Name) {
        if (auto Sym = OptimizeLayer.findSymbol(Name, false))
          return Sym;
        return JITSymbol(nullptr);
      },
  // ...

.. code-block:: c++

  // ...
  return OptimizeLayer.addModuleSet(std::move(Ms),
                                    make_unique<SectionMemoryManager>(),
                                    std::move(Resolver));
  // ...

.. code-block:: c++

  // ...
  return OptimizeLayer.findSymbol(MangledNameStream.str(), true);
  // ...

.. code-block:: c++

  // ...
  OptimizeLayer.removeModuleSet(H);
  // ...

Next we need to replace references to 'CompileLayer' with references to
OptimizeLayer in our key methods: addModule, findSymbol, and removeModule. In
addModule we need to be careful to replace both references: the findSymbol call
inside our resolver, and the call through to addModuleSet.

.. code-block:: c++

  std::unique_ptr<Module> optimizeModule(std::unique_ptr<Module> M) {
    // Create a function pass manager.
    auto FPM = llvm::make_unique<legacy::FunctionPassManager>(M.get());

    // Add some optimizations.
    FPM->add(createInstructionCombiningPass());
    FPM->add(createReassociatePass());
    FPM->add(createGVNPass());
    FPM->add(createCFGSimplificationPass());
    FPM->doInitialization();

    // Run the optimizations over all functions in the module being added to
    // the JIT.
    for (auto &F : *M)
      FPM->run(F);

    return M;
  }

At the bottom of our JIT we add a private method to do the actual optimization:
*optimizeModule*. This function sets up a FunctionPassManager, adds some passes
to it, runs it over every function in the module, and then returns the mutated
module. The specific optimizations are the same ones used in
`Chapter 4 <LangImpl04.html>`_ of the "Implementing a language with LLVM"
tutorial series. Readers may visit that chapter for a more in-depth
discussion of these, and of IR optimization in general.

And that's it in terms of changes to KaleidoscopeJIT: When a module is added via
addModule the OptimizeLayer will call our optimizeModule function before passing
the transformed module on to the CompileLayer below. Of course, we could have
called optimizeModule directly in our addModule function and not gone to the
bother of using the IRTransformLayer, but doing so gives us another opportunity
to see how layers compose. It also provides a neat entry point to the *layer*
concept itself, because IRTransformLayer turns out to be one of the simplest
implementations of the layer concept that can be devised:

.. code-block:: c++

  template <typename BaseLayerT, typename TransformFtor>
  class IRTransformLayer {
  public:
    typedef typename BaseLayerT::ModuleSetHandleT ModuleSetHandleT;

    IRTransformLayer(BaseLayerT &BaseLayer,
                     TransformFtor Transform = TransformFtor())
      : BaseLayer(BaseLayer), Transform(std::move(Transform)) {}

    template <typename ModuleSetT, typename MemoryManagerPtrT,
              typename SymbolResolverPtrT>
    ModuleSetHandleT addModuleSet(ModuleSetT Ms,
                                  MemoryManagerPtrT MemMgr,
                                  SymbolResolverPtrT Resolver) {

      for (auto I = Ms.begin(), E = Ms.end(); I != E; ++I)
        *I = Transform(std::move(*I));

      return BaseLayer.addModuleSet(std::move(Ms), std::move(MemMgr),
                                  std::move(Resolver));
    }

    void removeModuleSet(ModuleSetHandleT H) { BaseLayer.removeModuleSet(H); }

    JITSymbol findSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
      return BaseLayer.findSymbol(Name, ExportedSymbolsOnly);
    }

    JITSymbol findSymbolIn(ModuleSetHandleT H, const std::string &Name,
                           bool ExportedSymbolsOnly) {
      return BaseLayer.findSymbolIn(H, Name, ExportedSymbolsOnly);
    }

    void emitAndFinalize(ModuleSetHandleT H) {
      BaseLayer.emitAndFinalize(H);
    }

    TransformFtor& getTransform() { return Transform; }

    const TransformFtor& getTransform() const { return Transform; }

  private:
    BaseLayerT &BaseLayer;
    TransformFtor Transform;
  };

This is the whole definition of IRTransformLayer, from
``llvm/include/llvm/ExecutionEngine/Orc/IRTransformLayer.h``, stripped of its
comments. It is a template class with two template arguments: ``BaesLayerT`` and
``TransformFtor`` that provide the type of the base layer and the type of the
"transform functor" (in our case a std::function) respectively. This class is
concerned with two very simple jobs: (1) Running every IR Module that is added
with addModuleSet through the transform functor, and (2) conforming to the ORC
layer interface. The interface consists of one typedef and five methods:

+------------------+-----------------------------------------------------------+
|     Interface    |                         Description                       |
+==================+===========================================================+
|                  | Provides a handle that can be used to identify a module   |
| ModuleSetHandleT | set when calling findSymbolIn, removeModuleSet, or        |
|                  | emitAndFinalize.                                          |
+------------------+-----------------------------------------------------------+
|                  | Takes a given set of Modules and makes them "available    |
|                  | for execution. This means that symbols in those modules   |
|                  | should be searchable via findSymbol and findSymbolIn, and |
|                  | the address of the symbols should be read/writable (for   |
|                  | data symbols), or executable (for function symbols) after |
|                  | JITSymbol::getAddress() is called. Note: This means that  |
|   addModuleSet   | addModuleSet doesn't have to compile (or do any other     |
|                  | work) up-front. It *can*, like IRCompileLayer, act        |
|                  | eagerly, but it can also simply record the module and     |
|                  | take no further action until somebody calls               |
|                  | JITSymbol::getAddress(). In IRTransformLayer's case       |
|                  | addModuleSet eagerly applies the transform functor to     |
|                  | each module in the set, then passes the resulting set     |
|                  | of mutated modules down to the layer below.               |
+------------------+-----------------------------------------------------------+
|                  | Removes a set of modules from the JIT. Code or data       |
|  removeModuleSet | defined in these modules will no longer be available, and |
|                  | the memory holding the JIT'd definitions will be freed.   |
+------------------+-----------------------------------------------------------+
|                  | Searches for the named symbol in all modules that have    |
|                  | previously been added via addModuleSet (and not yet       |
|    findSymbol    | removed by a call to removeModuleSet). In                 |
|                  | IRTransformLayer we just pass the query on to the layer   |
|                  | below. In our REPL this is our default way to search for  |
|                  | function definitions.                                     |
+------------------+-----------------------------------------------------------+
|                  | Searches for the named symbol in the module set indicated |
|                  | by the given ModuleSetHandleT. This is just an optimized  |
|                  | search, better for lookup-speed when you know exactly     |
|                  | a symbol definition should be found. In IRTransformLayer  |
|   findSymbolIn   | we just pass this query on to the layer below. In our     |
|                  | REPL we use this method to search for functions           |
|                  | representing top-level expressions, since we know exactly |
|                  | where we'll find them: in the top-level expression module |
|                  | we just added.                                            |
+------------------+-----------------------------------------------------------+
|                  | Forces all of the actions required to make the code and   |
|                  | data in a module set (represented by a ModuleSetHandleT)  |
|                  | accessible. Behaves as if some symbol in the set had been |
|                  | searched for and JITSymbol::getSymbolAddress called. This |
| emitAndFinalize  | is rarely needed, but can be useful when dealing with     |
|                  | layers that usually behave lazily if the user wants to    |
|                  | trigger early compilation (for example, to use idle CPU   |
|                  | time to eagerly compile code in the background).          |
+------------------+-----------------------------------------------------------+

This interface attempts to capture the natural operations of a JIT (with some
wrinkles like emitAndFinalize for performance), similar to the basic JIT API
operations we identified in Chapter 1. Conforming to the layer concept allows
classes to compose neatly by implementing their behaviors in terms of the these
same operations, carried out on the layer below. For example, an eager layer
(like IRTransformLayer) can implement addModuleSet by running each module in the
set through its transform up-front and immediately passing the result to the
layer below. A lazy layer, by contrast, could implement addModuleSet by
squirreling away the modules doing no other up-front work, but applying the
transform (and calling addModuleSet on the layer below) when the client calls
findSymbol instead. The JIT'd program behavior will be the same either way, but
these choices will have different performance characteristics: Doing work
eagerly means the JIT takes longer up-front, but proceeds smoothly once this is
done. Deferring work allows the JIT to get up-and-running quickly, but will
force the JIT to pause and wait whenever some code or data is needed that hasn't
already been processed.

Our current REPL is eager: Each function definition is optimized and compiled as
soon as it's typed in. If we were to make the transform layer lazy (but not
change things otherwise) we could defer optimization until the first time we
reference a function in a top-level expression (see if you can figure out why,
then check out the answer below [1]_). In the next chapter, however we'll
introduce fully lazy compilation, in which function's aren't compiled until
they're first called at run-time. At this point the trade-offs get much more
interesting: the lazier we are, the quicker we can start executing the first
function, but the more often we'll have to pause to compile newly encountered
functions. If we only code-gen lazily, but optimize eagerly, we'll have a slow
startup (which everything is optimized) but relatively short pauses as each
function just passes through code-gen. If we both optimize and code-gen lazily
we can start executing the first function more quickly, but we'll have longer
pauses as each function has to be both optimized and code-gen'd when it's first
executed. Things become even more interesting if we consider interproceedural
optimizations like inlining, which must be performed eagerly. These are
complex trade-offs, and there is no one-size-fits all solution to them, but by
providing composable layers we leave the decisions to the person implementing
the JIT, and make it easy for them to experiment with different configurations.

`Next: Adding Per-function Lazy Compilation <BuildingAJIT3.html>`_

Full Code Listing
=================

Here is the complete code listing for our running example with an
IRTransformLayer added to enable optimization. To build this example, use:

.. code-block:: bash

    # Compile
    clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orc native` -O3 -o toy
    # Run
    ./toy

Here is the code:

.. literalinclude:: ../../examples/Kaleidoscope/BuildingAJIT/Chapter2/KaleidoscopeJIT.h
   :language: c++

.. [1] When we add our top-level expression to the JIT, any calls to functions
       that we defined earlier will appear to the ObjectLinkingLayer as
       external symbols. The ObjectLinkingLayer will call the SymbolResolver
       that we defined in addModuleSet, which in turn calls findSymbol on the
       OptimizeLayer, at which point even a lazy transform layer will have to
       do its work.
