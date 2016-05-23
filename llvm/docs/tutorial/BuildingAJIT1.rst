=======================================================
Kaleidoscope: Building an ORC-based JIT in LLVM
=======================================================

.. contents::
   :local:

**This tutorial is under active development. It is incomplete and details may
change frequently.** Nonetheless we invite you to try it out as it stands, and
we welcome any feedback.

Chapter 1 Introduction
======================

Welcome to Chapter 1 of the "Building an ORC-based JIT in LLVM" tutorial. This
tutorial runs through the implementation of a JIT compiler using LLVM's
On-Request-Compilation (ORC) APIs. It begins with a simplified version of the
KaleidoscopeJIT class used in the
`Implementing a language with LLVM <LangImpl1.html>`_ tutorials and then
introduces new features like optimization, lazy compilation and remote
execution.

The goal of this tutorial is to introduce you to LLVM's ORC JIT APIs, show how
these APIs interact with other parts of LLVM, and to teach you how to recombine
them to build a custom JIT that is suited to your use-case.

The structure of the tutorial is:

- Chapter #1: Investigate the simple KaleidoscopeJIT class. This will
  introduce some of the basic concepts of the ORC JIT APIs, including the
  idea of an ORC *Layer*.

- `Chapter #2 <BuildingAJIT2.html>`_: Extend the basic KaleidoscopeJIT by adding
  a new layer that will optimize IR and generated code.

- `Chapter #3 <BuildingAJIT3.html>`_: Further extend the JIT by adding a
  Compile-On-Demand layer to lazily compile IR.

- `Chapter #4 <BuildingAJIT4.html>`_: Improve the laziness of our JIT by
  replacing the Compile-On-Demand layer with a custom layer that uses the ORC
  Compile Callbacks API directly to defer IR-generation until functions are
  called.

- `Chapter #5 <BuildingAJIT5.html>`_: Add process isolation by JITing code into
  a remote process with reduced privileges using the JIT Remote APIs.

To provide input for our JIT we will use the Kaleidoscope REPL from
`Chapter 7 <LangImpl7.html>`_ of the "Implementing a language in LLVM tutorial",
with one minor modification: We will remove the FunctionPassManager from the
code for that chapter and replace it with optimization support in our JIT class
in Chapter #2.

Finally, a word on API generations: ORC is the 3rd generation of LLVM JIT API.
It was preceeded by MCJIT, and before that by the (now deleted) legacy JIT.
These tutorials don't assume any experience with these earlier APIs, but
readers acquainted with them will see many familiar elements. Where appropriate
we will make this connection with the earlier APIs explicit to help people who
are transitioning from them to ORC.

JIT API Basics
==============

The purpose of a JIT compiler is to compile code "on-the-fly" as it is needed,
rather than compiling whole programs to disk ahead of time as a traditional
compiler does. To support that aim our initial, bare-bones JIT API will be:

1. Handle addModule(Module &M) -- Make the given IR module available for
   execution.
2. JITSymbol findSymbol(const std::string &Name) -- Search for pointers to
   symbols (functions or variables) that have been added to the JIT.
3. void removeModule(Handle H) -- Remove a module from the JIT, releasing any
   memory that had been used for the compiled code.

A basic use-case for this API, executing the 'main' function from a module,
will look like:

.. code-block:: c++

    std::unique_ptr<Module> M = buildModule();
    JIT J;
    Handle H = J.addModule(*M);
    int (*Main)(int, char*[]) =
      (int(*)(int, char*[])J.findSymbol("main").getAddress();
    int Result = Main();
    J.removeModule(H);

The APIs that we build in these tutorials will all be variations on this simple
theme. Behind the API we will refine the implementation of the JIT to add
support for optimization and lazy compilation. Eventually we will extend the
API itself to allow higher-level program representations (e.g. ASTs) to be
added to the JIT.

KaleidoscopeJIT
===============

In the previous section we described our API, now we examine a simple
implementation of it: The KaleidoscopeJIT class [1]_ that was used in the
`Implementing a language with LLVM <LangImpl1.html>`_ tutorials. We will use
the REPL code from `Chapter 7 <LangImpl7.html>`_ of that tutorial to supply the
input for our JIT: Each time the user enters an expression the REPL will add a
new IR module containing the code for that expression to the JIT. If the
expression is a top-level expression like '1+1' or 'sin(x)', the REPL will also
use the findSymbol method of our JIT class find and execute the code for the
expression, and then use the removeModule method to remove the code again
(since there's no way to re-invoke an anonymous expression). In later chapters
of this tutorial we'll modify the REPL to enable new interactions with our JIT
class, but for now we will take this setup for granted and focus our attention on
the implementation of our JIT itself.

Our KaleidoscopeJIT class is defined in the KaleidoscopeJIT.h header. After the
usual include guards and #includes [2]_, we get to the definition of our class:

.. code-block:: c++

    #ifndef LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H
    #define LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H

    #include "llvm/ExecutionEngine/ExecutionEngine.h"
    #include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
    #include "llvm/ExecutionEngine/Orc/CompileUtils.h"
    #include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
    #include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
    #include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
    #include "llvm/IR/Mangler.h"
    #include "llvm/Support/DynamicLibrary.h"

    namespace llvm {
    namespace orc {

    class KaleidoscopeJIT {
    private:

      std::unique_ptr<TargetMachine> TM;
      const DataLayout DL;
      ObjectLinkingLayer<> ObjectLayer;
      IRCompileLayer<decltype(ObjectLayer)> CompileLayer;

    public:

      typedef decltype(CompileLayer)::ModuleSetHandleT ModuleHandleT;

Our class begins with four members: A TargetMachine, TM, which will be used
to build our LLVM compiler instance; A DataLayout, DL, which will be used for
symbol mangling (more on that later), and two ORC *layers*: An
ObjectLinkingLayer, and an IRCompileLayer. The ObjectLinkingLayer is the
foundation of our JIT: it takes in-memory object files produced by a
compiler and links them on the fly to make them executable. This
JIT-on-top-of-a-linker design was introduced in MCJIT, where the linker was
hidden inside the MCJIT class itself. In ORC we expose the linker as a visible,
reusable component so that clients can access and configure it directly
if they need to. In this tutorial our ObjectLinkingLayer will just be used to
support the next layer in our stack: the IRCompileLayer, which will be
responsible for taking LLVM IR, compiling it, and passing the resulting
in-memory object files down to the object linking layer below.

After our member variables comes typedef: ModuleHandle. This is the handle
type that will be returned from our JIT's addModule method, and which can be
used to remove a module again using the removeModule method. The IRCompileLayer
class already provides a convenient handle type
(IRCompileLayer::ModuleSetHandleT), so we will just provide a type-alias for
this.

.. code-block:: c++

      KaleidoscopeJIT()
          : TM(EngineBuilder().selectTarget()), DL(TM->createDataLayout()),
            CompileLayer(ObjectLayer, SimpleCompiler(*TM)) {
        llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
      }

      TargetMachine &getTargetMachine() { return *TM; }

Next up we have our class constructor. We begin by initializing TM using the
EngineBuilder::selectTarget helper method, which constructs a TargetMachine for
the current process. Next we use our newly created TargetMachine to initialize
DL, our DataLayout. Then we initialize our IRCompileLayer. Our IRCompile layer
needs two things: (1) A reference to our object linking layer, and (2) a
compiler instance to use to perform the actual compilation from IR to object
files. We use the off-the-shelf SimpleCompiler instance for now, but in later
chapters we will substitute our own configurable compiler classes. Finally, in
the body of the constructor, we call the DynamicLibrary::LoadLibraryPermanently
method with a nullptr argument. Normally the LoadLibraryPermanently method is
called with the path of a dynamic library to load, but when passed a null
pointer it will 'load' the host process itself, making its exported symbols
available for execution.

.. code-block:: c++

    ModuleHandleT addModule(std::unique_ptr<Module> M) {
      // We need a memory manager to allocate memory and resolve symbols for this
      // new module. Create one that resolves symbols by looking back into the
      // JIT.
      auto Resolver = createLambdaResolver(
          [&](const std::string &Name) {
            if (auto Sym = CompileLayer.findSymbol(Name, false))
              return RuntimeDyld::SymbolInfo(Sym.getAddress(), Sym.getFlags());
            return RuntimeDyld::SymbolInfo(nullptr);
          },
          [](const std::string &S) { return nullptr; });
      std::vector<std::unique_ptr<Module>> Ms;
      Ms.push_back(std::move(M));
      return CompileLayer.addModuleSet(singletonSet(std::move(M)),
                                       make_unique<SectionMemoryManager>(),
                                       std::move(Resolver));
    }

*To be done: describe addModule -- createLambdaResolver, resolvers, memory
managers, why 'module set' rather than a single module...*

.. code-block:: c++

  JITSymbol findSymbol(const std::string Name) {
    std::string MangledName;
    raw_string_ostream MangledNameStream(MangledName);
    Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    return CompileLayer.findSymbol(MangledNameStream.str(), true);
  }

  void removeModule(ModuleHandle H) {
    CompileLayer.removeModuleSet(H);
  }

*To be done: describe findSymbol and removeModule -- why do we mangle? what's
the relationship between findSymbol and resolvers, why remove modules...*

*To be done: Conclusion, exercises (maybe a utility for a standalone IR JIT,
like a mini-LLI), feed to next chapter.*

Full Code Listing
=================

Here is the complete code listing for our running example, enhanced with
mutable variables and var/in support. To build this example, use:

.. code-block:: bash

    # Compile
    clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orc native` -O3 -o toy
    # Run
    ./toy

Here is the code:

.. literalinclude:: ../../examples/Kaleidoscope/BuildingAJIT/Chapter1/KaleidoscopeJIT.h
   :language: c++

`Next: Extending the KaleidoscopeJIT <BuildingAJIT2.html>`_

.. [1] Actually we use a cut-down version of KaleidoscopeJIT that makes a
       simplifying assumption: symbols cannot be re-defined. This will make it
       impossible to re-define symbols in the REPL, but will make our symbol
       lookup logic simpler. Re-introducing support for symbol redefinition is
       left as an exercise for the reader. (The KaleidoscopeJIT.h used in the
       original tutorials will be a helpful reference).

.. [2] +-----------------------+-----------------------------------------------+
       |         File          |               Reason for inclusion            |
       +=======================+===============================================+
       |   ExecutionEngine.h   | Access to the EngineBuilder::selectTarget     |
       |                       | method.                                       |
       +-----------------------+-----------------------------------------------+
       |                       | Access to the                                 |
       | RTDyldMemoryManager.h | RTDyldMemoryManager::getSymbolAddressInProcess|
       |                       | method.                                       |
       +-----------------------+-----------------------------------------------+
       |    CompileUtils.h     | Provides the SimpleCompiler class.            |
       +-----------------------+-----------------------------------------------+
       |   IRCompileLayer.h    | Provides the IRCompileLayer class.            |
       +-----------------------+-----------------------------------------------+
       |                       | Access the createLambdaResolver function,     |
       |   LambdaResolver.h    | which provides easy construction of symbol    |
       |                       | resolvers.                                    |
       +-----------------------+-----------------------------------------------+
       |  ObjectLinkingLayer.h | Provides the ObjectLinkingLayer class.        |
       +-----------------------+-----------------------------------------------+
       |       Mangler.h       | Provides the Mangler class for platform       |
       |                       | specific name-mangling.                       |
       +-----------------------+-----------------------------------------------+
       |   DynamicLibrary.h    | Provides the DynamicLibrary class, which      |
       |                       | makes symbols in the host process searchable. |
       +-----------------------+-----------------------------------------------+
