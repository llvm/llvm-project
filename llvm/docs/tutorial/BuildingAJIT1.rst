=======================================================
Building a JIT: Starting out with KaleidoscopeJIT
=======================================================

.. contents::
   :local:

Chapter 1 Introduction
======================

**Warning: This tutorial is currently being updated to account for ORC API
changes. Only Chapter 1 is up-to-date.**

**Example code from Chapters 2 to 4 will compile and run, but has not been
updated**

Welcome to Chapter 1 of the "Building an ORC-based JIT in LLVM" tutorial. This
tutorial runs through the implementation of a JIT compiler using LLVM's
On-Request-Compilation (ORC) APIs. It begins with a simplified version of the
KaleidoscopeJIT class used in the
`Implementing a language with LLVM <LangImpl01.html>`_ tutorials and then
introduces new features like concurrent compilation, optimization, lazy
compilation and remote execution.

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

To provide input for our JIT we will use a lightly modified version of the
Kaleidoscope REPL from `Chapter 7 <LangImpl07.html>`_ of the "Implementing a
language in LLVM tutorial".

Finally, a word on API generations: ORC is the 3rd generation of LLVM JIT API.
It was preceded by MCJIT, and before that by the (now deleted) legacy JIT.
These tutorials don't assume any experience with these earlier APIs, but
readers acquainted with them will see many familiar elements. Where appropriate
we will make this connection with the earlier APIs explicit to help people who
are transitioning from them to ORC.

JIT API Basics
==============

The purpose of a JIT compiler is to compile code "on-the-fly" as it is needed,
rather than compiling whole programs to disk ahead of time as a traditional
compiler does. To support that aim our initial, bare-bones JIT API will have
just two functions:

1. void addModule(std::unique_ptr<Module> M) -- Make the given IR module
   available for execution.
2. Expected<JITSymbol> lookup() -- Search for pointers to
   symbols (functions or variables) that have been added to the JIT.

A basic use-case for this API, executing the 'main' function from a module,
will look like:

.. code-block:: c++

  std::unique_ptr<Module> M = buildModule();
  JIT J;
  J.addModule(*M);
  auto *Main = (int(*)(int, char*[]))J.lookup("main");.getAddress();
  int Result = Main();

The APIs that we build in these tutorials will all be variations on this simple
theme. Behind this API we will refine the implementation of the JIT to add
support for concurrent compilation, optimization and lazy compilation.
Eventually we will extend the API itself to allow higher-level program
representations (e.g. ASTs) to be added to the JIT.

KaleidoscopeJIT
===============

In the previous section we described our API, now we examine a simple
implementation of it: The KaleidoscopeJIT class [1]_ that was used in the
`Implementing a language with LLVM <LangImpl01.html>`_ tutorials. We will use
the REPL code from `Chapter 7 <LangImpl07.html>`_ of that tutorial to supply the
input for our JIT: Each time the user enters an expression the REPL will add a
new IR module containing the code for that expression to the JIT. If the
expression is a top-level expression like '1+1' or 'sin(x)', the REPL will also
use the lookup method of our JIT class find and execute the code for the
expression. In later chapters of this tutorial we will modify the REPL to enable
new interactions with our JIT class, but for now we will take this setup for
granted and focus our attention on the implementation of our JIT itself.

Our KaleidoscopeJIT class is defined in the KaleidoscopeJIT.h header. After the
usual include guards and #includes [2]_, we get to the definition of our class:

.. code-block:: c++

  #ifndef LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H
  #define LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H

  #include "llvm/ADT/StringRef.h"
  #include "llvm/ExecutionEngine/JITSymbol.h"
  #include "llvm/ExecutionEngine/Orc/CompileUtils.h"
  #include "llvm/ExecutionEngine/Orc/Core.h"
  #include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
  #include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
  #include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
  #include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
  #include "llvm/ExecutionEngine/SectionMemoryManager.h"
  #include "llvm/IR/DataLayout.h"
  #include "llvm/IR/LLVMContext.h"
  #include <memory>

  namespace llvm {
  namespace orc {

  class KaleidoscopeJIT {
  private:

    ExecutionSession ES;
    RTDyldObjectLinkingLayer ObjectLayer{ES, getMemoryMgr};
    IRCompileLayer CompileLayer{ES, ObjectLayer,
                                ConcurrentIRCompiler(getJTMB())};
    DataLayout DL{cantFail(getJTMB().getDefaultDataLayoutForTarget())};
    MangleAndInterner Mangle{ES, DL};
    ThreadSafeContext Ctx{llvm::make_unique<LLVMContext>()};

    static JITTargetMachineBuilder getJTMB() {
      return cantFail(JITTargetMachineBuilder::detectHost());
    }

    static std::unique_ptr<SectionMemoryManager> getMemoryMgr(VModuleKey) {
      return llvm::make_unique<SectionMemoryManager>();
    }

We begin with the ExecutionSession member, ``ES``, which provides context for
our running JIT'd code. It holds the string pool for symbol names, the global
mutex that guards the critical sections of JIT operations, error logging
facilities, and other utilities. For basic use cases such as this, a default
constructed ExecutionSession is all we will need. We will investigate more
advanced uses of ExecutionSession in later chapters. Following our
ExecutionSession we have two ORC *layers*: an RTDyldObjectLinkingLayer and an
IRCompileLayer. We will be talking more about layers in the next chapter, but
for now you can think of them as analogous to LLVM Passes: they wrap up useful
JIT utilities behind an easy to compose interface. The first layer, ObjectLayer,
is the foundation of our JIT: it takes in-memory object files produced by a
compiler and links them on the fly to make them executable. This
JIT-on-top-of-a-linker design was introduced in MCJIT, however the linker was
hidden inside the MCJIT class. In ORC we expose the linker so that clients can
access and configure it directly if they need to. In this tutorial our
ObjectLayer will just be used to support the next layer in our stack: the
CompileLayer, which will be responsible for taking LLVM IR, compiling it, and
passing the resulting in-memory object files down to the object linking layer
below. Our ObjectLayer is constructed with a reference to the ExecutionSession
and the getMemoryMgr utility function, which it uses to generate a new memory
manager for each object file as it is added. Next up is our CompileLayer, which
is initialized with a reference to the ExecutionSession, a reference to the
ObjectLayer (where it will send the objects produced by the compiler), and an IR
compiler instance. In this case we are using the ConcurrentIRCompiler class
which is constructed with a JITTargetMachineBuilder and can be called to compile
IR concurrently from several threads (though in this chapter we will only use
one).

Following the ExecutionSession and layers we have three supporting member
variables. The DataLayout, ``DL``; and MangleAndInterner, ``Mangle`` members are
used to support portable lookups based on IR symbol names (more on that when we
get to our ``lookup`` function below), and the ThreadSafeContext member,
``Ctx``, manages an LLVMContext that can be used while building IR Modules for
the JIT.

After that, we have two static utility functions. The ``getJTMB()`` function
returns a JITTargetMachineBuilder, which is a factory for building LLVM
TargetMachine instances that are used by the compiler. In this first tutorial we
will only need one (implicitly created) TargetMachine, but in future tutorials
that enable concurrent compilation we will need one per thread. This is why we
use a target machine builder, rather than a single TargetMachine. (note: Older
LLVM JIT APIs that did not support concurrent compilation were constructed with
a single TargetMachines). The ``getMemoryMgr()`` function constructs instances
of RuntimeDyld::MemoryManager, and is used by the linking layer to generate a
new memory manager for each object file.

.. code-block:: c++

  public:

    KaleidoscopeJIT() {
      ES.getMainJITDylib().setGenerator(
        cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(DL)));
    }

    const DataLayout &getDataLayout() const { return DL; }

    LLVMContext &getContext() { return *Ctx.getContext(); }

Next up we have our class constructor. Our members have already been
initialized, so the one thing that remains to do is to tweak the configuration
of the *JITDylib* that we will store our code in. We want to modify this dylib
to contain not only the symbols that we add to it, but also the symbols from
our REPL process as well. We do this by attaching a
``DynamicLibrarySearchGenerator`` instance using the
``DynamicLibrarySearchGenerator::GetForCurrentProcess`` method.

Following the constructor we have the ``getDataLayout()`` and ``getContext()``
methods. These are used to make data structures created and managed by the JIT
(especially the LLVMContext) available to the REPL code that will build our
IR modules.

.. code-block:: c++

  void addModule(std::unique_ptr<Module> M) {
    cantFail(CompileLayer.add(ES.getMainJITDylib(),
                              ThreadSafeModule(std::move(M), Ctx)));
  }

  Expected<JITEvaluatedSymbol> lookup(StringRef Name) {
    return ES.lookup({&ES.getMainJITDylib()}, Mangle(Name.str()));
  }

Now we come to the first of our JIT API methods: addModule. This method is
responsible for adding IR to the JIT and making it available for execution. In
this initial implementation of our JIT we will make our modules "available for
execution" by adding them to the CompileLayer, which will it turn store the
Module in the main JITDylib. This process will create new symbol table entries
in the JITDylib for each definition in the module, and will defer compilation of
the module until any of its definitions is looked up. Note that this is not lazy
compilation: just referencing a definition, even if it is never used, will be
enough to trigger compilation. In later chapters we will teach our JIT to defer
compilation of functions until they're actually called.  To add our Module we
must first wrap it in a ThreadSafeModule instance, which manages the lifetime of
the Module's LLVMContext (our Ctx member) in a thread-friendly way. In our
example, all modules will share the Ctx member, which will exist for the
duration of the JIT. Once we switch to concurrent compilation in later chapters
we will use a new context per module.

Our last method is ``lookup``, which allows us to look up addresses for
function and variable definitions added to the JIT based on their symbol names.
As noted above, lookup will implicitly trigger compilation for any symbol
that has not already been compiled. Our lookup method calls through to
`ExecutionSession::lookup`, passing in a list of dylibs to search (in our case
just the main dylib), and the symbol name to search for, with a twist: We have
to *mangle* the name of the symbol we're searching for first. The ORC JIT
components use mangled symbols internally the same way a static compiler and
linker would, rather than using plain IR symbol names. This allows JIT'd code
to interoperate easily with precompiled code in the application or shared
libraries. The kind of mangling will depend on the DataLayout, which in turn
depends on the target platform. To allow us to remain portable and search based
on the un-mangled name, we just re-produce this mangling ourselves using our
``Mangle`` member function object.

This brings us to the end of Chapter 1 of Building a JIT. You now have a basic
but fully functioning JIT stack that you can use to take LLVM IR and make it
executable within the context of your JIT process. In the next chapter we'll
look at how to extend this JIT to produce better quality code, and in the
process take a deeper look at the ORC layer concept.

`Next: Extending the KaleidoscopeJIT <BuildingAJIT2.html>`_

Full Code Listing
=================

Here is the complete code listing for our running example. To build this
example, use:

.. code-block:: bash

    # Compile
    clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o toy
    # Run
    ./toy

Here is the code:

.. literalinclude:: ../../examples/Kaleidoscope/BuildingAJIT/Chapter1/KaleidoscopeJIT.h
   :language: c++

.. [1] Actually we use a cut-down version of KaleidoscopeJIT that makes a
       simplifying assumption: symbols cannot be re-defined. This will make it
       impossible to re-define symbols in the REPL, but will make our symbol
       lookup logic simpler. Re-introducing support for symbol redefinition is
       left as an exercise for the reader. (The KaleidoscopeJIT.h used in the
       original tutorials will be a helpful reference).

.. [2] +-----------------------------+-----------------------------------------------+
       |         File                |               Reason for inclusion            |
       +=============================+===============================================+
       |        JITSymbol.h          | Defines the lookup result type                |
       |                             | JITEvaluatedSymbol                            |
       +-----------------------------+-----------------------------------------------+
       |       CompileUtils.h        | Provides the SimpleCompiler class.            |
       +-----------------------------+-----------------------------------------------+
       |           Core.h            | Core utilities such as ExecutionSession and   |
       |                             | JITDylib.                                     |
       +-----------------------------+-----------------------------------------------+
       |      ExecutionUtils.h       | Provides the DynamicLibrarySearchGenerator    |
       |                             | class.                                        |
       +-----------------------------+-----------------------------------------------+
       |      IRCompileLayer.h       | Provides the IRCompileLayer class.            |
       +-----------------------------+-----------------------------------------------+
       |  JITTargetMachineBuilder.h  | Provides the JITTargetMachineBuilder class.   |
       +-----------------------------+-----------------------------------------------+
       | RTDyldObjectLinkingLayer.h  | Provides the RTDyldObjectLinkingLayer class.  |
       +-----------------------------+-----------------------------------------------+
       |   SectionMemoryManager.h    | Provides the SectionMemoryManager class.      |
       +-----------------------------+-----------------------------------------------+
       |        DataLayout.h         | Provides the DataLayout class.                |
       +-----------------------------+-----------------------------------------------+
       |        LLVMContext.h        | Provides the LLVMContext class.               |
       +-----------------------------+-----------------------------------------------+
