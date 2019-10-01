===============================
ORC Design and Implementation
===============================

Introduction
============

This document aims to provide a high-level overview of the design and
implementation of the ORC JIT APIs. Except where otherwise stated, all
discussion applies to the design of the APIs as of LLVM verison 9 (ORCv2).

.. contents::
   :local:

Use-cases
=========

ORC provides a modular API for building JIT compilers. There are a range
of use cases for such an API:

1. The LLVM tutorials use a simple ORC-based JIT class to execute expressions
compiled from a toy languge: Kaleidoscope.

2. The LLVM debugger, LLDB, uses a cross-compiling JIT for expression
evaluation. In this use case, cross compilation allows expressions compiled
in the debugger process to be executed on the debug target process, which may
be on a different device/architecture.

3. In high-performance JITs (e.g. JVMs, Julia) that want to make use of LLVM's
optimizations within an existing JIT infrastructure.

4. In interpreters and REPLs, e.g. Cling (C++) and the Swift interpreter.

By adoping a modular, library-based design we aim to make ORC useful in as many
of these contexts as possible.

Features
========

ORC provides the following features:

- *JIT-linking* links relocatable object files (COFF, ELF, MachO) [1]_ into a
  target process an runtime. The target process may be the same process that
  contains the JIT session object and jit-linker, or may be another process
  (even one running on a different machine or architecture) that communicates
  with the JIT via RPC.

- *LLVM IR compilation*, which is provided by off the shelf components
  (IRCompileLayer, SimpleCompiler, ConcurrentIRCompiler) that make it easy to
  add LLVM IR to a JIT'd process.

- *Eager and lazy compilation*. By default, ORC will compile symbols as soon as
  they are looked up in the JIT session object (``ExecutionSession``). Compiling
  eagerly by default makes it easy to use ORC as a simple in-memory compiler for
  an existing JIT. ORC also provides a simple mechanism, lazy-reexports, for
  deferring compilation until first call.

- *Support for custom compilers and program representations*. Clients can supply
   custom compilers for each symbol that they define in their JIT session. ORC
   will run the user-supplied compiler when the a definition of a symbol is
   needed. ORC is actually fully language agnostic: LLVM IR is not treated
   specially, and is supported via the same wrapper mechanism (the
   ``MaterializationUnit`` class) that is used for custom compilers.

- *Concurrent JIT'd code* and *concurrent compilation*. JIT'd code may spawn
  multiple threads, and may re-enter the JIT (e.g. for lazy compilation)
  concurrently from multiple threads. The ORC APIs also support running multiple
  compilers concurrently, and provides off-the-shelf infrastructure to track
  dependencies on running compiles (e.g. to ensure that we never call into code
  until it is safe to do so, even if that involves waiting on multiple
  compiles).

- *Orthogonality* and *composability*: Each of the features above can be used (or
  not) independently. It is possible to put ORC components together to make a
  non-lazy, in-process, single threaded JIT or a lazy, out-of-process,
  concurrent JIT, or anything in between.

LLJIT and LLLazyJIT
===================

ORC provides two basic JIT classes off-the-shelf. These are useful both as
examples of how to assemble ORC components to make a JIT, and as replacements
for earlier LLVM JIT APIs (e.g. MCJIT).

The LLJIT class uses an IRCompileLayer and RTDyldObjectLinkingLayer to support
compilation of LLVM IR and linking of relocatable object files. All operations
are performed eagerly on symbol lookup (i.e. a symbol's definition is compiled
as soon as you attempt to look up its address). LLJIT is a suitable replacement
for MCJIT in most cases (note: some more advanced features, e.g.
JITEventListeners are not supported yet).

The LLLazyJIT extends LLJIT and adds a CompileOnDemandLayer to enable lazy
compilation of LLVM IR. When an LLVM IR module is added via the addLazyIRModule
method, function bodies in that module will not be compiled until they are first
called. LLLazyJIT aims to provide a replacement of LLVM's original (pre-MCJIT)
JIT API.

LLJIT and LLLazyJIT instances can be created using their respective builder
classes: LLJITBuilder and LLazyJITBuilder. For example, assuming you have a
module ``M`` loaded on an ThreadSafeContext ``Ctx``:

.. code-block:: c++

  // Try to detect the host arch and construct an LLJIT instance.
  auto JIT = LLJITBuilder().create();

  // If we could not construct an instance, return an error.
  if (!JIT)
    return JIT.takeError();

  // Add the module.
  if (auto Err = JIT->addIRModule(TheadSafeModule(std::move(M), Ctx)))
    return Err;

  // Look up the JIT'd code entry point.
  auto EntrySym = JIT->lookup("entry");
  if (!EntrySym)
    return EntrySym.takeError();

  auto *Entry = (void(*)())EntrySym.getAddress();

  Entry();

The builder clasess provide a number of configuration options that can be
specified before the JIT instance is constructed. For example:

.. code-block:: c++

  // Build an LLLazyJIT instance that uses four worker threads for compilation,
  // and jumps to a specific error handler (rather than null) on lazy compile
  // failures.

  void handleLazyCompileFailure() {
    // JIT'd code will jump here if lazy compilation fails, giving us an
    // opportunity to exit or throw an exception into JIT'd code.
    throw JITFailed();
  }

  auto JIT = LLLazyJITBuilder()
               .setNumCompileThreads(4)
               .setLazyCompileFailureAddr(
                   toJITTargetAddress(&handleLazyCompileFailure))
               .create();

  // ...

For users wanting to get started with LLJIT a minimal example program can be
found at ``llvm/examples/HowToUseLLJIT``.

Design Overview
===============

ORC's JIT'd program model aims to emulate the linking and symbol resolution
rules used by the static and dynamic linkers. This allows ORC to JIT
arbitrary LLVM IR, including IR produced by an ordinary static compiler (e.g.
clang) that uses constructs like symbol linkage and visibility, and weak and
common symbol definitions.

To see how this works, imagine a program ``foo`` which links against a pair
of dynamic libraries: ``libA`` and ``libB``. On the command line, building this
system might look like:

.. code-block:: bash

  $ clang++ -shared -o libA.dylib a1.cpp a2.cpp
  $ clang++ -shared -o libB.dylib b1.cpp b2.cpp
  $ clang++ -o myapp myapp.cpp -L. -lA -lB
  $ ./myapp

In ORC, this would translate into API calls on a "CXXCompilingLayer" (with error
checking omitted for brevity) as:

.. code-block:: c++

  ExecutionSession ES;
  RTDyldObjectLinkingLayer ObjLinkingLayer(
      ES, []() { return llvm::make_unique<SectionMemoryManager>(); });
  CXXCompileLayer CXXLayer(ES, ObjLinkingLayer);

  // Create JITDylib "A" and add code to it using the CXX layer.
  auto &LibA = ES.createJITDylib("A");
  CXXLayer.add(LibA, MemoryBuffer::getFile("a1.cpp"));
  CXXLayer.add(LibA, MemoryBuffer::getFile("a2.cpp"));

  // Create JITDylib "B" and add code to it using the CXX layer.
  auto &LibB = ES.createJITDylib("B");
  CXXLayer.add(LibB, MemoryBuffer::getFile("b1.cpp"));
  CXXLayer.add(LibB, MemoryBuffer::getFile("b2.cpp"));

  // Specify the search order for the main JITDylib. This is equivalent to a
  // "links against" relationship in a command-line link.
  ES.getMainJITDylib().setSearchOrder({{&LibA, false}, {&LibB, false}});
  CXXLayer.add(ES.getMainJITDylib(), MemoryBuffer::getFile("main.cpp"));

  // Look up the JIT'd main, cast it to a function pointer, then call it.
  auto MainSym = ExitOnErr(ES.lookup({&ES.getMainJITDylib()}, "main"));
  auto *Main = (int(*)(int, char*[]))MainSym.getAddress();

  int Result = Main(...);


This example tells us nothing about *how* or *when* compilation will happen.
That will depend on the implementation of the hypothetical CXXCompilingLayer,
but the linking rules will be the same regardless. For example, if a1.cpp and
a2.cpp both define a function "foo" the API should generate a duplicate
definition error. On the other hand, if a1.cpp and b1.cpp both define "foo"
there is no error (different dynamic libraries may define the same symbol). If
main.cpp refers to "foo", it should bind to the definition in LibA rather than
the one in LibB, since main.cpp is part of the "main" dylib, and the main dylib
links against LibA before LibB.

Many JIT clients will have no need for this strict adherence to the usual
ahead-of-time linking rules and should be able to get by just fine by putting
all of their code in a single JITDylib. However, clients who want to JIT code
for languages/projects that traditionally rely on ahead-of-time linking (e.g.
C++) will find that this feature makes life much easier.

Symbol lookup in ORC serves two other important functions, beyond basic lookup:
(1) It triggers compilation of the symbol(s) searched for, and (2) it provides
the synchronization mechanism for concurrent compilation. The pseudo-code for
the lookup process is:

.. code-block:: none

  construct a query object from a query set and query handler
  lock the session
  lodge query against requested symbols, collect required materializers (if any)
  unlock the session
  dispatch materializers (if any)

In this context a materializer is something that provides a working definition
of a symbol upon request. Generally materializers wrap compilers, but they may
also wrap a linker directly (if the program representation backing the
definitions is an object file), or even just a class that writes bits directly
into memory (if the definitions are stubs). Materialization is the blanket term
for any actions (compiling, linking, splatting bits, registering with runtimes,
etc.) that is requried to generate a symbol definition that is safe to call or
access.

As each materializer completes its work it notifies the JITDylib, which in turn
notifies any query objects that are waiting on the newly materialized
definitions. Each query object maintains a count of the number of symbols that
it is still waiting on, and once this count reaches zero the query object calls
the query handler with a *SymbolMap* (a map of symbol names to addresses)
describing the result. If any symbol fails to materialize the query immediately
calls the query handler with an error.

The collected materialization units are sent to the ExecutionSession to be
dispatched, and the dispatch behavior can be set by the client. By default each
materializer is run on the calling thread. Clients are free to create new
threads to run materializers, or to send the work to a work queue for a thread
pool (this is what LLJIT/LLLazyJIT do).

Top Level APIs
==============

Many of ORC's top-level APIs are visible in the example above:

- *ExecutionSession* represents the JIT'd program and provides context for the
  JIT: It contains the JITDylibs, error reporting mechanisms, and dispatches the
  materializers.

- *JITDylibs* provide the symbol tables.

- *Layers* (ObjLinkingLayer and CXXLayer) are wrappers around compilers and
  allow clients to add uncompiled program representations supported by those
  compilers to JITDylibs.

Several other important APIs are used explicitly. JIT clients need not be aware
of them, but Layer authors will use them:

- *MaterializationUnit* - When XXXLayer::add is invoked it wraps the given
  program representation (in this example, C++ source) in a MaterializationUnit,
  which is then stored in the JITDylib. MaterializationUnits are responsible for
  describing the definitions they provide, and for unwrapping the program
  representation and passing it back to the layer when compilation is required
  (this ownership shuffle makes writing thread-safe layers easier, since the
  ownership of the program representation will be passed back on the stack,
  rather than having to be fished out of a Layer member, which would require
  synchronization).

- *MaterializationResponsibility* - When a MaterializationUnit hands a program
  representation back to the layer it comes with an associated
  MaterializationResponsibility object. This object tracks the definitions
  that must be materialized and provides a way to notify the JITDylib once they
  are either successfully materialized or a failure occurs.

Handy utilities
===============

TBD: absolute symbols, aliases, off-the-shelf layers.

Laziness
========

Laziness in ORC is provided by a utility called "lazy-reexports". The aim of
this utility is to re-use the synchronization provided by the symbol lookup
mechanism to make it safe to lazily compile functions, even if calls to the
stub occur simultaneously on multiple threads of JIT'd code. It does this by
reducing lazy compilation to symbol lookup: The lazy stub performs a lookup of
its underlying definition on first call, updating the function body pointer
once the definition is available. If additional calls arrive on other threads
while compilation is ongoing they will be safely blocked by the normal lookup
synchronization guarantee (no result until the result is safe) and can also
proceed as soon as compilation completes.

TBD: Usage example.

Supporting Custom Compilers
===========================

TBD.

Low Level (MCJIT style) Use
===========================

TBD.

Future Features
===============

TBD: Speculative compilation. Object Caches.

.. [1] Formats/architectures vary in terms of supported features. MachO and
       ELF tend to have better support than COFF. Patches very welcome!