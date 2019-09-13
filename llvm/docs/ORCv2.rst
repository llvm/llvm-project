===============================
ORC Design and Implementation
===============================

.. contents::
   :local:

Introduction
============

This document aims to provide a high-level overview of the design and
implementation of the ORC JIT APIs. Except where otherwise stated, all
discussion applies to the design of the APIs as of LLVM version 9 (ORCv2).

Use-cases
=========

ORC provides a modular API for building JIT compilers. There are a range
of use cases for such an API. For example:

1. The LLVM tutorials use a simple ORC-based JIT class to execute expressions
compiled from a toy language: Kaleidoscope.

2. The LLVM debugger, LLDB, uses a cross-compiling JIT for expression
evaluation. In this use case, cross compilation allows expressions compiled
in the debugger process to be executed on the debug target process, which may
be on a different device/architecture.

3. In high-performance JITs (e.g. JVMs, Julia) that want to make use of LLVM's
optimizations within an existing JIT infrastructure.

4. In interpreters and REPLs, e.g. Cling (C++) and the Swift interpreter.

By adopting a modular, library-based design we aim to make ORC useful in as many
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

  // Cast the entry point address to a function pointer.
  auto *Entry = (void(*)())EntrySym.getAddress();

  // Call into JIT'd code.
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
clang) that uses constructs like symbol linkage and visibility, and weak [3]_
and common symbol definitions.

To see how this works, imagine a program ``foo`` which links against a pair
of dynamic libraries: ``libA`` and ``libB``. On the command line, building this
program might look like:

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
      ES, []() { return std::make_unique<SectionMemoryManager>(); });
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

v  int Result = Main(...);

This example tells us nothing about *how* or *when* compilation will happen.
That will depend on the implementation of the hypothetical CXXCompilingLayer.
The same linker-based symbol resolution rules will apply regardless of that
implementation, however. For example, if a1.cpp and a2.cpp both define a
function "foo" then ORCv2 will generate a duplicate definition error. On the
other hand, if a1.cpp and b1.cpp both define "foo" there is no error (different
dynamic libraries may define the same symbol). If main.cpp refers to "foo", it
should bind to the definition in LibA rather than the one in LibB, since
main.cpp is part of the "main" dylib, and the main dylib links against LibA
before LibB.

Many JIT clients will have no need for this strict adherence to the usual
ahead-of-time linking rules, and should be able to get by just fine by putting
all of their code in a single JITDylib. However, clients who want to JIT code
for languages/projects that traditionally rely on ahead-of-time linking (e.g.
C++) will find that this feature makes life much easier.

Symbol lookup in ORC serves two other important functions, beyond providing
addresses for symbols: (1) It triggers compilation of the symbol(s) searched for
(if they have not been compiled already), and (2) it provides the
synchronization mechanism for concurrent compilation. The pseudo-code for the
lookup process is:

.. code-block:: none

  construct a query object from a query set and query handler
  lock the session
  lodge query against requested symbols, collect required materializers (if any)
  unlock the session
  dispatch materializers (if any)

In this context a materializer is something that provides a working definition
of a symbol upon request. Usually materializers are just wrappers for compilers,
but they may also wrap a jit-linker directly (if the program representation
backing the definitions is an object file), or may even be a class that writes
bits directly into memory (for example, if the definitions are
stubs). Materialization is the blanket term for any actions (compiling, linking,
splatting bits, registering with runtimes, etc.) that are required to generate a
symbol definition that is safe to call or access.

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

Transitioning from ORCv1 to ORCv2
=================================

Since LLVM 7.0, new ORC development work has focused on adding support for
concurrent JIT compilation. The new APIs (including new layer interfaces and
implementations, and new utilities) that support concurrency are collectively
referred to as ORCv2, and the original, non-concurrent layers and utilities
are now referred to as ORCv1.

The majority of the ORCv1 layers and utilities were renamed with a 'Legacy'
prefix in LLVM 8.0, and have deprecation warnings attached in LLVM 9.0. In LLVM
10.0 ORCv1 will be removed entirely.

Transitioning from ORCv1 to ORCv2 should be easy for most clients. Most of the
ORCv1 layers and utilities have ORCv2 counterparts [2]_ that can be directly
substituted. However there are some design differences between ORCv1 and ORCv2
to be aware of:

  1. ORCv2 fully adopts the JIT-as-linker model that began with MCJIT. Modules
     (and other program representations, e.g. Object Files)  are no longer added
     directly to JIT classes or layers. Instead, they are added to ``JITDylib``
     instances *by* layers. The ``JITDylib`` determines *where* the definitions
     reside, the layers determine *how* the definitions will be compiled.
     Linkage relationships between ``JITDylibs`` determine how inter-module
     references are resolved, and symbol resolvers are no longer used. See the
     section `Design Overview`_ for more details.

     Unless multiple JITDylibs are needed to model linkage relationsips, ORCv1
     clients should place all code in the main JITDylib (returned by
     ``ExecutionSession::getMainJITDylib()``). MCJIT clients should use LLJIT
     (see `LLJIT and LLLazyJIT`_).

  2. All JIT stacks now need an ``ExecutionSession`` instance. ExecutionSession
     manages the string pool, error reporting, synchronization, and symbol
     lookup.

  3. ORCv2 uses uniqued strings (``SymbolStringPtr`` instances) rather than
     string values in order to reduce memory overhead and improve lookup
     performance. See the subsection `How to manage symbol strings`_.

  4. IR layers require ThreadSafeModule instances, rather than
     std::unique_ptr<Module>s. ThreadSafeModule is a wrapper that ensures that
     Modules that use the same LLVMContext are not accessed concurrently.
     See `How to use ThreadSafeModule and ThreadSafeContext`_.

  5. Symbol lookup is no longer handled by layers. Instead, there is a
     ``lookup`` method on JITDylib that takes a list of JITDylibs to scan.

     .. code-block:: c++

       ExecutionSession ES;
       JITDylib &JD1 = ...;
       JITDylib &JD2 = ...;

       auto Sym = ES.lookup({&JD1, &JD2}, ES.intern("_main"));

  6. Module removal is not yet supported. There is no equivalent of the
     layer concept removeModule/removeObject methods. Work on resource tracking
     and removal in ORCv2 is ongoing.

For code examples and suggestions of how to use the ORCv2 APIs, please see
the section `How-tos`_.

How-tos
=======

How to manage symbol strings
############################

Symbol strings in ORC are uniqued to improve lookup performance, reduce memory
overhead, and allow symbol names to function as efficient keys. To get the
unique ``SymbolStringPtr`` for a string value, call the
``ExecutionSession::intern`` method:

  .. code-block:: c++

    ExecutionSession ES;
    /// ...
    auto MainSymbolName = ES.intern("main");

If you wish to perform lookup using the C/IR name of a symbol you will also
need to apply the platform linker-mangling before interning the string. On
Linux this mangling is a no-op, but on other platforms it usually involves
adding a prefix to the string (e.g. '_' on Darwin). The mangling scheme is
based on the DataLayout for the target. Given a DataLayout and an
ExecutionSession, you can create a MangleAndInterner function object that
will perform both jobs for you:

  .. code-block:: c++

    ExecutionSession ES;
    const DataLayout &DL = ...;
    MangleAndInterner Mangle(ES, DL);

    // ...

    // Portable IR-symbol-name lookup:
    auto Sym = ES.lookup({&ES.getMainJITDylib()}, Mangle("main"));

How to create JITDylibs and set up linkage relationships
########################################################

In ORC, all symbol definitions reside in JITDylibs. JITDylibs are created by
calling the ``ExecutionSession::createJITDylib`` method with a unique name:

  .. code-block:: c++

    ExecutionSession ES;
    auto &JD = ES.createJITDylib("libFoo.dylib");

The JITDylib is owned by the ``ExecutionEngine`` instance and will be freed
when it is destroyed.

A JITDylib representing the JIT main program is created by ExecutionEngine by
default. A reference to it can be obtained by calling
``ExecutionSession::getMainJITDylib()``:

  .. code-block:: c++

    ExecutionSession ES;
    auto &MainJD = ES.getMainJITDylib();

How to use ThreadSafeModule and ThreadSafeContext
#################################################

ThreadSafeModule and ThreadSafeContext are wrappers around Modules and
LLVMContexts respectively. A ThreadSafeModule is a pair of a
std::unique_ptr<Module> and a (possibly shared) ThreadSafeContext value. A
ThreadSafeContext is a pair of a std::unique_ptr<LLVMContext> and a lock.
This design serves two purposes: providing a locking scheme and lifetime
management for LLVMContexts. The ThreadSafeContext may be locked to prevent
accidental concurrent access by two Modules that use the same LLVMContext.
The underlying LLVMContext is freed once all ThreadSafeContext values pointing
to it are destroyed, allowing the context memory to be reclaimed as soon as
the Modules referring to it are destroyed.

ThreadSafeContexts can be explicitly constructed from a
std::unique_ptr<LLVMContext>:

  .. code-block:: c++

    ThreadSafeContext TSCtx(std::make_unique<LLVMContext>());

ThreadSafeModules can be constructed from a pair of a std::unique_ptr<Module>
and a ThreadSafeContext value. ThreadSafeContext values may be shared between
multiple ThreadSafeModules:

  .. code-block:: c++

    ThreadSafeModule TSM1(
      std::make_unique<Module>("M1", *TSCtx.getContext()), TSCtx);

    ThreadSafeModule TSM2(
      std::make_unique<Module>("M2", *TSCtx.getContext()), TSCtx);

Before using a ThreadSafeContext, clients should ensure that either the context
is only accessible on the current thread, or that the context is locked. In the
example above (where the context is never locked) we rely on the fact that both
``TSM1`` and ``TSM2``, and TSCtx are all created on one thread. If a context is
going to be shared between threads then it must be locked before any accessing
or creating any Modules attached to it. E.g.

  .. code-block:: c++

    ThreadSafeContext TSCtx(std::make_unique<LLVMContext>());

    ThreadPool TP(NumThreads);
    JITStack J;

    for (auto &ModulePath : ModulePaths) {
      TP.async(
        [&]() {
          auto Lock = TSCtx.getLock();
          auto M = loadModuleOnContext(ModulePath, TSCtx.getContext());
          J.addModule(ThreadSafeModule(std::move(M), TSCtx));
        });
    }

    TP.wait();

To make exclusive access to Modules easier to manage the ThreadSafeModule class
provides a convenience function, ``withModuleDo``, that implicitly (1) locks the
associated context, (2) runs a given function object, (3) unlocks the context,
and (3) returns the result generated by the function object. E.g.

  .. code-block:: c++

    ThreadSafeModule TSM = getModule(...);

    // Dump the module:
    size_t NumFunctionsInModule =
      TSM.withModuleDo(
        [](Module &M) { // <- Context locked before entering lambda.
          return M.size();
        } // <- Context unlocked after leaving.
      );

Clients wishing to maximize possibilities for concurrent compilation will want
to create every new ThreadSafeModule on a new ThreadSafeContext. For this
reason a convenience constructor for ThreadSafeModule is provided that implicitly
constructs a new ThreadSafeContext value from a std::unique_ptr<LLVMContext>:

  .. code-block:: c++

    // Maximize concurrency opportunities by loading every module on a
    // separate context.
    for (const auto &IRPath : IRPaths) {
      auto Ctx = std::make_unique<LLVMContext>();
      auto M = std::make_unique<LLVMContext>("M", *Ctx);
      CompileLayer.add(ES.getMainJITDylib(),
                       ThreadSafeModule(std::move(M), std::move(Ctx)));
    }

Clients who plan to run single-threaded may choose to save memory by loading
all modules on the same context:

  .. code-block:: c++

    // Save memory by using one context for all Modules:
    ThreadSafeContext TSCtx(std::make_unique<LLVMContext>());
    for (const auto &IRPath : IRPaths) {
      ThreadSafeModule TSM(parsePath(IRPath, *TSCtx.getContext()), TSCtx);
      CompileLayer.add(ES.getMainJITDylib(), ThreadSafeModule(std::move(TSM));
    }

How to Add Process and Library Symbols to the JITDylibs
=======================================================

JIT'd code typically needs access to symbols in the host program or in
supporting libraries. References to process symbols can be "baked in" to code
as it is compiled by turning external references into pre-resolved integer
constants, however this ties the JIT'd code to the current process's virtual
memory layout (meaning that it can not be cached between runs) and makes
debugging lower level program representations difficult (as all external
references are opaque integer values). A bettor solution is to maintain symbolic
external references and let the jit-linker bind them for you at runtime. To
allow the JIT linker to find these external definitions their addresses must
be added to a JITDylib that the JIT'd definitions link against.

Adding definitions for external symbols could be done using the absoluteSymbols
function:

  .. code-block:: c++

    const DataLayout &DL = getDataLayout();
    MangleAndInterner Mangle(ES, DL);

    auto &JD = ES.getMainJITDylib();

    JD.define(
      absoluteSymbols({
        { Mangle("puts"), pointerToJITTargetAddress(&puts)},
        { Mangle("gets"), pointerToJITTargetAddress(&getS)}
      }));

Manually adding absolute symbols for a large or changing interface is cumbersome
however, so ORC provides an alternative to generate new definitions on demand:
*definition generators*. If a definition generator is attached to a JITDylib,
then any unsuccessful lookup on that JITDylib will fall back to calling the
definition generator, and the definition generator may choose to generate a new
definition for the missing symbols. Of particular use here is the
``DynamicLibrarySearchGenerator`` utility. This can be used to reflect the whole
exported symbol set of the process or a specific dynamic library, or a subset
of either of these determined by a predicate.

For example, to load the whole interface of a runtime library:

  .. code-block:: c++

    const DataLayout &DL = getDataLayout();
    auto &JD = ES.getMainJITDylib();

    JD.setGenerator(DynamicLibrarySearchGenerator::Load("/path/to/lib"
                                                        DL.getGlobalPrefix()));

    // IR added to JD can now link against all symbols exported by the library
    // at '/path/to/lib'.
    CompileLayer.add(JD, loadModule(...));

Or, to expose a whitelisted set of symbols from the main process:

  .. code-block:: c++

    const DataLayout &DL = getDataLayout();
    MangleAndInterner Mangle(ES, DL);

    auto &JD = ES.getMainJITDylib();

    DenseSet<SymbolStringPtr> Whitelist({
        Mangle("puts"),
        Mangle("gets")
      });

    // Use GetForCurrentProcess with a predicate function that checks the
    // whitelist.
    JD.setGenerator(
      DynamicLibrarySearchGenerator::GetForCurrentProcess(
        DL.getGlobalPrefix(),
        [&](const SymbolStringPtr &S) { return Whitelist.count(S); }));

    // IR added to JD can now link against any symbols exported by the process
    // and contained in the whitelist.
    CompileLayer.add(JD, loadModule(...));

Future Features
===============

TBD: Speculative compilation. Object Caches.

.. [1] Formats/architectures vary in terms of supported features. MachO and
       ELF tend to have better support than COFF. Patches very welcome!

.. [2] The ``LazyEmittingLayer``, ``RemoteObjectClientLayer`` and
       ``RemoteObjectServerLayer`` do not have counterparts in the new
       system. In the case of ``LazyEmittingLayer`` it was simply no longer
       needed: in ORCv2, deferring compilation until symbols are looked up is
       the default. The removal of ``RemoteObjectClientLayer`` and
       ``RemoteObjectServerLayer`` means that JIT stacks can no longer be split
       across processes, however this functionality appears not to have been
       used.

.. [3] Weak definitions are currently handled correctly within dylibs, but if
       multiple dylibs provide a weak definition of a symbol then each will end
       up with its own definition (similar to how weak definitions are handled
       in Windows DLLs). This will be fixed in the future.
