===============================
ORC Design and Implementation
===============================

Introduction
============

This document aims to provide a high-level overview of the ORC APIs and
implementation. Except where otherwise stated, all discussion applies to
the design of the APIs as of LLVM verison 9 (ORCv2).

Use-cases
=========

ORC aims to provide a modular API for building in-memory compilers,
including JIT compilers. There are a wide range of use cases for such
in-memory compilers. For example:

1. The LLVM tutorials use an in-memory compiler to execute expressions
compiled from a toy languge: Kaleidoscope.

2. The LLVM debugger, LLDB, uses a cross-compiling in-memory compiler for
expression evaluation within the debugger. Here, cross compilation is used
to allow expressions compiled within the debugger session to be executed on
the debug target, which may be a different device/architecture.

3. In high-performance JITs (e.g. JVMs, Julia) that want to make use of LLVM's
optimizations within an existing JIT infrastructure.

4. In interpreters and REPLs, e.g. Cling (C++) and the Swift interpreter.

By adoping a modular, library based design we aim to make ORC useful in as many
of these contexts as possible.

Features
========

ORC provides the following features:

- JIT-linking: Allows relocatable object files (COFF, ELF, MachO)[1]_ to be
  added to a JIT session. The objects will be loaded, linked, and made
  executable in a target process, which may be the same process that contains
  the JIT session and linker, or may be another process (even one running on a
  different machine or architecture) that communicates with the JIT via RPC.

- LLVM IR compilation: Off the shelf components (IRCompileLayer, SimpleCompiler,
  ConcurrentIRCompiler) allow LLVM IR to be added to a JIT session and made
  executable.

- Lazy compilation: ORC provides lazy-compilation stubs that can be used to
  defer compilation of functions until they are called at runtime.

- Custom compilers: Clients can supply custom compilers for each symbol that
  they define in their JIT session. ORC will run the user-supplied compiler when
  the a definition of a symbol is needed.

- Concurrent JIT'd code and concurrent compilation: Since most compilers are
  embarrassingly parallel ORC provides off-the-shelf infrastructure for running
  compilers concurrently and ensures that their work is done before allowing
  dependent threads of JIT'd code to proceed.

- Orthogonality and composability: Each of the features above can be used (or
  not) independently. It is possible to put ORC components together to make a
  non-lazy, in-process, single threaded JIT or a lazy, out-of-process,
  concurrent JIT, or anything in between.

LLJIT and LLLazyJIT
===================

While ORC is a library for building JITs it also provides two basic JIT
implementations off-the-shelf. These are useful both as replacements for
earlier LLVM JIT APIs (e.g. MCJIT), and as examples of how to build a JIT
class out of ORC components.

The LLJIT class supports compilation of LLVM IR and linking of relocatable
object files. All operations are performed eagerly on symbol lookup (i.e. a
symbol's definition is compiled as soon as you attempt to look up its address).

The LLLazyJIT extends LLJIT to add lazy compilation of LLVM IR. When an LLVM
IR module is added via the addLazyIRModule method, function bodies in that
module will not be compiled until they are first called.

Design Overview
===============

ORC's JIT'd program model aims to emulate the linking and symbol resolution
rules used by the static and dynamic linkers. This allows ORC to JIT LLVM
IR (which was designed for static compilation) naturally, including support
for linker-specific constructs like weak symbols, symbol linkage, and
visibility. To see how this works, imagine a program ``foo`` which links
against a pair of dynamic libraries: ``libA`` and ``libB``. On the command
line building this system might look like:


.. code-block:: bash

  $ clang++ -shared -o libA.dylib a1.cpp a2.cpp
  $ clang++ -shared -o libB.dylib b1.cpp b2.cpp
  $ clang++ -o myapp myapp.cpp -L. -lA -lB
  $ ./myapp

This would translate into ORC API calls on a "CXXCompilingLayer"
(with error-check omitted for brevity) as:

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


How and when the JIT compilation in this example occurs would depend on the
implementation of the hypothetical CXXCompilingLayer, but the linking rules
should be the same regardless. For example, if a1.cpp and a2.cpp both define a
function "foo" the API should generate a duplicate definition error. On the
other hand, if a1.cpp and b1.cpp both define "foo" there is no error (different
dynamic libraries may define the same symbol). If main.cpp refers to "foo", it
should bind to the definition in LibA rather than the one in LibB, since
main.cpp is part of the "main" dylib, and the main dylib links against LibA
before LibB.

Many JIT clients will have no need for this strict adherence to the usual
ahead-of-time linking rules and should be able to get by just fine by putting
all of their code in a single JITDylib. However, clients who want to JIT code
for languages/projects that traditionally rely on ahead-of-time linking (e.g.
C++) will find that this feature makes life much easier.

Symbol lookup in ORC serves two other important functions which we discuss in
more detail below: (1) It triggers compilation of the symbol(s) searched for,
and (2) it provides the synchronization mechanism for concurrent compilation.

When a lookup call is made, it searches for a *set* of requested symbols
(single symbol lookup is implemented as a convenience function on top of the
bulk-lookup APIs). The *materializers* for these symbols (usually compilers,
but in general anything that ultimately writes a usable definition into
memory) are collected and passed to the ExecutionSession's
dispatchMaterialization method. By performing lookups on multiple symbols at
once we ensure that the JIT knows about all required work for that query
up-front. By making the dispatchMaterialization function client configurable
we make it possible to execute the materializers on multiple threads
concurrently.

Under the hood, lookup operations are implemented in terms of query objects.
The first search for any given symbol triggers *materialization* of that symbol
and appends the query to the symbol table entry. Any subsequent lookup for that
symbol (lookups can be made from any thread at any time after the JIT is set up)
will simply append its query object to the list of queries waiting on that
symbol's definition. Once a definition has been materialized ORC will notify all
queries that are waiting on it, and once all symbols for a query have been
materialized the caller is notified (via a callback) that the query completed
successfully (the successful result is a map of symbol names to addresses). If
any symbol fails to materialize then all pending queries for that symbol are
notified of the failure.

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