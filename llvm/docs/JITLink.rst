====================================
JITLink and ORC's ObjectLinkingLayer
====================================

.. contents::
   :local:

Introduction
============

This document aims to provide a high-level overview of the design and API
of the JITLink library. It assumes some familiarity with the concepts of
linking and relocatable object files, but should not require deep expertise.
If you know what a section, symbol, and relocation are you should find this
document accessible. If it is not, please submit a patch (:doc:`Contributing`)
or file a bug (:doc:`HowToSubmitABug`).

JITLink is a library for :ref:`JIT Linking`. It was built to support the ORC
JIT APIs and is most commonly accesed via ORC's ObjectLinkingLayer API. JITLink
was developed with the aim of supporting the full set of features provided by
each object format; including static initializers, exception handling, thread
local variables, runtime registration, and more. Supporting these features
enables ORC to execute code generated from source languages which rely on these
features (e.g. C++ requires static initializers to support static constructors,
eh-frame registration for exceptions, and TLV support for thread locals; Swift
and Objective-C require language runtime registration for many features). For
some object format features support is provided entirely within JITLink, and for
others it is provided in cooperation with the ORC runtime.

JITLink aims to support the following features, some of which are still under
development:

1. Cross-process and cross-architecture linking of single relocatable objects.

2. Support for all object format features.

3. Open linker data structure (``LinkGraph``) and pass system.

JITLink and ObjectLinkingLayer
==============================

``ObjectLinkingLayer`` is an ORC Layer that enables objects to be added to a
``JITDylib``, or emitted from some higher level program representation, and
linked into an executor process (which may be the same process) on request.
Either way, when an object is emitted the ObjectLinkingLayer constructs a
``LinkGraph`` (see :ref:`Constructing LinkGraphs`) and calls JITLink's
``link`` function.

``ObjectLinkingLayer`` provides a plugin API,
:literal:`ObjectLinkingLayer::Plugin`, which users can subclass in order to
inspect and modify `LinkGraph`s at link time, and react to important JIT events
(such as an object being emitted into target memory). This enables many features
and optimizations that were not possible under MCJIT or RuntimeDyld.

ObjectLinkingLayer Plugins
--------------------------

The ``ObjectLinkingLayer::Plugin`` class  provides the following  methods:

* ::

    virtual void modifyPassConfig(MaterializationResponsibility &MR,
                                  const Triple &TT,
                                  jitlink::PassConfiguration &Config)``

  Called each time a LinkGraph is about to be linked. Override to install
  custom JITLink *Passes* to run during the link process for this graph.

* ``virtual void notifyLoaded(MaterializationResponsibility &MR)``

  Called before the link begins. Override to set up initial state if needed.

* ``virtual Error notifyEmitted(MaterializationResponsibility &MR)``

  Called after link is complete and code has been emitted to the executor
  process. Override to finalize state if needed.

* ``virtual Error notifyFailed(MaterializationResponsibility &MR)``

  Called if link fails. Override to handle failure if needed.

* ``virtual Error notifyRemovingResources(ResourceKey K)``

  Called if/when a request is made to remove resouces for key *K*. Override to
  free any resources that this plugin allocated for objects associated with *K*.
  Note: Either both of ``notifyRemovingResources`` and
  ``notifyTransferringResources`` should be implemented, or neither should be.
  Implementing one but not the other will lead to resource management bugs.

* ::

    virtual void notifyTransferringResources(ResourceKey DstKey,
                                             ResourceKey SrcKey)

  Called if/when a request is made to reassociate resources from *SrcKey* to
  *DstKey*. Override to update the plugin's resource tracking maps (if any).
  Note: Either both of ``notifyRemovingResources`` and
  ``notifyTransferringResources`` should be implemented, or neither should be.
  Implementing one but not the other will lead to resource management bugs.

Plugin instances are added to an ``ObjectLinkingLayer`` by
calling the ``addPlugin`` method [1]_:

.. code-block:: c++

  // Plugin class to print the set of defined symbols in an object when that
  // object is linked.
  class MyPlugin : public ObjectLinkingLayer::Plugin {
  public:

    // Add passes to print the set of defined symbols after dead-stripping.
    virtual void modifyPassConfig(MaterializationResponsibility &MR,
                                 const Triple &TT,
                                 jitlink::PassConfiguration &Config) {
      Config.PrePrunePasses.push_back(

    }

    // JITLink pass to print all defined symbols in G.
    Error printAllSymbols(LinkGraph &G) {
      for (auto *Sym : G.defined_symbols())
        if (Sym->hasName())
          dbgs() << Sym->getName() << "\n";
      return Error::success();
    }
  };

  // Create our LLJIT instance using a custom object linking layer setup.
  // This gives us a chance to install our plugin.
  auto J = ExitOnErr(LLJITBuilder()
             .setObjectLinkingLayerCreator(
               [](ExecutionSession &ES, const Triple &T) {
                 // Manually set up the ObjectLinkingLayer for our LLJIT
                 // instance.
                 auto OLL = std::make_unique<ObjectLinkingLayer>(
                     ES, std::make_unique<jitlink::InProcessMemoryManager>());

                 // Install our plugin:
                 OLL->addPlugin(std::make_unique<MyPlugin>();

                 return OLL;
               })
             .create());

  // Add an object to the JIT. Nothing happens here: linking isn't triggered
  // until we look up some symbol in our object.
  ExitOnErr(J->addObject(loadFromDisk("main.o")));

  // Plugin triggers here when our lookup of main triggers linking of main.o
  auto MainSym = J->lookup("main");

LinkGraph
=========

JITLink maps all relocatable object formats to a generic ``LinkGraph`` type
that is designed to make linking fast and easy (``LinkGraph`` instances can
also be created manually. See :ref:`Constructing LinkGraphs`).

Relocatable object formats (e.g. COFF, ELF, MachO) differ in their details,
but share a common goal: to represent machine level code and data with
annotations that allow them to be relocated in a virtual address space. To
this end they usually contain names (symbols) for content defined inside the
file or externally, chunks of content that must be moved as a unit (sections
or subsections, depending on the format), and annotations describing how to
patch content based on the final address of some target symbol/section
(relocations).

At a high level, the ``LinkGraph`` type represents these concepts as a decorated
graph. Nodes in the graph represent symbols and content, and edges represent
relocations. Each of the elements of the graph is listed here:

* ``Addressable`` -- A node in the link graph that can be assigned an address
  in the executor process's virtual address space.

  Absolute and external symbols are represented using plain ``Addressable``
  instances. Content defined inside the object file is represented as a subclass
  of ``Addressable``: ``Block``.

* ``Block`` -- An ``Addressable`` node that has ``Content`` (or is marked as
  zero-filled), a parent ``Section``, a ``Size``, an ``Alignment`` (and an
  ``AlignmentOffset``), and a list of ``Edge``s.

  Blocks provide a container for binary content which must remain contiguous in
  the target address space (they are a unit for layout purposes). Many
  interesting low level operations on ``LinkGraph``s involve inspecting or
  mutating block content or edges.

  * ``Content`` is represented as an ``llvm::StringRef``. Content is only
    available for non-zero-filled blocks (use ``isZeroFill`` to check).

  * ``Section`` is represented as a ``Section&`` reference. The ``Section``
    class is described in more detail below.

  * ``Size`` is represented as a ``size_t``, and is accessible via the
    ``size_t getSize()`` method for both content and zero-filled blocks.

  * ``Alignment`` is represented as a ``uint64_t``, and represents the minimum
    alignment requirement (in bytes) of the start of the block.

  * ``AlignmentOffset`` is represented as a ``uint64_t``, and represents the
    offset from the alignment required for the start of the block. This is
    required to support blocks whose minimum alignment requirement comes from
    data at some non-zero offset from the start of the block. E.g. if a block
    consists of a single byte (with byte alignment) followed by a uint64_t (with
    8-byte alignment), then the block will have 8-byte alignment with an
    alignment offset of 7.

  * list of ``Edge``s is represented as a vector of ``Edge`` instances. The
    ``Edge`` class is described in more detail below.

* ``Symbol`` -- An offset from an ``Addressable`` (often a ``Block``), with an
  optional ``Name``, a ``Linkage``, a ``Scope``, a ``Callable`` flag, and a
  ``Live`` flag.

  Symbols make it possible to name content (blocks and addressables are
  anonymous), or target content with an ``Edge``.

  * ``Name`` is represented as an ``llvm::StringRef`` (equal to
    ``llvm::StringRef()`` if the symbol has no name).

  * ``Linkage`` is one of *Strong* or *Weak*. The ``JITLinkContext`` can use
    this flag to determine whether this symbol definition should be kept or
    dropped.

  * ``Scope`` is one of *Default*, *Hidden*, or *Local*. The ``JITLinkContext``
    can use this to determine who should be able to see the symbol. A symbol
    with default scope should be globally visible. A symbol with hidden scope
    should be visible to other definitions within the same simulated dylib
    (e.g. ORC ``JITDylib``) or executable, but not from elsewhere. A symbol
    with local scope should only be visible within the current ``LinkGraph``.

  * ``Callable`` is set to true if this symbol can be called. This can be used
    to automate the introduction of call-stubs for lazy compilation.

  * ``Live`` should be set before pruning (see :ref:`Generic Link Algorithm`)
    for the root symbols that a client wants to link. JITLink's dead-stripping
    algorithm will propagate liveness flags through the graph to all reachable
    symbols before deleting any symbols (and blocks) that are not marked live.

* ``Edge`` -- A quad of an ``Offset`` (implicitly from the start of the
  containing ``Block``), a ``Kind`` (describing the relocation type), a
  ``Target``, and an ``Addend``.

  Edges represent relocations, and occasionally other relationships, between
  blocks and symbols.

  * ``Offset`` is an offset from the start of the ``Block`` containing the
    ``Edge``.

  * ``Kind`` is a relocation type -- it describes what kinds of changes (if
    any) should be made to block content at the given ``Offset`` based on the
    address of the ``Target``.

  * ``Target`` is a pointer to a ``Symbol``, representing whose address is
    relevant to the fixup calculation specified by the edge's ``Kind``.

  * ``Addend`` is a constant whose interpretation is determined by the edge's
    ``Kind``.

* ``Section`` -- A set of ``Symbol`` instances, plus a set of ``Block``
  instances, with a ``Name``, a set of ``ProtectionFlags``, and an ``Ordinal``.

  Sections make it easy to iterate over the symbols or blocks associated with
  a particular section in the source object file.

  * ``blocks()`` returns an iterator over the set of blocks defined in the
    section (as ``Block*``s).

  * ``symbols()`` returns an iterator over the set of symbols defined in the
    section (as ``Symbol*``s).

  * ``Name`` is represented as an ``llvm::StringRef``.

  * ``ProtectionFlags`` are represented as a sys::Memory::ProtectionFlags enum.
    They describe whether the section is readable, writable, executable, or
    some combination of these. The most common combinations are ``RW-`` for
    writable data, ``R--`` for constant data, and ``R-X`` for code.

  * ``SectionOrdinal`` is a number that orders te section relative to others.
    It is usually used to preserve section order within a segment (a set of
    sections with the same memory protections) when laying out memory.

For the graph-theorists: The ``LinkGraph`` is bipartite, with one set of
``Symbol`` nodes and one set of ``Addressable`` nodes. Each ``Symbol`` node has
one (implicit) edge to its target ``Addressable``. Each ``Block`` has a set of
edges (possibly empty, represented as ``Edge`` instances) back to elements of
the ``Symbol`` set. For convenience and performance of common algorithms,
symbols and blocks are further grouped into ``Sections``.

The ``LinkGraph`` itself provides operations for constructing, removing, and
iterating over sections, symbols, and blocks. It also provides metadata
and utilities relevant to the linking process:

* Graph element operations

  * ``sections`` returns an iterator over all sections in the graph.

  * ``findSectionByName`` returns a pointer to the section with the given
    name (as a ``Section*``) if it exists, otherwise returns a nullptr.

  * ``blocks`` returns an iterator over all blocks in the graph (across all
    sections).

  * ``defined_symbols`` returns an iterator over all defined symbols in the
    graph (across all sections).

  * ``external_symbols`` returns an iterator over all external symbols in the
    graph.

  * ``absolute_symbols`` returns an iterator over all absolute symbols in the
    graph.

  * ``createSection`` creates a section with a given name and protection flags.

  * ``createContentBlock`` creates a block with the given initial content,
    parent section, address, alignment, and alignment offset.

  * ``createZeroFillBlock`` creates a zero-fill block with the given size,
    parent section, address, alignment, and alignment offset.

  * ``addExternalSymbol`` creates a new addressable and symbol with a given
    name, size, and linkage.

  * ``addAbsoluteSymbol`` creates a new addressable and symbol with a given
    name, address, size, linkage, scope, and liveness.

  * ``addCommonSymbol`` convenience function for creating a zero-filled block
    and weak symbol with a given name, scope, section, initial address, size,
    alignment and liveness.

  * ``addAnonymousSymbol`` creates a new anonymous symbol for a given block,
    offset, size, callable-ness, and liveness.

  * ``addDefinedSymbol`` creates a new symbol for a given block with a name,
    offset, size, linkage, scope, callable-ness and liveness.

  * ``makeExternal`` transforms a formerly defined symbol into an external one
    by creating a new addressable and pointing the symbol at it. The existing
    block is not deleted, but can be manually removed (if unreferenced) by
    calling ``removeBlock``. All edges to the symbol remain valid, but the
    symbol must now be defined outside this ``LinkGraph``.

  * ``removeExternalSymbol`` removes an external symbol and its target
    addressable. The target addressable must not be referenced by any other
    symbols.

  * ``removeAbsoluteSymbol`` removes an absolute symbol and its target
    addressable. The target addressable must not be referenced by any other
    sybols.

  * ``removeDefinedSymbol`` removes a defined symbol, but *does not* remove
    its target block.

  * ``removeBlock`` removes the given block.

  * ``splitBlock`` split a given block in two at a given index (useful where
    it is known that a block contains decomposable records, e.g. CFI records
    in an eh-frame section).

* Graph utility operations

  * ``getName`` returns the name of this graph, which is usually based on the
    name of the input object file.

  * ``getTargetTriple`` returns an `llvm::Triple` for the executor process.

  * ``getPointerSize`` returns the size of a pointer (in bytes) in the executor
    process.

  * ``getEndinaness`` returns the endianness of the executor process.

  * ``allocateString`` copies data from a given ``llvm::Twine`` into the
    link graph's internal allocator. This can be used to ensure that content
    created inside a pass outlives that pass's execution.

Generic Link Algorithm
======================

JITLink provides a generic link algorithm which can be extended / modified at
certain points by the introduction of JITLink :ref:`Passes`:

#. Phase 1

   #. Run pre-prune passes

      These passes are called on the graph before it is pruned. At this stage
      LinkGraph nodes still have their original vmaddrs. A mark-live pass
      (supplied by the ``JITLinkContext``) will be run at the end of this
      sequence to mark the initial set of live symbols.

      Notable use cases: marking nodes live, accessing/copying graph data that
      well be pruned (e.g. metadata that's important for the JIT, but not needed
      for the link process).

   #. Prune (dead-strip) LinkGraph

      Removes all symbols and blocks not reachable from the initial set of live
      symbols.

      This allows JITLink to remove unreachable symbols / content, including
      overridden weak and redundant ODR definitions.

   #. Run post-prune passes

      These passes are run on the graph after dead-stripping, but before memory
      is allocated or nodes assigned their final target vmaddrs.

      Passes run at this stage benefit from pruning, as dead functions and data
      have been stripped from the graph. However new content call still be added
      to the graph, as target and working memory have not been allocated yet.

      Notable use cases: Building Global Offset Table (GOT), Procedure Linkage
      Table (PLT), and Thread Local Variable (TLV) entries.

   #. Sort blocks into segments

      Sorts all blocks by ordinal and then address. Collects sections with
      matching permissions into segments and computes the size of these
      segments for memory allocation.

   #. Allocate segment memory, update node addresses

      Calls the ``JITLinkContext``'s ``JITLinkMemoryManager`` to allocate both
      working and target memory for the graph, then updates all node addresses
      to their assigned target address.

      Note: This step only updates the addresses of nodes defined in this graph.
      External symbols will still have null addresses.

   #. Run post-allocation passes

      These passes are run on the graph after working and target memory have
      been allocated, but before the ``JITLinkContext`` is notified of the
      final addresses of the symbols in the graph. This gives these passes a
      chance to set up data structures associated with target addresses before
      any JITLink clients (especially ORC queries for symbol resolution) can
      attempt to access them.

      Notable use cases: Setting up mappings between target addresses and
      JIT data structures, such as a mapping between ``__dso_handle`` and
      ``JITDylib*``.

   #. Notify the ``JITLinkContext`` of the assigned symbol addresses.

      Calls ``JITLinkContext::notifyResolved`` on the link graph, allowing
      clients to react to the symbol address assignments made for this graph.
      In ORC this is used to notify any pending queries for *resolved* symbols,
      including pending queries from concurrently running JITLink instances that
      have reached the next step and are waiting on the address of a symbol in
      this graph to proceed with their link.

   #. Identify external symbols and resolve asynchronously.

      Calls the ``JITLinkContext`` to resolve the target address of any external
      symbols in the graph. This step is asynchronous -- JITLink will pack the
      link state into a *continuation* to be run once the symbols are resolved.

      This is the final step of Phase 1.

#. Phase 2.

   #. Apply external symbol resolution results.

      This updates the addresses of all external symbols. At this point all
      nodes in the graph have their final target addresses, however node
      content still points back to the original data in the object file.

   #. Run pre-fixup passes.

      These passes are called on the graph after all nodes have been assigned
      their final target addresses, but before node content is copied into
      working memory and fixed up. Passes run at this stage can make late
      optimizations to the graph and content based on address layout.

      Notable use cases: GOT and PLT relaxation, where GOT and PLT acceses are
      bypassed for fixp targets that are directly accessible under the assigned
      memory layout.

   #. Copy block content to working memory and apply fixups.

      Copies all block content into allocated working memory (following the
      target layout) and applies fixups. Graph blocks are updated to point at
      the fixed up content.

   #. Run post-fixup passes.

      These passes are called on the graph after fixups have been applied and
      blocks updated to point to the fixed up content.

      Post-fixup passes can inspect blocks contents to see the exact bytes that
      will be copied to the assigned target addresses.

   #. Finalize memory asynchronously.

      Calls the ``JITLinkMemoryManager`` to copy working memory to the executor
      process and apply the requested permissions. This step is asynchronous --
      JITLink will pack the link state into a *continuation* to be run once
      memory has been copied and protected.

      This is the final step of Phase 2.

#. Phase 3.

   #. Notify the context that the graph has been emitted.

      Calls ``JITLinkContext::notifyFinalized`` and hands off the
      ``JITLinkMemoryManager::Allocation`` object for this graph's memory
      allocation. This allows the context to track/hold memory allocations and
      react to the newly emitted definitions. In ORC this is used to update the
      ``ExecutionSession``s dependence graph, which may result in these symbols
      (and possibly others) becoming *Ready* if all of their dependencies have
      also been emitted.

Passes
------

JITLink passes are ``std::function<Error(LinkGraph&)>`` instances. They are free
to inspect and modify the given ``LinkGraph``, subject to the constraints of
whatever phase they are running in (see :ref:`Generic Link Algorithm`). If a
pass returns ``Error::success()`` then linking continues. If a pass returns
a failure value then linking is stopped and the ``JITLinkContext`` is notified
that the link failed.

Passes may be used by both JITLink backends (e.g. MachO/x86-64 implements GOT
and PLT construction as a pass), and clients.

In combination with the open ``LinkGraph`` API, JITLink passes enable the
implementation of powerful new features. For example:

* Relaxation optimizations -- A pre-fixup pass can inspect GOT accesses and
  PLT calls and identify situations where their ultimate targets are close
  enough to be accessed directly. In this case the pass can rewrite the
  instruction stream of the containing block and update the fixup edges to
  make the access direct.

  Code for this looks like:

.. code-block:: c++

  Error relaxGOTEdges(LinkGraph &G) {
    for (auto *B : G.blocks())
      for (auto &E : B->edges())
        if (E.getKind() == x86_64::GOTLoad) {
          auto &GOTTarget = getGOTEntryTarget(E.getTarget());
          if (isInRange(B.getFixupAddress(E), GOTTarget)) {
            // Rewrite B.getContent() at fixup address from
            // MOVQ to LEAQ

            // Update edge target and kind.
            E.setTarget(GOTTarget);
            E.setKind(x86_64::PCRel32);
          }
        }

    return Error::success();
  }

* Metadata registration -- Post allocation passes can be used to record the
  address range of sections in the target. This can be used to register the
  metadata (e.g exception handling frames, language metadata) in the target
  once memory has been finalized.

.. code-block:: c++

  Error registerEHFrameSection(LinkGraph &G) {
    if (auto *Sec = G.findSectionByName("__eh_frame")) {
      SectionRange SR(*Sec);
      registerEHFrameSection(SR.getStart(), SR.getEnd());
    }

    return Error::success();
  }

* Record call sites for later mutation -- A post-allocation pass can record
  the call sites of all calls to a particular function, allowing those call
  sites to be updated later at runtime (e.g. for instrumentation, or to
  enable the function to be lazily compiled but still called directly after
  compilation).

.. code-block:: c++

  StringRef FunctionName = "foo";
  std::vector<JITTargetAddress> CallSitesForFunction;

  auto RecordCallSites =
    [&](LinkGraph &G) -> Error {
      for (auto *B : G.blocks())
        for (auto &E : B.edges())
          if (E.getKind() == CallEdgeKind &&
              E.getTarget().hasName() &&
              E.getTraget().getName() == FunctionName)
            CallSitesForFunction.push_back(B.getFixupAddress(E));
      return Error::success();
    };

Memory Management with JITLinkMemoryManager
-------------------------------------------

JIT linking requires allocation of two kinds of memory: working memory in the
JIT process and target memory in the execution process (these processes and
memory allocations may be one and the same, depending on how the user wants
to build their JIT). It also requires that these allocations conform to the
requested code model in the target process (e.g. MachO/x86-64's Small code
model requires that all code and data for a simulated dylib be allocated within
4Gb). Finally, it is natural to make the memory manager responsible for
transferring memory to the target address space and applying memory protections,
since the memory manager must know how to communicate with the executor, and
since sharing and protection assignment can often be efficiently managed (in
the common case of running across processes on the same machine for security)
via the host operating system's virtual memory management APIs.

To satisfy these requirements ``JITLinkMemoryManager`` adopts the following
design: The memory manager itself has just one virtual method that returns a
``JITLinkMemoryManager::Allocation``:

.. code-block:: c++

  virtual Expected<std::unique_ptr<Allocation>>
  allocate(const JITLinkDylib *JD, const SegmentsRequestMap &Request) = 0;

This method takes a ``JITLinkDylib*`` representing the target simulated
dylib, and the full set of sections that must be allocated for this object.
``JITLinkMemoryManager`` implementations can (optionally) use the ``JD``
argument to manage a per-simulated-dylib memory pool (since code model
constraints are typically imposed on a per-dylib basis, and not across
dylibs) [2]_. The ``Request`` argument, by describing all sections in the current
object up-front, allows the implementer to allocate those sections as a
single slab, either within a pre-allocated per-jitdylib pool or directly
from system memory.

All subsequent operations are provided by the
``JITLinkMemoryManager::Allocation`` interface:

* ``virtual MutableArrayRef<char> getWorkingMemory(ProtectionFlags Seg)``

  Should be overriden to return the address in working memory of the segment
  with the given protection flags.

* ``virtual JITTargetAddress getTargetMemory(ProtectionFlags Seg)``

  Should be overriden to return the address in the executor's address space of
  the segment with the given protection flags.

* ``virtual void finalizeAsync(FinalizeContinuation OnFinalize)``

  Should be overridden to copy the contents of working memory to the target
  address space and apply memory protections for all segments. Where working
  memory and target memory are separate, this method should deallocate the
  working memory.

* ``virtual Error deallocate()``

  Should be overriden to deallocate memory in the target address space.

JITLink provides a simple in-process implementation of this interface:
``InProcessMemoryManager``. It allocates pages once and re-uses them as both
working and target memory.

ORC provides a cross-process ``JITLinkMemoryManager`` based on an ORC-RPC-based
implementation of the ``orc::TargetProcessControl`` API:
``OrcRPCTPCJITLinkMemoryManager``. This API uses TargetProcessControl API calls
to allocate and manage memory in a remote process. The underlying communication
channel is determined by the ORC-RPC channel type. Common options include unix
sockets or TCP.

JITLinkMemoryManager and Security
---------------------------------

JITLink's ability to link JIT'd code for a separate executor process can be
used to improve the security of a JIT system: The executor process can be
sandboxed, run within a VM, or even run on a fully separate machine.

JITLink's memory manager interface is flexible enough to allow for a range
of trade-offs between performance and security. E.g. On a system where code
pages must be signed (preventing code from being updated) the memory manager
can drop working memory pages after linking to free up memory in the process
running JITLink. On a system that allows RWX pages, the memory manager may
allocate code-pages this way to enable code modification without further
overhead. Finally, if RWX pages are not permitted but dual-virtual-mappings of
physical memory pages are, then the memory manager can dual map physical code
pages as RW- in the JITLink process and R-X in the executor process, allowing
modification from the JITLink process but not from the executor at the cost
of extra administrative overhead for the dual mapping.

Error Handling
--------------

JITLink makes extensive use of the ``llvm::Error`` type (see the error handling
section of :doc:`ProgrammersManual` for details). The link process itself,
passes, the memory manager interface, and operations on the ``JITLinkContext``
are all permitted to fail. Link graph construction utilities (especially
parsers for object formats) are encouraged to validate input, and validate
fixups (e.g. with range checks) before application.

At the moment, any error will halt the link process and notify the context
of failure. In ORC, reported failures are propagated to and queries pending on
definitions provided by the failing link, and also through edges of the
dependence graph to any queries waiting on dependent symbols.

Connection to the ORC Runtime
=============================

The ORC Runtime (currently under development) aims to provide runtime support
for advanced JIT features, including object format features that require
non-trivial action in the executor (e.g. running initializers, managing thread
local storage, registering with language runtimes, ...).

ORC Runtime support for object format features typically requires cooperation
between the runtime (which executes in the executor process) and JITLink (which
can inspect LinkGraphs to determine what actions must be taken). For example:
Execution of MachO static initializers in the ORC runtime is performed by the
``jit_dlopen`` function, which calls back to the JIT process to ask for the
list of ``__mod_init`` sections to walk. This list is collated by the
``MachOPlatformPlugin``, which installs a pass to record this information for
each object as it's linked into the target.

Constructing LinkGraphs
=======================

Clients usually access and manipulate ``LinkGraph`` instances that were created
for them by an ``ObjectLinkingLayer`` instance, but they can be created manually:

#. By directly constructing and populating a ``LinkGraph`` instance.

#. By using the `createLinkGraph`` family of functions to create a ``LinkGraph``
   from an in-memory buffer containing an object file. This is how
   ``ObjectLinkingLayer`` usually creates ``LinkGraphs``.

  #. ``createLinkGraph_<Object-Format>_<Architecture>`` can be used when
      both the object format and achitecture are known ahead of time.

  #. ``createLinkGraph_<Object-Format>`` can be used when the object format is
     known ahead of time, but the architecture is not. In this case the
     architecture will be determined by inspection of the object header.

  #. ``createLinkGraph`` can be used when neither the object format nor
     the architecture are known ahead of time. In this case the object header
     will be inspected to determine both the format and architecture.

JIT Linking
===========

The JIT linker concept was introduced in LLVM's earlier generation of JIT APIs,
MCJIT, where the RuntimeDyld component enabled re-use of LLVM as an in-memory
compiler by adding an in-memory link step to the end of the usual compiler
pipeline. Rather than dumping relocatable objects to disk as a compiler usually
would, MCJIT passed them to RuntimeDyld to be linked into a target process.

This approach to linking differs from standard *static* or *dynamic* linking:

A *static linker* takes one or more relocatable object files as input and links
them into an executable or dynamic library on disk.

A *dynamic linker* applies relocations to executables and dynamic libraries that
have been loaded into memory.

A *JIT linker* takes a single relocatable object file at a time and links it
into a target process, usually using a context object to allow the linked code
to resolve symbols in the target.

RuntimeDyld
-----------

In order to keep RuntimeDyld's implementation simple MCJIT imposed some
restrictions on compiled code:

#. It had to use the Large code model, and often restricted available relocation
   models in order to limit the kinds of relocations that had to be supported.

#. It required strong linkage and default visibility on all symbols -- behavior
   for other linkages/visibilities was not well defined.

#. It constrained and/or prohibited the use of features requiring runtime
   support, e.g. static initializers or thread local storage.

As a result of these restrictions not all language features supported by LLVM
worked under MCJIT, and objects to be loaded under the JIT had to be compiled to
target it (precluding the use of precompiled code from other sources under the
JIT).

RuntimeDyld also provided very limited visibility into the linking process
itself: Clients could access conservative estimates of section size
(RuntimeDyld bundled stub size and padding estimates into the section size
value) and the final relocated bytes, but could not access RuntimeDyld's
internal object representations.

Eliminating these restrictions and limitations was one of the primary motivations
for the development of JITLink.

The llvm-jitlink tool
=====================

The ``llvm-jitlink`` tool is a command line wrapper for the JITLink library.
It loads some set of relocatable object files and then links them using
JITLink. Depending on the options used it will then execute them, or validate
the linked memory.

The ``llvm-jitlink`` tool was originally designed to aid JITLink development by
providing a simple environment for testing.

Basic usage
-----------

By default, ``llvm-jitlink`` will link the set of objects passed on the command
line, then search for a "main" function and execute it:

.. code-block:: sh

  % cat hello-world.c
  #include <stdio.h>

  int main(int argc, char *argv[]) {
    printf("hello, world!\n");
    return 0;
  }

  % clang -c -o hello-world.o hello-world.c
  % llvm-jitlink hello-world.o
  Hello, World!

Multiple objects may be specified, and arguments may be provided to the JIT'd
main function using the -args option:

.. code-block:: sh

  % cat print-args.c
  #include <stdio.h>

  void print_args(int argc, char *argv[]) {
    for (int i = 0; i != argc; ++i)
      printf("arg %i is \"%s\"\n", i, argv[i]);
  }

  % cat pring-args-main.c
  void print_args(int argc, char *argv[]);

  int main(int argc, char *argv[]) {
    print_args(argc, argv);
    return 0;
  }

  % clang -c -o print-args.o print-args.c
  % clang -c -o print-args-main.o print-args-main.c
  % llvm-jitlink print-args.o print-args-main.o -args a b c
  arg 0 is "a"
  arg 1 is "b"
  arg 2 is "c"

Alternative entry points may be specified using the ``-entry <entry point
name>`` option.

Other options can be found by calling ``llvm-jitlink -help``.

llvm-jitlink as a regression testing utility
--------------------------------------------

One of the primary aims of ``llvm-jitlink`` was to enable readable regression
tests for JITLink. To do this it supports two options:

The ``-noexec`` option tells llvm-jitlink to stop after looking up the entry
point, and before attempting to execute it. Since the linked code is not
executed, this can be used to link for other targets even if you do not have
access to the target being linked (the ``-define-abs`` or ``-phone-externals``
options can be used to supply any missing definitions in this case).

The ``-check <check-file>`` option can be used to run a set of ``jitlink-check``
expressions against working memory. It is typically used in conjunction with
``-noexec``, since the aim is to validate JIT'd memory rather than to run the
code and ``-noexec`` allows us to link for any supported target architecture
from the current process. In ``-check`` mode, ``llvm-jitlink`` will scan the
given check-file for lines of the form ``# jitlink-check: <expr>``. See
examples of this usage in ``llvm/test/ExecutionEngine/JITLink``.

Remote execution via llvm-jitlink-executor
------------------------------------------

By default ``llvm-jitlink`` will link the given objects into its own process,
but this can be overridden by two options:

The ``-oop-executor[=/path/to/executor]`` option tells ``llvm-jitlink`` to
execute the given executor (which defaults to ``llvm-jitlink-executor``) and
communicate with it via file descriptors which it passes to the executor
as the first argument with the format ``filedescs=<in-fd>,<out-fd>``.

The ``-oop-executor-connect=<host>:<port>`` option tells ``llvm-jitlink`` to
connect to an already running executor via TCP on the given host and port. To
use this option you will need to start ``llvm-jitlink-executor`` manually with
``listen=<host>:<port>`` as the first argument.

Harness mode
------------

The ``-harness`` option allows a set of input objects to be designated as a test
harness, with the regular object files implicitly treated as objects to be
tested. Definitions of symbols in the harness set override definitions in the
test set, and external references from the harness set cause automatic scope
promotion of local symbols in the test set (these modifications to the usual
linker rules are accomplished via an ``ObjectLinkingLayer::Plugin`` installed by
``llvm-jitlink`` when it sees the ``-harness`` option).

With these modifications in place we can selectively test functions in an object
file by mocking those function's callees. For example, suppose we have an object
file, ``test_code.o``, compiled from the following C source (which we need not
have access to):

.. code-block:: c

  void irrelevant_function() { irrelevant_external(); }

  int function_to_mock(int X) {
    return /* some function of X */;
  }

  static void function_to_test() {
    ...
    int Y = function_to_mock();
    printf("Y is %i\n", Y);
  }

If we want to know how ``function_to_test`` behaves when we change the behavior
of ``function_to_mock`` we can test it by writing a test harness:

.. code-block:: c

  void function_to_test();

  int function_to_mock(int X) {
    printf("used mock utility function\n");
    return 42;
  }

  int main(int argc, char *argv[]) {
    function_to_test():
    return 0;
  }

Under normal circumstances these objects could not be linked together:
``function_to_test`` is static and could not be resolved outside
``test_code.o``, the two ``function_to_mock`` functions would result in a
duplicate definition error, and ``irrelevant_external`` is undefined.
However, using ``-harness`` and ``-phony-externals`` we can run this code
with:

.. code-block:: sh

  % clang -c -o test_code_harness.o test_code_harness.c
  % llvm-jitlink -phony-externals test_code.o -harness test_code_harness.o
  used mock utility function
  Y is 42

The ``-harness`` option may be of interest to people who want to perform some
very late testing on build products to verify that compiled code behaves as
expected. On basic C test cases this is relatively straightforward. Mocks for
more complicated languages (e.g. C++) are much tricker: Any code involving
classes tends to have a lot of non-trivial surface area (e.g. vtables) that
would require great care to mock.

Tips for JITLink backend developers
-----------------------------------

#. Make liberal use of assert and ``llvm::Error``. Do *not* assume that the input
   object is well formed: Return any errors produced by libObject (or your own
   object parsing code) and validate as you construct. Think carefully about the
   distinction between contract (which should be validated with asserts and
   llvm_unreachable) and environmental errors (which should generate
   ``llvm::Error``s).

#. Don't assume you're linking in-process. Use libSupport's sized,
   endian-specific types when reading/writing content in the ``LinkGraph``.

As a "minimum viable" JITLink wrapper, the ``llvm-jitlink`` tool is an
invaluable resource for developres bringing a new JITLink backend. A standard
workflow is to start by throwing an unsupported object at the tool and seeing
what error is returned, then fixing that (you can often make a reasonable guess
at what should be done based on existing code for other formats or
architectures).

In debug builds of LLVM, the ``-debug-only=jitlink`` option dumps logs from the
JITLink library during the link process. These can be useful for spotting some bugs at
a glance. The ``-debug-only=llvm_jitlink`` option dumps logs from the ``llvm-jitlink``
tool, which can be useful for debugging both testcases (it is often less verbose than
``-debug-only=jitlink``) and the tool itself.

The ``-oop-executor`` and ``-oop-executor-connect`` options are helpful for testing
handling of cross-process and cross-architecture use cases.

Roadmap
=======

JITLink is under active development. Work so far has focused on the MachO
implementation. In LLVM 12 there is limited support for ELF on x86-64.

Major outstanding projects include:

* Refactor architecture support to maximize sharing across formats.

  All formats should be able to share the bulk of the architecture specific
  code (especially relocations) for each supported architecture.

* Refactor ELF link graph construction.

  ELF's link graph construction is currently implemented in the `ELF_x86_64.cpp`
  file, and tied to the x86-64 relocation parsing code. The bulk of the code is
  generic and should be split into an ELFLinkGraphBuilder base class along the
  same lines as the existing generic MachOLinkGraphBuilder.

* Implement ELF suport for arm64.

  Once the architecture support code has been refactored to enable sharing and
  ELF link graph construction has been refactored to allow re-use we should be
  able to construct an ELF / arm64 JITLink implementation by combining
  these existing pieces.

* Implement support for new architectures.

* Implement support for COFF.

  There is no COFF implementation of JITLink yet. Such an implementation should
  follow the MachO and ELF paths: a generic COFFLinkGraphBuilder base class that
  can be specialized for each architecture.

* Design and implement a shared-memory based JITLinkMemoryManager.

  One use-case that is expected to be common is out-of-process linking targeting
  another process on the same machine. This allows JITs to sandbox JIT'd code.
  For this use case a shared-memory based JITLinkMemoryManager would provide the
  most efficient form of allocation. Creating one will require designing a
  generic API for shared memory though, as LLVM does not currently have one.

JITLink Availability and Feature Status
---------------------------------------

.. list-table:: Avalability and Status
   :widths: 10 30 30 30
   :header-rows: 1

   * - Architecture
     - ELF
     - COFF
     - MachO
   * - arm64
     -
     -
     - Partial (small code model, PIC relocation model only)
   * - x86-64
     - Partial
     -
     - Full (except TLV and debugging)

.. [1] See ``llvm/examples/OrcV2Examples/LLJITWithObjectLinkingLayerPlugin`` for
       a full worked example.

.. [2] If not for *hidden* scoped symbols we could eliminate the
       ``JITLinkDylib*`` argument to ``JITLinkMemoryManager::allocate`` and
       treat every object as a separate simulated dylib for the purposes of
       memory layout. Hidden symbols break this by generating in-range accesses
       to external symbols, requiring the access and symbol to be allocated
       within range of one another. That said, providing a pre-reserved address
       range pool for each simulated dylib guarantees that the relaxation
       optimizations will kick in for all intra-dylib references, which is good
       for performance (at the cost of whatever overhead is introduced by
       reserving the address-range up-front).
