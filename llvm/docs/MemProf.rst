====================================
MemProf: Memory Profiling for LLVM
====================================

.. contents::
   :local:
   :depth: 2

Introduction
============

MemProf is a profile-guided optimization (PGO) feature for memory. It enables the compiler to optimize memory allocations and static data layout based on runtime profiling information. By understanding the "hotness", lifetime, and access frequency of memory, the compiler can make better decisions about where to place data (both heap and static), reducing fragmentation and improving cache locality.

Motivation
----------

Traditional PGO focuses on control flow (hot vs. cold code). MemProf extends this concept to data. It answers questions like:

*   Which allocation sites are "hot" (frequently accessed)?
*   Which allocation sites are "cold" (rarely accessed)?
*   What is the lifetime of an allocation?

This information enables optimizations such as:

*   **Heap Layout Optimization:** Grouping objects with similar lifetimes or access density. This currently requires an allocator that supports the necessary interfaces (e.g., tcmalloc).
*   **Static Data Partitioning:** Segregating frequently accessed (hot) global variables and constants from rarely accessed (cold) ones to improve data locality and TLB utilization.

User Manual
===========

This section describes how to use MemProf to profile and optimize your application.

Building with MemProf Instrumentation
-------------------------------------

To enable MemProf instrumentation, compile your application with the ``-fmemory-profile`` flag. Make sure to include debug information (``-gmlt`` and ``-fdebug-info-for-profiling``) and frame pointers to ensure accurate stack traces and line number reporting.

.. code-block:: bash

    clang++ -fmemory-profile -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -fno-optimize-sibling-calls -fdebug-info-for-profiling -gmlt -O2 -fno-pie -no-pie -Wl,-z,noseparate-code -Wl,--build-id source.cpp -o app

.. note::
    Link with ``-fmemory-profile`` as well to link the necessary runtime libraries. If you use a separate link step, ensure the flag is passed to the linker.
    On Linux, the flags ``-fno-pie -no-pie -Wl,-z,noseparate-code -Wl,--build-id`` are currently required to ensure the binary layout (executable segment at offset 0) and Build ID presence are compatible with the ``llvm-profdata`` profile reader.

Running and Generating Profiles
-------------------------------

Run the instrumented application. By default, MemProf writes a raw profile file named ``memprof.profraw.<pid>`` to the current directory upon exit. Control the runtime behavior using the ``MEMPROF_OPTIONS`` environment variable. Common options include:

*   ``log_path``: Redirects runtime logs (e.g., ``stdout``, ``stderr``, or a file path).
*   ``print_text``: If set to ``true``, prints a text-based summary of the profile to the log path.
*   ``verbosity``: Controls the level of debug output.

**Example:**

.. code-block:: bash

    MEMPROF_OPTIONS=log_path=stdout:print_text=true ./app

.. _Processing Profiles:

Processing Profiles
-------------------

Raw profiles must be indexed before the compiler can use them. Use ``llvm-profdata`` to merge and index the raw profiles.

.. code-block:: bash

    llvm-profdata merge memprof.profraw.* --profiled-binary ./app -o memprof.memprofdata

To dump the profile in YAML format (useful for debugging or creating test cases):

.. code-block:: bash

    llvm-profdata show --memory memprof.memprofdata > memprof.yaml

Merge MemProf profiles with standard PGO instrumentation profiles if you have both (optional).

Using Profiles for Optimization
-------------------------------

Feed the indexed profile back into the compiler using the ``-fmemory-profile-use=`` option (or low-level passes options).

.. code-block:: bash

    clang++ -fmemory-profile-use=memprof.memprofdata -O2 -Wl,-mllvm,-enable-memprof-context-disambiguation -Wl,-mllvm,-optimize-hot-cold-new -Wl,-mllvm,-supports-hot-cold-new source.cpp -o optimized_app -ltcmalloc

If invoking the optimizer directly via ``opt``:

.. code-block:: bash

    opt -passes='memprof-use<profile-filename=memprof.memprofdata>' ...

The compiler uses the profile data to annotate allocation instructions with ``!memprof`` metadata (`MemProf Metadata Documentation <https://llvm.org/docs/LangRef.html#memprof-metadata>`_), distinguishing between "hot", "cold", and "notcold" allocations. This metadata guides downstream optimizations. Additionally, callsites which are part of allocation contexts are also annotated with ``!callsite`` metadata (`Callsite Metadata Documentation <https://llvm.org/docs/LangRef.html#callsite-metadata>`_).

.. note::
    Ensure that the same debug info flags (e.g. ``-gmlt`` and ``-fdebug-info-for-profiling``) used during instrumentation are also passed during this compilation step to enable correct matching of the profile data.
    For the optimized binary to fully utilize the hot/cold hinting, it must be linked with an allocator that supports this mechanism, such as `tcmalloc <https://github.com/google/tcmalloc>`_. TCMalloc provides an API (``tcmalloc::hot_cold_t``) that accepts a hint (0 for cold, 255 for hot) to guide data placement and improve locality. To indicate that the library supports these interfaces, the ``-mllvm -supports-hot-cold-new`` flag is used during the LTO link.

Context Disambiguation (LTO)
----------------------------

To fully benefit from MemProf, especially for common allocation wrappers, enabling **ThinLTO** (preferred) or **Full LTO** is required. This allows the compiler to perform **context disambiguation**.

Consider the following example:

.. code-block:: cpp

    void *allocate() { return new char[10]; }

    void hot_path() {
      // This path is executed frequently.
      allocate();
    }

    void cold_path() {
       // This path is executed rarely.
       allocate();
    }

Without context disambiguation, the compiler sees a single ``allocate`` function called from both hot and cold contexts. It must conservatively assume the allocation is "not cold" or "ambiguous".

With LTO and MemProf:

1.  The compiler constructs a whole-program call graph.
2.  It identifies that ``allocate`` has distinct calling contexts with different behaviors.
3.  It **clones** ``allocate`` into two versions: one for the hot path and one for the cold path.
4.  The call in ``cold_path`` is updated to call the cloned "cold" version of ``allocate``, which can then be optimized (e.g., by passing a cold hint to the allocator).

Static Data Partitioning
------------------------

MemProf profiles guide the layout of static data (e.g., global variables, constants). The goal is to separate "hot" data from "cold" data in the binary, placing hot data into specific sections (e.g., ``.rodata.hot``) to minimize the number of pages required for the working set.

This feature uses a hybrid approach:

1.  **Symbolizable Data:** Data with external or local linkage (tracked by the symbol table) is partitioned based on data access profiles collected via instrumentation (`PR <https://github.com/llvm/llvm-project/pull/142884>`_) or hardware performance counters (e.g., Intel PEBS events such as ``MEM_INST_RETIRED.ALL_LOADS``).
2.  **Module-Internal Data:** Data not tracked by the symbol table (e.g., jump tables, constant pools, internal globals) has its hotness inferred from standard PGO code execution profiles.

To enable this feature, pass the following flags to the compiler:

*   ``-fpartition-static-data-sections``: Instructs the compiler to generate `.hot` and `.unlikely` section prefixes for hot and cold static data respectively in the relocatable object files.
*   ``-Wl,-z,keep-data-section-prefix``: Informs the LLD linker that `.data.rel.ro.hot` and `.data.rel.ro.unlikely` as relro sections. LLD requires all relro sections to be contiguous and this flag allows us to interleave the hotness-suffixed `.data.rel.ro` sections with other relro sections.
*   ``-Wl,-script=<linker_script>``: Group hot and/or cold data sections, and order the data sections.

.. code-block:: bash

    clang++ -fmemory-profile-use=memprof.memprofdata -fpartition-static-data-sections -fuse-ld=lld -Wl,-z,keep-data-section-prefix -O2 source.cpp -o optimized_app

The optimized layout clusters hot static data, improving dTLB and cache efficiency.

.. note::
   When both PGO profiles and memory profiles are provided (using
   ``-fprofile-use`` and ``-fmemory-profile-use``), global variable hotness are
   inferred from a combination of PGO profile and data access profile:

   * For data covered by both profiles (e.g., module-internal data with symbols
     in the executable), the hotness is the max of PGO profile hotness and data
     access profile hotness.

   * For data covered by only one profile, the hotness is inferred from that
     profile. Most notably, symbolizable data with external linkage is only
     covered by data access profile, and module-internal unsymbolizable data is
     only covered by PGO profile.

Developer Manual
================

This section provides an overview of the MemProf architecture and implementation for contributors.

Architecture Overview
---------------------

MemProf consists of three main components:

1.  **Instrumentation Pass (Compile-time):** Memory accesses are instrumented to increment the access count held in a shadow memory location, or alternatively to call into the runtime. Memory allocations are intercepted by the runtime library.
2.  **Runtime Library (Link-time/Run-time):** Manages shadow memory and tracks allocation contexts and access statistics.
3.  **Profile Analysis (Post-processing/Compile-time):** Tools and passes that read the profile, annotate the IR using metadata, and perform context disambiguation if necessary when LTO is enabled.

Detailed Workflow (LTO)
-----------------------

The optimization process, using LTO, involves several steps:

1. **Matching (MemProfUse Pass):** The memprof profile is mapped onto allocation calls and callsites which are part of the allocation context using debug information. MemProf metadata is attached to the call instructions in the IR. If the allocation call site is unambiguously cold (or hot) an attribute is added directly which guides the transformation.
2.  **Metadata Serialization:** For ThinLTO, during the summary analysis step, MemProf metadata (``!memprof`` and ``!callsite``) is serialized into the module summary. This is implemented in ``llvm/lib/Analysis/ModuleSummaryAnalysis.cpp``.
3.  **Whole Program Graph Construction:** During the LTO step, the compiler constructs a whole-program ``CallsiteContextGraph`` to analyze and disambiguate contexts. This graph identifies where allocation contexts diverge (e.g., same function called from hot vs. cold paths). This logic resides in ``llvm/lib/Transforms/IPO/MemProfContextDisambiguation.cpp``.
4.  **Cloning Decisions:** The analysis identifies which functions and callsites need to be cloned to isolate cold allocation paths from hot ones using the ``CallsiteContextGraph``.
5.  **LTO Backend:** The actual cloning of functions happens in the ``MemProfContextDisambiguation`` pass. The replacement of allocation calls (e.g., ``operator new`` to the ``hot_cold_t`` variant) happens in ``SimplifyLibCalls`` during the ``InstCombine`` pass. These transformations are guided by the decisions made during the LTO step.

Source Structure
----------------

*   **Runtime:** ``compiler-rt/lib/memprof``

    *   Contains the runtime implementation, including shadow memory mapping, interceptors (malloc, free, etc.), and the thread-local storage for recording stats.

*   **Instrumentation:** ``llvm/lib/Transforms/Instrumentation/MemProfInstrumentation.cpp``

    *   Implements the LLVM IR pass that adds instrumentation calls.

*   **Profile Data:** ``llvm/include/llvm/ProfileData/MemProf.h`` and ``MemProfData.inc``

    *   Defines the profile format, data structures (like ``MemInfoBlock``), and serialization logic.

*   **Use Pass:** ``llvm/lib/Transforms/Instrumentation/MemProfUse.cpp``

    *   Reads the profile and annotates the IR with metadata.

*   **Context Disambiguation:** ``llvm/lib/Transforms/IPO/MemProfContextDisambiguation.cpp``

    *   Implements the analysis and transformations (e.g., cloning) for resolving ambiguous allocation contexts using LTO.

*   **Transformation:** ``llvm/lib/Transforms/Utils/SimplifyLibCalls.cpp``

    *   Implements the rewriting of allocation calls based on the hot/cold hints.

*   **Static Data Partitioning:** ``llvm/lib/CodeGen/AsmPrinter/AsmPrinter.cpp``, ``llvm/lib/CodeGen/StaticDataSplitter.cpp``, and ``llvm/lib/CodeGen/StaticDataAnnotator.cpp``

    *   Implements the splitting of static data (Jump tables, Module-internal global variables, and Constant pools) into hot and cold sections using branch profile data. ``StaticDataAnnotator`` iterates over global variables and sets their section prefixes based on the profile analysis from ``StaticDataSplitter``.

Runtime Implementation
----------------------

The runtime uses a **shadow memory** scheme similar to AddressSanitizer (ASan) but optimized for profiling.

*   **Shadow Mapping:** Application memory is mapped to shadow memory.

*   **Granularity:** The default granularity is 64 bytes. One byte of shadow memory tracks the access state of 64 bytes of application memory.

*   **MemInfoBlock (MIB):** A key data structure that stores statistics for an allocation context, including: ``AllocCount``, ``TotalAccessCount``, ``TotalLifetime``, and ``Min/MaxAccessDensity``.

Profile Format
--------------

The MemProf profile is a schema-based binary format designed for extensibility. Key structures include:

*   **Frame:** Represents a function in the call stack (Function GUID, Line, Column).
*   **CallStack:** A sequence of Frames identifying the context of an allocation.
*   **MemInfoBlock:** The statistics gathered for a specific CallStack.

The format supports versioning to allow adding new fields to the MIB without breaking backward compatibility.

Static Data Profile
~~~~~~~~~~~~~~~~~~~

To support static data partitioning, the profile format includes a payload for symbolized data access profiles. This maps data addresses to canonical symbol names (or module source location for internal data) and access counts. This enables the compiler to identify which global variables are hot.

Testing

-------



When making changes to MemProf, verify your changes using the following test suites:



1.  **Runtime Tests:**



    *   Location: ``compiler-rt/test/memprof``

    *   Purpose: Verify the runtime instrumentation, shadow memory behavior, and profile generation.



2.  **Profile Manipulation Tests:**



    *   Location: ``llvm/test/tools/llvm-profdata``

    *   Purpose: Verify that ``llvm-profdata`` can correctly merge, show, and handle MemProf profiles.



3.  **Instrumentation & Optimization Tests:**



    *   Location: ``llvm/test/Transforms/PGOProfile``

    *   Purpose: Verify the correctness of the ``MemProfUse`` pass, metadata annotation, and IR transformations.



4.  **ThinLTO & Context Disambiguation Tests:**



    *   Location: ``llvm/test/ThinLTO/X86/memprof*`` and ``llvm/test/Transforms/MemProfContextDisambiguation``

    *   Purpose: Verify context disambiguation, cloning, and summary analysis during ThinLTO and LTO.

Testing with YAML Profiles
--------------------------

You can create MemProf profiles in YAML format for testing purposes. This is useful for creating small, self-contained test cases without needing to run a binary.

1.  **Create a YAML Profile:** You can start by dumping a real profile to YAML (see :ref:`Processing Profiles` above) or writing one from scratch.

2.  **Convert to Indexed Format:** Use ``llvm-profdata`` to convert the YAML to the indexed MemProf format.

    .. code-block:: bash

        llvm-profdata merge --memprof-version=4 profile.yaml -o profile.memprofdata

3.  **Run the Compiler:** Use the indexed profile with ``opt`` or ``clang``.

    .. code-block:: bash

        opt -passes='memprof-use<profile-filename=profile.memprofdata>' test.ll -S

**Example YAML Profile:** ::

   ---
   HeapProfileRecords:
     - GUID:            _Z3foov
       AllocSites:
         - Callstack:
             - { Function: _Z3foov, LineOffset: 0, Column: 22, IsInlineFrame: false }
             - { Function: main, LineOffset: 2, Column: 5, IsInlineFrame: false }
           MemInfoBlock:
             TotalSize:                  400
             AllocCount:                 1
             TotalLifetimeAccessDensity: 1
             TotalLifetime:              1000000
       CallSites:       []
   ...
