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

*   **Heap Layout Optimization:** Grouping objects with similar lifetimes or access density.
*   **Static Data Partitioning:** Segregating frequently accessed (hot) global variables and constants from rarely accessed (cold) ones to improve data locality and TLB utilization.

User Manual
===========

This section describes how to use MemProf to profile and optimize your application.

Building with MemProf
---------------------

To enable MemProf instrumentation, compile your application with the ``-fmemory-profile`` flag. Make sure to include debug information (``-gmlt`` and ``-fdebug-info-for-profiling``) and frame pointers to ensure accurate stack traces and line number reporting.

.. code-block:: bash

    clang++ -fmemory-profile -fdebug-info-for-profiling -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -gmlt -O2 source.cpp -o app

.. note::
    Link with ``-fmemory-profile`` as well to link the necessary runtime libraries. If you use a separate link step, ensure the flag is passed to the linker.

Running and Generating Profiles
-------------------------------

Run the instrumented application. By default, MemProf writes a raw profile file named ``memprof.profraw.<pid>`` to the current directory upon exit.

.. code-block:: bash

    ./app

Control the runtime behavior using the ``MEMPROF_OPTIONS`` environment variable. Common options include:

*   ``log_path``: Redirects runtime logs (e.g., ``stdout``, ``stderr``, or a file path).
*   ``print_text``: If set to ``true``, prints a text-based summary of the profile to the log path.
*   ``verbosity``: Controls the level of debug output.

**Example:**

.. code-block:: bash

    MEMPROF_OPTIONS=log_path=stdout:print_text=true ./app

Processing Profiles
-------------------

Raw profiles must be indexed before the compiler can use them. Use ``llvm-profdata`` to merge and index the raw profiles.

.. code-block:: bash

    llvm-profdata merge memprof.profraw.* --profiled-binary ./app -o memprof.memprofdata

To dump the profile in YAML format (useful for debugging or creating test cases):

.. code-block:: bash

    llvm-profdata show --memory memprof.memprofdata > memprof.yaml

Merge MemProf profiles with standard PGO instrumentation profiles if you have both.

Using Profiles for Optimization
-------------------------------

Feed the indexed profile back into the compiler using the ``-fmemory-profile-use=`` option (or low-level passes options).

.. code-block:: bash

    clang++ -fmemory-profile-use=memprof.memprofdata -O2 source.cpp -o optimized_app -ltcmalloc

If invoking the optimizer directly via ``opt``:

.. code-block:: bash

    opt -passes='memprof-use<profile-filename=memprof.memprofdata>' ...

The compiler uses the profile data to annotate allocation instructions with metadata (e.g., ``!memprof``), distinguishing between "hot", "cold", and "notcold" allocations. This metadata guides downstream optimizations.

.. note::
    For the optimized binary to utilize the hot/cold hinting, it must be linked with an allocator that supports this mechanism, such as `tcmalloc <https://github.com/google/tcmalloc>`_. TCMalloc extends operator new that accepts a hint (0 for cold, 255 for hot) to guide data placement and improve locality.

Context Disambiguation (LTO/ThinLTO)
------------------------------------

To fully benefit from MemProf, especially for common allocation wrappers, enabling **ThinLTO** (preferred) or LTO is required. This allows the compiler to perform **context disambiguation**.

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

With ThinLTO and MemProf:
1.  The compiler constructs a whole-program call graph.
2.  It identifies that ``allocate`` has distinct calling contexts with different behaviors.
3.  It **clones** ``allocate`` into two versions: one for the hot path and one for the cold path.
4.  The call in ``cold_path`` is updated to call the cloned "cold" version of ``allocate``, which can then be optimized (e.g., by passing a cold hint to the allocator).

Static Data Partitioning
------------------------

MemProf profiles guide the layout of static data (e.g., global variables, constants). The goal is to separate "hot" data from "cold" data in the binary, placing hot data into specific sections (e.g., ``.rodata.hot``) to minimize the number of pages required for the working set.

This feature uses a hybrid approach:

1.  **Symbolizable Data:** Data with external or local linkage (tracked by the symbol table) is partitioned based on data access profiles collected via instrumentation (`draft <https://github.com/llvm/llvm-project/pull/142884>`_) or hardware performance counters (e.g., Intel PEBS events such as ``MEM_INST_RETIRED.ALL_LOADS``).
2.  **Module-Internal Data:** Data not tracked by the symbol table (e.g., jump tables, constant pools, internal globals) has its hotness inferred from standard PGO code execution profiles.

To enable this feature, pass the following flags to the compiler:

*   ``-memprof-annotate-static-data-prefix``: Enables annotation of global variables in IR.
*   ``-split-static-data``: Enables partitioning of other data (like jump tables) in the backend.
*   ``-Wl,-z,keep-data-section-prefix``: Instructs the linker (LLD) to group hot and cold data sections together.

.. code-block:: bash

    clang++ -fmemory-profile-use=memprof.memprofdata -mllvm -memprof-annotate-static-data-prefix -mllvm -split-static-data -fuse-ld=lld -Wl,-z,keep-data-section-prefix -O2 source.cpp -o optimized_app

The optimized layout clusters hot static data, improving dTLB and cache efficiency.

Developer Manual
================

This section provides an overview of the MemProf architecture and implementation for contributors.

Architecture Overview
---------------------

MemProf consists of three main components:

1.  **Instrumentation Pass (Compile-time):** Injects code to record memory allocations and accesses.
2.  **Runtime Library (Link-time/Run-time):** Manages shadow memory and tracks allocation contexts and access statistics.
3.  **Profile Analysis (Post-processing/Compile-time):** Tools and passes that read the profile, annotate the IR, and perform advanced optimizations like context disambiguation for ThinLTO.

Detailed Workflow (ThinLTO)
---------------------------

The optimization process, particularly context disambiguation, involves several steps during the ThinLTO pipeline:

1.  **Metadata Serialization:** During the ThinLTO summary analysis step, MemProf metadata (including MIBs and CallStacks) is serialized into the module summary. This is implemented in ``llvm/lib/Analysis/ModuleSummaryAnalysis.cpp``.
2.  **Whole Program Graph Construction:** During the ThinLTO indexing step, the compiler constructs a whole-program ``CallingContextGraph`` to analyze and disambiguate contexts. This graph identifies where allocation contexts diverge (e.g., same function called from hot vs. cold paths). This logic resides in ``llvm/lib/Transforms/IPO/MemProfContextDisambiguation.cpp``.
3.  **Auxiliary Graph & Cloning Decisions:** An auxiliary graph is constructed to guide the cloning process. The analysis identifies which functions and callsites need to be cloned to isolate cold allocation paths from hot ones.
4.  **ThinLTO Backend:** The actual cloning of functions and replacement of allocation calls (e.g., ``operator new``) happens in the ThinLTO backend passes. These transformations are guided by the decisions made during the indexing step.

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
*   **Context Disambiguation:** ``llvm/lib/transforms/ipo/MemProfContextDisambiguation.cpp``
    *   Implements the analysis and transformations (e.g., cloning) for resolving ambiguous allocation contexts, particularly during ThinLTO.

Runtime Implementation
----------------------

The runtime uses a **shadow memory** scheme similar to AddressSanitizer (ASan) but optimized for profiling.
*   **Shadow Mapping:** Application memory is mapped to shadow memory.
*   **Granularity:** The default granularity is 64 bytes. One byte of shadow memory tracks the access state of 64 bytes of application memory.
*   **MemInfoBlock (MIB):** A key data structure that stores statistics for an allocation context, including:
    *   ``AllocCount``
    *   ``TotalAccessCount``
    *   ``TotalLifetime``
    *   ``Min/MaxAccessDensity``

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
    *   Location: ``llvm/test/ThinLTO/X86``
    *   Purpose: Verify context disambiguation, cloning, and summary analysis during ThinLTO.

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

**Example YAML Profile:**

.. code-block:: yaml

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
