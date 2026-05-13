==================================
The LLVM Instrumentor Pass
==================================

.. contents::
   :local:

Introduction
============

The **Instrumentor** is a highly configurable instrumentation pass for LLVM-IR
that allows users to insert custom runtime function calls at various program
points. Unlike traditional instrumentation tools that are hardcoded for
specific purposes (like sanitizers or profilers), the Instrumentor provides a
flexible, configuration-driven approach where users can specify:

- **What** to instrument (loads, stores, allocations, function calls, etc.)
- **Where** to instrument (before or after operations)
- **What information** to pass to the runtime (pointers, values, sizes, types, etc.)
- **Whether** to modify program behavior (replace values, redirect pointers, etc.)

The Instrumentor is designed to support a wide variety of use cases including:

- Custom memory profilers and trackers
- Performance analysis tools
- Dynamic program analysis
- Debugging and tracing utilities
- Stack usage monitoring
- Custom sanitizers and checkers

To use the Instrumentor it is recommended to run the wizard script located at
`./llvm/utils/instrumentor-config-wizard.py`. The script will interactively
create a configuration file and a stub runtime which is required to be linked
into the instrumented program.

Key Features
============

Configurable Instrumentation Opportunities
-------------------------------------------

The Instrumentor supports instrumentation at multiple levels:

**Instruction-level:**
  - **Load instructions**: Instrument memory reads with access to pointer, loaded value, alignment, size, atomicity, etc.
  - **Store instructions**: Instrument memory writes with access to pointer, stored value, alignment, size, atomicity, etc.
  - **Alloca instructions**: Instrument stack allocations with access to size, alignment, and allocated address

**Function-level:**
  - **Function entry**: Instrument at function start with access to function name, address, arguments, etc.
  - **Function exit**: Instrument at function return

**Future extensions:**
  - Basic block entry/exit
  - Module-level initialization
  - Global variable access

PRE and POST Instrumentation
-----------------------------

Each instrumentation opportunity supports two positions:

- **PRE**: Insert instrumentation **before** the operation occurs

  - For loads: can inspect/modify the pointer before reading
  - For stores: can inspect/modify the pointer and value before writing
  - For allocas: can modify the allocation size
  - For functions: instrument at function entry, inspect/replace arguments

- **POST**: Insert instrumentation **after** the operation occurs

  - For loads: can inspect/modify the loaded value
  - For stores: instrument after the write completes
  - For allocas: can inspect/modify the allocated address
  - For functions: instrument at function exit

Selective Argument Passing
---------------------------

For each instrumentation opportunity, users can individually enable/disable specific arguments to control:

- What information is passed to the runtime function
- The signature of the generated runtime function
- Performance overhead (fewer arguments = faster calls)

For example, for load instrumentation, you can choose to pass:

- Pointer address
- Pointer address space
- Loaded value
- Value size
- Alignment
- Value type ID
- Atomicity ordering
- Synchronization scope
- Volatility flag
- Unique instrumentation ID

Value Replacement
-----------------

The Instrumentor supports **replacing** values returned from the runtime:

- **Load replacement**: The runtime can provide a different value than what was loaded from memory
- **Store replacement**: The runtime can modify the pointer or value being stored
- **Alloca replacement**: The runtime can provide a different allocation size or replace the allocated address
- **Argument replacement**: The runtime can modify the arguments passed to a function

This enables use cases like:

- Value redirection for debugging
- Custom memory allocators
- Fault injection
- Taint tracking

Instrumentation Filtering
-------------------------

The Instrumentor provides fine-grained control over what gets instrumented:

- **Target regex**: Match against the target triple (e.g., ``x86_64-.*-linux``)
- **Host/GPU toggle**: Separately enable/disable CPU and GPU instrumentation
- **Function filtering**: Exclude runtime functions from instrumentation via a regular expression

Configuration System
====================

The Instrumentor uses a JSON-based configuration system that allows users to:

1. Generate a default configuration showing all available options
2. Interactively customize the configuration using the wizard
3. Load and modify existing configurations
4. Generate runtime stub implementations

Configuration File Format
-------------------------

The configuration file is a JSON document with the following structure:

.. code-block:: json

   {
     "configuration": {
       "runtime_prefix": "__instrumentor_",
       "target_regex": "",
       "host_enabled": true,
       "gpu_enabled": true
     },
     "function_pre": {
       "function": {
         "enabled": true,
         "address": true,
         "name": true,
         "id": true
       }
     },
     "instruction_pre": {
       "load": {
         "enabled": true,
         "pointer": true,
         "pointer.replace": false,
         "value_size": true,
         "id": true
       },
       "store": {
         "enabled": true,
         "pointer": true,
         "value": true,
         "value_size": true
       }
     },
     "instruction_post": {
       "load": {
         "enabled": true,
         "value": true,
         "value.replace": false
       }
     }
   }

Configuration Sections
----------------------

**configuration**
  Global settings that apply to all instrumentation:

  - ``runtime_prefix``: Prefix for all runtime function names (default: ``__instrumentor_``)
  - ``target_regex``: Regular expression to filter targets (empty = all targets)
  - ``host_enabled``: Enable instrumentation for CPU targets (default: true)
  - ``gpu_enabled``: Enable instrumentation for GPU targets (default: true)

**function_pre / function_post**
  Function-level instrumentation configuration.

**instruction_pre / instruction_post**
  Instruction-level instrumentation configuration, with subsections for each instruction type (``load``, ``store``, ``alloca``, etc.).

Argument Configuration
----------------------

For each instrumentation opportunity, arguments are configured with:

- **enabled**: Boolean to enable/disable the entire opportunity
- **<argument_name>**: Boolean to enable/disable passing this argument
- **<argument_name>.replace**: Boolean to enable value replacement (only for replaceable arguments)
- **<argument_name>.description**: Human-readable description of the argument

The Configuration Wizard
=========================

The Instrumentor includes an interactive configuration wizard that simplifies the process of creating and modifying configurations.

Running the Wizard
------------------

.. code-block:: bash

   # Run the wizard interactively
   ./llvm/utils/instrumentor-config-wizard.py

   # Specify output location
   ./llvm/utils/instrumentor-config-wizard.py -o my_config.json

   # Use specific opt binary
   ./llvm/utils/instrumentor-config-wizard.py --opt-path /path/to/opt

   # Load and modify existing configuration
   ./llvm/utils/instrumentor-config-wizard.py --input existing.json -o modified.json

Wizard Workflow
---------------

The wizard guides you through five steps:

**Step 1: Select Instrumentation Types**
  Choose which types of operations to instrument (load, store, alloca, function, etc.). This is a high-level selection - you can configure individual arguments later.

**Step 2: PRE vs POST Configuration**
  Decide whether PRE and POST instrumentation should use the same configuration or different configurations. This saves time when you want both positions to have identical settings.

**Step 3: Base Configuration**
  Configure global settings:

  - Runtime prefix for function names
  - Target regex for filtering
  - Enable/disable host (CPU) instrumentation
  - Enable/disable GPU instrumentation

**Step 4: Configure Arguments**
  For each enabled instrumentation type, select which arguments to pass to the runtime function. You can:

  - Toggle individual arguments on/off
  - Enable value replacement for replaceable arguments
  - Enable all or disable all arguments
  - Configure PRE and POST separately (if selected in Step 2)

**Step 5: Review and Save**
  Review your configuration and optionally generate runtime stub implementations. The wizard displays a summary and provides commands for using the configuration with ``opt`` and ``clang``.

Generating Runtime Stubs
-------------------------

The wizard can automatically generate C stub implementations of your runtime functions:

1. In Step 5, select 'g' to generate stubs
2. Specify the output file path (default: ``<config_name>_stubs.c``)
3. The wizard creates a C file with stub implementations that print their arguments

The generated stubs are useful as:

- Starting templates for implementing your runtime
- Documentation of the expected function signatures
- Quick prototypes for testing instrumentation

Example stub output:

.. code-block:: c

   void __instrumentor_pre_load(void *pointer, int32_t pointer_as,
                                 uint64_t value_size, int32_t id) {
     printf("load pre -- pointer: %p, pointer_as: %i, "
            "value_size: %lu, id: %i\n",
            pointer, pointer_as, value_size, id);
   }

Usage Examples
==============

Basic Usage with opt
--------------------

**Step 1: (Optional) Generate a default configuration**

.. code-block:: bash

   opt -passes=instrumentor \
       -instrumentor-write-config-file=config.json \
       -disable-output \
       input.ll

This creates ``config.json`` with all available instrumentation opportunities and their arguments.

**Step 2: Customize the configuration**

Edit ``config.json`` manually or use the wizard (no input needed):

.. code-block:: bash

   ./llvm/utils/instrumentor-config-wizard.py --input config.json -o custom.json

**Step 3: Apply instrumentation**

.. code-block:: bash

   opt -passes=instrumentor \
       -instrumentor-read-config-file=custom.json \
       input.ll -S -o instrumented.ll

The instrumented output contains calls to your runtime functions at the configured program points.

Using with Clang
----------------

To instrument during compilation:

.. code-block:: bash

   clang -mllvm -enable-instrumentor \
         -mllvm -instrumentor-read-config-file=config.json \
         source.c -o program

Complete Workflow Example
--------------------------

Here's a complete example for creating a simple memory access profiler:

**1. Create configuration with the wizard:**

.. code-block:: bash

   ./llvm/utils/instrumentor-config-wizard.py -o memory_profiler.json

   # In the wizard:
   # - Enable: load, store
   # - Use same config for PRE/POST: yes
   # - Base config: keep defaults
   # - For load/store: enable pointer, value_size, id
   # - Generate stubs: yes (memory_profiler_stubs.c)

**2. Implement the runtime:**

.. code-block:: c

   // memory_runtime.c
   #include <stdio.h>
   #include <stdint.h>

   static uint64_t load_count = 0;
   static uint64_t store_count = 0;

   void __instrumentor_pre_load(void *pointer, uint64_t value_size,
                                  int32_t id) {
     load_count++;
     printf("Load from %p (size: %lu, id: %d)\n",
            pointer, value_size, id);
   }

   void __instrumentor_pre_store(void *pointer, uint64_t value_size,
                                   int32_t id) {
     store_count++;
     printf("Store to %p (size: %lu, id: %d)\n",
            pointer, value_size, id);
   }

   __attribute__((destructor))
   void print_stats(void) {
     printf("Total loads: %lu\n", load_count);
     printf("Total stores: %lu\n", store_count);
   }

**3. Instrument and compile:**

.. code-block:: bash

   # Instrument the program
   clang -emit-llvm -S -o program.ll program.c
   opt -passes=instrumentor \
       -instrumentor-read-config-file=memory_profiler.json \
       program.ll -S -o program_inst.ll

   # Compile with runtime
   clang program_inst.ll memory_runtime.c -o program

**4. Run and observe:**

.. code-block:: bash

   ./program
   # Output includes:
   # Load from 0x7ffc12345678 (size: 4, id: 1)
   # Store to 0x7ffc12345680 (size: 8, id: 2)
   # ...
   # Total loads: 42
   # Total stores: 27

Advanced Use Cases
==================

Stack Usage Profiling
----------------------

Configure alloca instrumentation to track stack allocations:

.. code-block:: json

   {
     "instruction_pre": {
       "alloca": {
         "enabled": true,
         "size": true,
         "alignment": true,
         "id": true
       }
     },
     "instruction_post": {
       "alloca": {
         "enabled": true,
         "address": true,
         "size": true
       }
     }
   }

Runtime implementation:

.. code-block:: c

   static uint64_t total_stack_usage = 0;
   static uint64_t peak_stack_usage = 0;
   static uint64_t current_stack_usage = 0;

   void __instrumentor_post_alloca(void *address, uint64_t size,
                                     int32_t id) {
     current_stack_usage += size;
     total_stack_usage += size;
     if (current_stack_usage > peak_stack_usage) {
       peak_stack_usage = current_stack_usage;
     }
   }

Value Replacement for Fault Injection
--------------------------------------

Use value replacement to inject faults:

.. code-block:: json

   {
     "instruction_post": {
       "load": {
         "enabled": true,
         "value": true,
         "value.replace": true,
         "pointer": true
       }
     }
   }

Runtime implementation:

.. code-block:: c

   // Replace every 1000th loaded value with zero
   static uint64_t load_counter = 0;

   uint64_t __instrumentor_post_load(uint64_t value, void *pointer) {
     if (++load_counter % 1000 == 0) {
       printf("Injecting fault at %p\n", pointer);
       return 0;  // Return fault value
     }
     return value;  // Return original value
   }

Function-Level Tracing
----------------------

Instrument function entry and exit:

.. code-block:: json

   {
     "function_pre": {
       "function": {
         "enabled": true,
         "name": true,
         "address": true,
         "num_arguments": true
       }
     },
     "function_post": {
       "function": {
         "enabled": true,
         "name": true
       }
     }
   }

Runtime implementation:

.. code-block:: c

   static int call_depth = 0;

   void __instrumentor_pre_function(char *name, void *address,
                                      int32_t num_args, int32_t id) {
     printf("%*sEntering %s (%p) with %d args\n",
            call_depth * 2, "", name, address, num_args);
     call_depth++;
   }

   void __instrumentor_post_function(char *name, int32_t id) {
     call_depth--;
     printf("%*sExiting %s\n", call_depth * 2, "", name);
   }

GPU Instrumentation
-------------------

The Instrumentor supports GPU targets (AMDGPU and NVPTX). Configure GPU-specific instrumentation:

.. code-block:: json

   {
     "configuration": {
       "runtime_prefix": "__gpu_runtime_",
       "target_regex": "(amdgcn|nvptx).*",
       "host_enabled": false,
       "gpu_enabled": true
     },
     "instruction_pre": {
       "load": {
         "enabled": true,
         "pointer": true,
         "pointer_as": true
       }
     }
   }

Note that GPU runtime functions must be implemented with appropriate device attributes.

Implementation Details
======================

Generated Runtime Function Signatures
--------------------------------------

The Instrumentor generates runtime function names following this pattern:

.. code-block:: text

   <runtime_prefix><position>_<opportunity_name>[_ind]

Where:

- ``<runtime_prefix>``: Configurable prefix (default: ``__instrumentor_``)
- ``<position>``: Either ``pre`` or ``post``
- ``<opportunity_name>``: Name of the instrumentation opportunity (``load``, ``store``, ``function``, etc.)
- ``_ind``: Optional suffix when indirection is used (see below)

Examples:

- ``__instrumentor_pre_load``
- ``__instrumentor_post_store``
- ``__instrumentor_pre_function``
- ``__instrumentor_pre_load_ind`` (with indirection)

Direct vs Indirect Arguments
-----------------------------

The Instrumentor uses two modes for passing arguments:

**Direct mode** (default):
  Arguments are passed by value. This is efficient but requires that all arguments fit in registers or can be passed through the stack efficiently.

**Indirect mode**:
  Arguments are passed by pointer. This is used automatically when:

  - Multiple replaceable arguments are enabled (requires indirection for all replaceable args)
  - An argument's value is too large (aggregate types, large values)

When indirect mode is used, a separate function with the ``_ind`` suffix is generated:

.. code-block:: c

   // Direct mode
   void __instrumentor_pre_load(void *pointer, uint64_t value_size);

   // Indirect mode (automatically generated when needed)
   void __instrumentor_pre_load_ind(void **pointer, uint32_t pointer_size,
                                     void *value_size, uint32_t value_size_size);

Users typically don't need to worry about this - the Instrumentor handles it automatically and the wizard-generated stubs show the correct signatures.

Unique IDs
----------

When the ``id`` argument is enabled, the Instrumentor assigns a unique 32-bit integer to each instrumentation call site:

- PRE positions get positive IDs (1, 2, 3, ...)
- POST positions get negative IDs (-1, -2, -3, ...)
- IDs are consistent across multiple runs

Caching
-------

The Instrumentor caches certain argument values between PRE and POST calls when possible:

- Values computed in PRE are reused in POST (e.g., pointer value)
- This reduces overhead and ensures consistency

Runtime Function Requirements
------------------------------

Runtime functions must be:

- Defined with external linkage
- Fast and non-blocking (to minimize instrumentation overhead)
- Thread-safe if the program is multi-threaded

Runtime functions **must not**:

- Call back into instrumented code (to avoid infinite recursion)

Performance Considerations
==========================

Overhead Factors
----------------

Instrumentation overhead depends on:

1. **Number of instrumentation points**: More instrumented operations = more overhead
2. **Number of arguments passed**: Each argument adds instructions and register pressure
3. **Runtime function complexity**: Complex runtime logic increases overhead
4. **Frequency of instrumented operations**: Instrumenting hot loops has high impact

Optimization Tips
-----------------

**Minimize arguments:**
  Only enable arguments you actually need. Passing fewer arguments reduces overhead.

**Use PRE or POST, not both:**
  If you only need one position, disable the other.

**Target filtering:**
  Use ``target_regex`` to instrument only specific targets or modules.

**Efficient runtime:**
  Keep runtime functions simple and fast. Consider:

  - Lock-free data structures
  - Thread-local storage
  - Batching outputs instead of per-call I/O
  - Sampling (instrument 1 in N calls)

**Build with optimizations:**
  Use ``-O2`` or ``-O3`` when compiling instrumented code. LLVM can optimize away some overhead.

Troubleshooting
===============

Common Issues
-------------

**"Could not find 'opt' binary"**
  The wizard can't locate the opt binary.

  - Specify the path: ``--opt-path /path/to/opt``

**"Indirection needed but not indicated"**
  An argument value is too large for direct passing. The Instrumentor handles this automatically, but you might see this warning. It's usually harmless - the indirect version of the function will be generated.

**Infinite recursion / stack overflow**
  Your runtime function is calling back into instrumented code. Solutions:

  - Ensure runtime functions don't trigger more instrumentation

**Linking errors**
  Runtime functions are undefined. You must:

  - Implement all enabled runtime functions
  - Link the runtime implementation with your program
  - Use the exact function signatures (check generated stubs)

**Unexpected instrumentation**
  More instrumentation than expected. Check:

  - The ``enabled`` flag for each opportunity
  - ``host_enabled`` / ``gpu_enabled`` settings
  - ``target_regex`` matches your target
  - Runtime functions aren't being instrumented (they should be automatically excluded)

Debugging Instrumented Code
----------------------------

**View instrumented IR:**

.. code-block:: bash

   opt -passes=instrumentor \
       -instrumentor-read-config-file=config.json \
       input.ll -S -o output.ll

   # Examine output.ll to see inserted calls

**Print configuration:**

.. code-block:: bash

   opt -passes=instrumentor \
       -instrumentor-write-config-file=debug_config.json \
       input.ll -disable-output

   # Examine debug_config.json to see all options

**Verify IR:**
  The Instrumentor automatically verifies the module after instrumentation. If verification fails, there's a bug in the Instrumentor or the configuration is invalid.

**Use debug builds:**
  Build LLVM with assertions enabled (``-DLLVM_ENABLE_ASSERTIONS=ON``) to catch issues early.

Extending the Instrumentor
===========================

The Instrumentor is designed to be extensible. To add new instrumentation opportunities:

1. **Define the opportunity class** inheriting from ``InstrumentationOpportunity``
2. **Implement getter/setter functions** for the arguments
3. **Add initialization** to populate the opportunity with arguments
4. **Register** the opportunity in ``InstrumentationConfig::populate()``
5. **Add tests** in ``llvm/test/Transforms/Instrumentor/``

See ``llvm/lib/Transforms/IPO/Instrumentor.cpp`` and ``llvm/include/llvm/Transforms/IPO/Instrumentor.h`` for examples (``LoadIO``, ``StoreIO``).

Future instrumentation opportunities being considered:

- Basic block entry/exit
- Branch instrumentation
- Call instructions
- Atomic operations
- Vector operations
- Exception handling
- Global variable access

Reference
=========

Command-Line Options
--------------------

**-instrumentor-read-config-file=<path>**
  Load instrumentation configuration from the specified JSON file.

**-instrumentor-write-config-file=<path>**
  Write the default instrumentation configuration to the specified JSON file (useful for generating templates).

Related Passes
--------------

The Instrumentor is more flexible but related to:

- **AddressSanitizer**: Specialized memory error detector
- **ThreadSanitizer**: Race condition detector
- **MemorySanitizer**: Uninitialized memory detector
- **DataFlowSanitizer**: Taint tracking
- **XRay**: Function call tracing with low overhead

The Instrumentor can implement similar functionality with custom runtime code, but specialized passes may have better performance for their specific use cases.

Further Reading
---------------

- Source code: ``llvm/lib/Transforms/IPO/Instrumentor.cpp``
- Header: ``llvm/include/llvm/Transforms/IPO/Instrumentor.h``
- Configuration wizard: ``llvm/utils/instrumentor-config-wizard.py``
