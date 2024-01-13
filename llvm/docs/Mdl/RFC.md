
# MDL: A Micro-Architecture Description Language for LLVM

November 2022                                   Reid Tatge        [tatge@google.com](mailto:tatge@google.com)

Updated January 2024


## **TL;DR:**

We’ve created a DSL and compiler for modeling micro-architecture that handles a very broad class of architectures - CPUs, GPUs, VLIWs, DSPs, ML accelerators, and embedded devices. This effort grew out of a need to quickly develop and experiment with high-quality compilers and tools to facilitate rapid architecture exploration. We named the DSL “MDL” for “Microarchitecture Description Language”.

While being significantly more expressive than TableGen’s Schedules and Itineraries used in LLVM, MDL is also more concise, and simpler to read and write while supporting a much broader class of embedded and accelerator architectures. We currently can automatically _generate _MDL descriptions for all upstream targets which are in many cases 1/10 the size of the equivalent TableGen descriptions.  We’ve integrated this with LLVM, and are sending out this RFC because we believe it could be valuable to the larger LLVM community. \


The MDL compiler, associated tools, and documentation are available as open source (at https://github.com/MPACT-ORG/llvm-project/tree/all), and we would like to explore adding this to the LLVM project, and encourage contributions from others.


## **Background**

Over the last few years, we have been using LLVM to develop a compiler backend for Google’s TPU machine learning accelerators.  TPUs have complex microarchitectures and pose a number of challenges that are not seen in in typical LLVM targets:



*   Clustered VLIW with partitioned register files.
*   Extremely deep pipelines with complex hazard conditions
*   Instructions with functional-unit-specific and/or cluster-specific behaviors
    *   Non-trivial and/or instance-specific latencies
    *   Complex resource usage
    *   Functional-unit-specific register constraints
*   Shared/allocated encoding resources (instructions need 1..M of N resources)
*   Explicitly managed hardware resources (register ports, internal datapaths, busses, etc)

While some of these problems manifest in a few upstream LLVM targets, this collection of problems is a superset of the problems directly addressed by LLVM - Schedules and Itineraries are simply not sufficient to model everything. Supporting this class of architecture is therefore code-intensive - it takes around 20,000 lines of C++ code to model the TPU sub-targets. This is brittle, hard to write, debug, test, and evolve over time. In contrast, the MDL description for these sub-targets is ~2,000 lines of text, and requires very little (if any) target-specific code in the backend.


## **Status**



*   We’ve created the MDL language and compiler for describing microarchitecture details, a methodology for integrating it with TableGen files for any target, and a set of APIs that can be used in a machine-independent way to inform back-end passes such as bundle-packing, instruction scheduling, and register allocation. 
*   To facilitate integration with LLVM, we built a tool which scrapes architectural information from TableGen files, and produces our MDL language for all upstream targets.
*   We’ve modified the CodeGen and MC libraries to (optionally) use our methodology for latency management.


## **Building**



*   You can build llvm with or without MDL support.  It is included by using the LLVM\_ENABLE\_MDL CMake parameter.  If included, it is currently used by default, and can be disabled with a command line option (--schedmdl=0).


## **Testing**



*   When built without MDL support, the compiler passes all check-all tests.
*   When built with MDL support, but disabled on the command line, the compielr passes all check-all tests.
*   When MDL support is enabled, it passs all but 190 tests (out of ~90K+ tests). 

There is a lot more to do. For example, we plan to enhance existing back-end scheduling passes and register allocation passes to cleanly handle a larger class of embedded and accelerator architectures, based on MDL-generated information.

We welcome feedback on the language design and associated tools and use model.  You can find the MDL design documentation, compiler, and other tools in our github repo in llvm/docs/mdl.

