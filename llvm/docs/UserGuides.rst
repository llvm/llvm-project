User Guides
===========

For those new to the LLVM system.

NOTE: If you are a user who is only interested in using an LLVM-based compiler,
you should look into `Clang <http://clang.llvm.org>`_ instead. The
documentation here is intended for users who have a need to work with the
intermediate LLVM representation.

.. toctree::
   :hidden:

   CMake
   CMakePrimer
   AdvancedBuilds
   HowToBuildOnARM
   HowToBuildWithPGO
   HowToCrossCompileBuiltinsOnArm
   HowToCrossCompileLLVM
   CommandGuide/index
   GettingStarted
   GettingStartedVS
   FAQ
   Lexicon
   HowToAddABuilder
   yaml2obj
   MarkdownQuickstartTemplate
   Phabricator
   TestingGuide
   tutorial/index
   ReleaseNotes
   Passes
   YamlIO
   GetElementPtr
   Frontend/PerformanceTips
   MCJITDesignAndImplementation
   ORCv2
   CodeOfConduct
   CompileCudaWithLLVM
   ReportingGuide
   Benchmarking
   Docker
   BuildingADistribution
   Remarks

:doc:`GettingStarted`
   Discusses how to get up and running quickly with the LLVM infrastructure.
   Everything from unpacking and compilation of the distribution to execution
   of some tools.

:doc:`CMake`
   An addendum to the main Getting Started guide for those using the `CMake
   build system <http://www.cmake.org>`_.

:doc:`HowToBuildOnARM`
   Notes on building and testing LLVM/Clang on ARM.

:doc:`HowToBuildWithPGO`
    Notes on building LLVM/Clang with PGO.

:doc:`HowToCrossCompileBuiltinsOnArm`
   Notes on cross-building and testing the compiler-rt builtins for Arm.

:doc:`HowToCrossCompileLLVM`
   Notes on cross-building and testing LLVM/Clang.

:doc:`GettingStartedVS`
   An addendum to the main Getting Started guide for those using Visual Studio
   on Windows.

:doc:`tutorial/index`
   Tutorials about using LLVM. Includes a tutorial about making a custom
   language with LLVM.

:doc:`LLVM Command Guide <CommandGuide/index>`
   A reference manual for the LLVM command line utilities ("man" pages for LLVM
   tools).

:doc:`Passes`
   A list of optimizations and analyses implemented in LLVM.

:doc:`FAQ`
   A list of common questions and problems and their solutions.

:doc:`Release notes for the current release <ReleaseNotes>`
   This describes new features, known bugs, and other limitations.

:doc:`LLVM Testing Infrastructure Guide <TestingGuide>`
   A reference manual for using the LLVM testing infrastructure.

:doc:`TestSuiteGuide`
  Describes how to compile and run the test-suite benchmarks.

`How to build the C, C++, ObjC, and ObjC++ front end`__
   Instructions for building the clang front-end from source.

   .. __: http://clang.llvm.org/get_started.html

:doc:`Lexicon`
   Definition of acronyms, terms and concepts used in LLVM.

:doc:`HowToAddABuilder`
   Instructions for adding new builder to LLVM buildbot master.

:doc:`YamlIO`
   A reference guide for using LLVM's YAML I/O library.

:doc:`GetElementPtr`
  Answers to some very frequent questions about LLVM's most frequently
  misunderstood instruction.

:doc:`Frontend/PerformanceTips`
   A collection of tips for frontend authors on how to generate IR
   which LLVM is able to effectively optimize.

:doc:`Docker`
   A reference for using Dockerfiles provided with LLVM.

:doc:`BuildingADistribution`
  A best-practices guide for using LLVM's CMake build system to package and
  distribute LLVM-based tools.

:doc:`Remarks`
   A reference on the implementation of remarks in LLVM.