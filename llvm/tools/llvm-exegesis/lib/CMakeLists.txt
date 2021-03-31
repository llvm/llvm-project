set(TARGETS_TO_APPEND "")

if (LLVM_TARGETS_TO_BUILD MATCHES "X86")
  add_subdirectory(X86)
  set(TARGETS_TO_APPEND "${TARGETS_TO_APPEND} X86")
endif()
if (LLVM_TARGETS_TO_BUILD MATCHES "AArch64")
  add_subdirectory(AArch64)
  set(TARGETS_TO_APPEND "${TARGETS_TO_APPEND} AArch64")
endif()
if (LLVM_TARGETS_TO_BUILD MATCHES "PowerPC")
  add_subdirectory(PowerPC)
  set(TARGETS_TO_APPEND "${TARGETS_TO_APPEND} PowerPC")
endif()
if (LLVM_TARGETS_TO_BUILD MATCHES "Mips")
  add_subdirectory(Mips)
  set(TARGETS_TO_APPEND "${TARGETS_TO_APPEND} Mips")
endif()

set(LLVM_EXEGESIS_TARGETS "${LLVM_EXEGESIS_TARGETS} ${TARGETS_TO_APPEND}" PARENT_SCOPE)

set(LLVM_LINK_COMPONENTS
  Analysis
  CodeGen
  Core
  ExecutionEngine
  GlobalISel
  MC
  MCDisassembler
  MCJIT
  MCParser
  Object
  ObjectYAML
  RuntimeDyld
  Support
  )

set(libs)
if(LLVM_ENABLE_LIBPFM AND HAVE_LIBPFM)
  list(APPEND libs pfm)
endif()

add_llvm_library(LLVMExegesis
  DISABLE_LLVM_LINK_LLVM_DYLIB
  STATIC
  Analysis.cpp
  Assembler.cpp
  BenchmarkResult.cpp
  BenchmarkRunner.cpp
  Clustering.cpp
  CodeTemplate.cpp
  Error.cpp
  LatencyBenchmarkRunner.cpp
  LlvmState.cpp
  MCInstrDescView.cpp
  ParallelSnippetGenerator.cpp
  PerfHelper.cpp
  RegisterAliasing.cpp
  RegisterValue.cpp
  SchedClassResolution.cpp
  SerialSnippetGenerator.cpp
  SnippetFile.cpp
  SnippetGenerator.cpp
  SnippetRepetitor.cpp
  Target.cpp
  UopsBenchmarkRunner.cpp

  LINK_LIBS ${libs}

  DEPENDS
  intrinsics_gen
  )
