if (LLVM_TARGETS_TO_BUILD MATCHES "X86")
  add_subdirectory(X86)
  set(LLVM_EXEGESIS_TARGETS "${LLVM_EXEGESIS_TARGETS} X86" PARENT_SCOPE)
endif()
if (LLVM_TARGETS_TO_BUILD MATCHES "AArch64")
  add_subdirectory(AArch64)
  set(LLVM_EXEGESIS_TARGETS "${LLVM_EXEGESIS_TARGETS} AArch64" PARENT_SCOPE)
endif()

set(libs)
if(LLVM_ENABLE_LIBPFM AND HAVE_LIBPFM)
  list(APPEND libs pfm)
endif()

add_llvm_library(LLVMExegesis
  Analysis.cpp
  Assembler.cpp
  BenchmarkResult.cpp
  BenchmarkRunner.cpp
  Clustering.cpp
  Latency.cpp
  LlvmState.cpp
  MCInstrDescView.cpp
  PerfHelper.cpp
  RegisterAliasing.cpp
  SnippetGenerator.cpp
  Target.cpp
  Uops.cpp

  STATIC

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  CodeGen
  Core
  ExecutionEngine
  GlobalISel
  MC
  MCDisassembler
  MCJIT
  Object
  ObjectYAML
  Support

  LINK_LIBS ${libs}
  )
