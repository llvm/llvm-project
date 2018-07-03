if (LLVM_TARGETS_TO_BUILD MATCHES "X86")
  add_subdirectory(X86)
  set(LLVM_EXEGESIS_TARGETS "${LLVM_EXEGESIS_TARGETS} X86" PARENT_SCOPE)
endif()
if (LLVM_TARGETS_TO_BUILD MATCHES "AArch64")
  add_subdirectory(AArch64)
  set(LLVM_EXEGESIS_TARGETS "${LLVM_EXEGESIS_TARGETS} AArch64" PARENT_SCOPE)
endif()

add_library(LLVMExegesis
  STATIC
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
  Target.cpp
  Uops.cpp
  )

llvm_update_compile_flags(LLVMExegesis)
llvm_map_components_to_libnames(libs
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
  )

if(LLVM_ENABLE_LIBPFM AND HAVE_LIBPFM)
  list(APPEND libs pfm)
endif()

target_link_libraries(LLVMExegesis ${libs})
set_target_properties(LLVMExegesis PROPERTIES FOLDER "Libraries")
