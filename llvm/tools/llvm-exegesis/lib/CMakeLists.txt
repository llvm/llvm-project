if (LLVM_TARGETS_TO_BUILD MATCHES "X86")
  add_subdirectory(X86)
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
  X86.cpp
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
