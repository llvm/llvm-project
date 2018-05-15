add_library(LLVMExegesis
  STATIC
  Analysis.cpp
  BenchmarkResult.cpp
  BenchmarkRunner.cpp
  Clustering.cpp
  InMemoryAssembler.cpp
  InstructionSnippetGenerator.cpp
  Latency.cpp
  LlvmState.cpp
  OperandGraph.cpp
  PerfHelper.cpp
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
  MCJIT
  Object
  Support
  )

target_link_libraries(LLVMExegesis ${libs})
set_target_properties(LLVMExegesis PROPERTIES FOLDER "Libraries")
