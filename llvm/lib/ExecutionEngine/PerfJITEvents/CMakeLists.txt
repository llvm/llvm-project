add_llvm_component_library(LLVMPerfJITEvents
  PerfJITEventListener.cpp
  )

add_dependencies(LLVMPerfJITEvents LLVMCodeGen)
