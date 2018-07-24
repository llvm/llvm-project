add_llvm_library(LLVMPerfJITEvents
  PerfJITEventListener.cpp
  )

add_dependencies(LLVMPerfJITEvents LLVMCodeGen)
