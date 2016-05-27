add_llvm_library(LLVMAMDGPUAsmPrinter
  AMDGPUInstPrinter.cpp
  )

add_dependencies(LLVMAMDGPUAsmPrinter LLVMAMDGPUUtils)
