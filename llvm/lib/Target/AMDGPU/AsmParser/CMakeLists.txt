add_llvm_library(LLVMAMDGPUAsmParser
  AMDGPUAsmParser.cpp
  )

add_dependencies(LLVMAMDGPUAsmParser LLVMAMDGPUUtils)
