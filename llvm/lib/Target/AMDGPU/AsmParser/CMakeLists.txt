add_llvm_component_library(LLVMAMDGPUAsmParser
  AMDGPUAsmParser.cpp
  )

add_dependencies(LLVMAMDGPUAsmParser LLVMAMDGPUUtils)
