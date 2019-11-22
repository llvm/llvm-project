include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_component_library(LLVMAArch64Disassembler
  AArch64Disassembler.cpp
  AArch64ExternalSymbolizer.cpp
  )

add_dependencies(LLVMAArch64Disassembler AArch64CommonTableGen)
