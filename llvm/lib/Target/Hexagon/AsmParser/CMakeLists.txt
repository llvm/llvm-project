include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_component_library(LLVMHexagonAsmParser
  HexagonAsmParser.cpp
  )

add_dependencies( LLVMHexagonAsmParser HexagonCommonTableGen )
