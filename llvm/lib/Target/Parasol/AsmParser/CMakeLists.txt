# Modified by Sunscreen under the AGPLv3 license; see the README at the
# repository root for more information

include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_component_library(LLVMParasolAsmParser
  ParasolAsmParser.cpp

  ADD_TO_COMPONENT
  Parasol
  )

add_dependencies( LLVMParasolAsmParser ParasolCommonTableGen )
