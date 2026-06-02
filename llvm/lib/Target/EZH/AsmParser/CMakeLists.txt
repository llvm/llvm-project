include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_component_library(LLVMEZHAsmParser
  EZHAsmParser.cpp

  LINK_COMPONENTS
  CodeGenTypes
  EZHDesc
  EZHInfo
  MC
  MCParser
  Support

  ADD_TO_COMPONENT
  EZH
  )

add_dependencies( LLVMEZHAsmParser EZHCommonTableGen )
