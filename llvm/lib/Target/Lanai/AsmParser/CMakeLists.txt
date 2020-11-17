include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_component_library(LLVMLanaiAsmParser
  LanaiAsmParser.cpp

  LINK_COMPONENTS
  MC
  MCParser
  Support
  LanaiDesc
  LanaiInfo

  ADD_TO_COMPONENT
  Lanai
  )

add_dependencies( LLVMLanaiAsmParser LanaiCommonTableGen )
