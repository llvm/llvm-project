include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. 
                     ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMMBlazeAsmParser
  MBlazeAsmLexer.cpp
  MBlazeAsmParser.cpp
  )

add_dependencies(LLVMMBlazeAsmParser MBlazeCommonTableGen)
