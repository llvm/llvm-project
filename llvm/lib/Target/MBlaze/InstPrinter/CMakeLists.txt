include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. 
                     ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMMBlazeAsmPrinter
  MBlazeInstPrinter.cpp
  )

add_dependencies(LLVMMBlazeAsmPrinter MBlazeCommonTableGen)
