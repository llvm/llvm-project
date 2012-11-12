include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMMSP430AsmPrinter
  MSP430InstPrinter.cpp
  )

add_dependencies(LLVMMSP430AsmPrinter MSP430CommonTableGen)
