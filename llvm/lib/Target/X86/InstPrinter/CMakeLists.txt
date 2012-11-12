include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMX86AsmPrinter
  X86ATTInstPrinter.cpp
  X86IntelInstPrinter.cpp
  X86InstComments.cpp
  )

add_dependencies(LLVMX86AsmPrinter X86CommonTableGen)
