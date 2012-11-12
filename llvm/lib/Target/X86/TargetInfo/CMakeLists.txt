include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMX86Info
  X86TargetInfo.cpp
  )

add_dependencies(LLVMX86Info X86CommonTableGen)
