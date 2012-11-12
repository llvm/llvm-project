include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMX86Utils
  X86ShuffleDecode.cpp
  )

add_dependencies(LLVMX86Utils X86CommonTableGen)
