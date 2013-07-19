include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMAArch64Utils
  AArch64BaseInfo.cpp
  )

add_dependencies(LLVMAArch64Utils AArch64CommonTableGen)
