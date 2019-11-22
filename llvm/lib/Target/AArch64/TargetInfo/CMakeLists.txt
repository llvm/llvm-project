include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_component_library(LLVMAArch64Info
  AArch64TargetInfo.cpp
  )

add_dependencies(LLVMAArch64Info AArch64CommonTableGen)
