macro(set_llvm_target_triple)
  set(LLVM_DEFAULT_TARGET_TRIPLE "${LLVM_DEFAULT_TARGET_TRIPLE_default}" CACHE STRING
  "Default target for which LLVM will generate code." )
  if (TARGET_TRIPLE)
    message(WARNING "TARGET_TRIPLE is deprecated and will be removed in a future release. "
    "Please use LLVM_DEFAULT_TARGET_TRIPLE instead.")
    set(LLVM_TARGET_TRIPLE "${TARGET_TRIPLE}")
  else()
    set(LLVM_TARGET_TRIPLE "${LLVM_DEFAULT_TARGET_TRIPLE}")
  endif()
  message(STATUS "LLVM host triple: ${LLVM_HOST_TRIPLE}")
  message(STATUS "LLVM default target triple: ${LLVM_DEFAULT_TARGET_TRIPLE}")
endmacro()
