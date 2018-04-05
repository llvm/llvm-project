if(LLVM_TARGETS_TO_BUILD MATCHES "X86")
  add_subdirectory(
    llvm-cfi-verify
  )
  if(LLVM_HOST_TRIPLE MATCHES "x86_64")
    add_subdirectory(
      llvm-exegesis
    )
  endif()
endif()

