if ("AArch64" IN_LIST LLVM_TARGETS_TO_BUILD)
  add_subdirectory(ARM64)
endif()

if ("PowerPC" IN_LIST LLVM_TARGETS_TO_BUILD)
  add_subdirectory(PPC64)
endif()

if ("X86" IN_LIST LLVM_TARGETS_TO_BUILD)
  add_subdirectory(x86)
endif()
