if("ARM" IN_LIST LLVM_TARGETS_TO_BUILD)
  add_subdirectory(ARM)
endif()

if("X86" IN_LIST LLVM_TARGETS_TO_BUILD)
  add_subdirectory(x86)
endif()
