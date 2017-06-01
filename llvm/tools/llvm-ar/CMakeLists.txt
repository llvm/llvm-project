set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  LibDriver
  Object
  Support
  )

add_llvm_tool(llvm-ar
  llvm-ar.cpp

  DEPENDS
  intrinsics_gen
  )

add_llvm_tool_symlink(llvm-ranlib llvm-ar)
add_llvm_tool_symlink(llvm-lib llvm-ar)
