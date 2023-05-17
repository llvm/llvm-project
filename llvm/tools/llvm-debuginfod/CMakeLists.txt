set(LLVM_LINK_COMPONENTS
  Support
  )
add_llvm_tool(llvm-debuginfod
  llvm-debuginfod.cpp
  )
target_link_libraries(llvm-debuginfod PRIVATE LLVMDebuginfod)
if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(debuginfod llvm-debuginfod)
endif()
