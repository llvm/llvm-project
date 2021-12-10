set(LLVM_LINK_COMPONENTS
  Debuginfod
  Support
  )
add_llvm_tool(llvm-debuginfod-find
  llvm-debuginfod-find.cpp
  )
if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(debuginfod-find llvm-debuginfod-find)
endif()
