set(LLVM_LINK_COMPONENTS
  Object
  Support
  )

add_llvm_tool(llvm-size
  llvm-size.cpp
  )

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(size llvm-size)
endif()
