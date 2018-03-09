set(LLVM_LINK_COMPONENTS
  Core
  Object
  Support
  )

add_llvm_tool(llvm-strings
  llvm-strings.cpp
  )

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(strings llvm-strings)
endif()
