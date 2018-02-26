set(LLVM_LINK_COMPONENTS
  Demangle
  Support
  )

add_llvm_tool(llvm-cxxfilt
  llvm-cxxfilt.cpp
  )

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(c++filt llvm-cxxfilt)
endif()
