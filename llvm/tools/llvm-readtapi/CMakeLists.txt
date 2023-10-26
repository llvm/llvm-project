set(LLVM_LINK_COMPONENTS
  Object
  Support
  TextAPI
  )

add_llvm_tool(llvm-readtapi
  llvm-readtapi.cpp
  DiffEngine.cpp
  )

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(readtapi llvm-readtapi)
endif()
