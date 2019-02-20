set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  BinaryFormat
  Core
  DlltoolDriver
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
add_llvm_tool_symlink(llvm-dlltool llvm-ar)

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(ar llvm-ar)
  add_llvm_tool_symlink(dlltool llvm-ar)
  add_llvm_tool_symlink(ranlib llvm-ar)
endif()
