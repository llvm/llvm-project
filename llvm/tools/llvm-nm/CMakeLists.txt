set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsInfos
  Core
  Demangle
  Object
  Support
  )

add_llvm_tool(llvm-nm
  llvm-nm.cpp

  DEPENDS
  intrinsics_gen
  )

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(nm llvm-nm)
endif()
