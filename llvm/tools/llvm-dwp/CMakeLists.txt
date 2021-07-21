set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  )

add_llvm_tool(llvm-dwp
  llvm-dwp.cpp
  DWP.cpp
  DWPError.cpp

  DEPENDS
  intrinsics_gen
  )

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(dwp llvm-dwp)
endif()
