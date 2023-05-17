set(LLVM_LINK_COMPONENTS
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  DebugInfoDWARF
  DWP
  MC
  Object
  Support
  TargetParser
  )

add_llvm_tool(llvm-dwp
  llvm-dwp.cpp

  DEPENDS
  intrinsics_gen
  )

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(dwp llvm-dwp)
endif()
