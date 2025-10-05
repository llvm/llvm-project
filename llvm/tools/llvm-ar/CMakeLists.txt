set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsInfos
  BinaryFormat
  Core
  DlltoolDriver
  LibDriver
  Object
  Support
  TargetParser
  )

add_llvm_tool(llvm-ar
  llvm-ar.cpp

  DEPENDS
  intrinsics_gen
  GENERATE_DRIVER
  )

setup_host_tool(llvm-ar LLVM_AR llvm_ar_exe llvm_ar_target)

add_llvm_tool_symlink(llvm-ranlib llvm-ar)
add_llvm_tool_symlink(llvm-lib llvm-ar)
add_llvm_tool_symlink(llvm-dlltool llvm-ar)

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(ar llvm-ar)
  add_llvm_tool_symlink(dlltool llvm-ar)
  add_llvm_tool_symlink(ranlib llvm-ar)
endif()
