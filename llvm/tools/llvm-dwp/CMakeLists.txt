set(LLVM_LINK_COMPONENTS
  DebugInfoDWARF
  DWP
  Object
  Option
  Support
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(DwpOptsTableGen)

add_llvm_tool(llvm-dwp
  llvm-dwp.cpp

  DEPENDS
  intrinsics_gen
  DwpOptsTableGen
  GENERATE_DRIVER
  )

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(dwp llvm-dwp)
endif()
