set(LLVM_LINK_COMPONENTS
  Object
  Option
  Support
  MC
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)

tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(ObjcopyTableGen)

add_llvm_tool(llvm-objcopy
  llvm-objcopy.cpp
  Object.cpp
  DEPENDS
  ObjcopyTableGen
  )

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(objcopy llvm-objcopy)
endif()
