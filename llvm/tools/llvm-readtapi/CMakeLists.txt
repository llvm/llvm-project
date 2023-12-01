set(LLVM_LINK_COMPONENTS
  Object
  Support
  Option
  TextAPI
  )

set(LLVM_TARGET_DEFINITIONS TapiOpts.td)
tablegen(LLVM TapiOpts.inc -gen-opt-parser-defs)
add_public_tablegen_target(ReadTAPIOptsTableGen)

add_llvm_tool(llvm-readtapi
  llvm-readtapi.cpp
  DiffEngine.cpp

  DEPENDS
  ReadTAPIOptsTableGen
  )

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(readtapi llvm-readtapi)
endif()
