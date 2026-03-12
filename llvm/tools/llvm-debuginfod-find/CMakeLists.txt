set(LLVM_LINK_COMPONENTS
  Option
  Object
  Support
  )
set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(DebugInfodFindOptsTableGen)

add_llvm_tool(llvm-debuginfod-find
  llvm-debuginfod-find.cpp
  DEPENDS
  DebugInfodFindOptsTableGen
  GENERATE_DRIVER
  )
if(NOT LLVM_TOOL_LLVM_DRIVER_BUILD)
  target_link_libraries(llvm-debuginfod-find PRIVATE LLVMDebuginfod)
endif()
if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(debuginfod-find llvm-debuginfod-find)
endif()
