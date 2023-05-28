set(LLVM_LINK_COMPONENTS
  Option
  Support
  )
set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(DebugInfodOptsTableGen)

add_llvm_tool(llvm-debuginfod
  llvm-debuginfod.cpp

  DEPENDS
  DebugInfodOptsTableGen
  )
target_link_libraries(llvm-debuginfod PRIVATE LLVMDebuginfod)
if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(debuginfod llvm-debuginfod)
endif()
