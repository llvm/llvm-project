set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)

tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(RcTableGen)

add_llvm_tool(llvm-rc
  llvm-rc.cpp
  ResourceScriptToken.cpp
  )
