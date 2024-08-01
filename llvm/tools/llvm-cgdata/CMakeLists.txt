set(LLVM_LINK_COMPONENTS
  CodeGen
  CodeGenData
  Core
  Object
  Option
  Support
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(CGDataOptsTableGen)

add_llvm_tool(llvm-cgdata
  llvm-cgdata.cpp

  DEPENDS
  intrinsics_gen
  CGDataOptsTableGen
  GENERATE_DRIVER
  )
