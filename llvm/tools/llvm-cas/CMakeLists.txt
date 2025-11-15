set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(LLVMCASToolTableGen)

set(LLVM_LINK_COMPONENTS
  Support
  CAS
  Option
  )

add_llvm_tool(llvm-cas
  llvm-cas.cpp

  DEPENDS
  ${tablegen_deps}
  LLVMCASToolTableGen
  )
