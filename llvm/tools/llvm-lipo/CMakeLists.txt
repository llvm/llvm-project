set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Object
  Option
  Support
)

set(LLVM_TARGET_DEFINITIONS LipoOpts.td)
tablegen(LLVM LipoOpts.inc -gen-opt-parser-defs)
add_public_tablegen_target(LipoOptsTableGen)

add_llvm_tool(llvm-lipo
  llvm-lipo.cpp
  DEPENDS
  LipoOptsTableGen
)
