set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/include/llvm/CodeGen/ValueTypes.td)
llvm_tablegen(GenVT.inc -gen-vt)
add_public_tablegen_target(vt_gen)
