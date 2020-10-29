set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TosaOpt)
add_public_tablegen_target(MLIRTosaPassIncGen)
add_dependencies(mlir-headers MLIRTosaPassIncGen)

add_mlir_doc(Passes -gen-pass-doc TosaPasses ./)
