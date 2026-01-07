set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TosaOpt)
add_mlir_dialect_tablegen_target(MLIRTosaPassIncGen)

add_mlir_doc(Passes TosaPasses ./ -gen-pass-doc)
