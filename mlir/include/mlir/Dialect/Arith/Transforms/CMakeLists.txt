set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Arith)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Arith)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Arith)
add_mlir_dialect_tablegen_target(MLIRArithTransformsIncGen)

add_mlir_doc(Passes ArithPasses ./ -gen-pass-doc)
