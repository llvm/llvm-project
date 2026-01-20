set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Math)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Math)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Math)
add_mlir_dialect_tablegen_target(MLIRMathTransformsIncGen)

add_mlir_doc(Passes MathPasses ./ -gen-pass-doc)
