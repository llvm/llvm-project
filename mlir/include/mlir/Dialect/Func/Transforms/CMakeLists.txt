set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Func)
add_mlir_dialect_tablegen_target(MLIRFuncTransformsIncGen)

add_mlir_doc(Passes FuncPasses ./ -gen-pass-doc)
