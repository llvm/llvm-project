set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Func)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Func)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Func)
add_mlir_dialect_tablegen_target(MLIRFuncTransformsIncGen)

add_mlir_doc(Passes FuncPasses ./ -gen-pass-doc)
