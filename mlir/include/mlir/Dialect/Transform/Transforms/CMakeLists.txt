set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transform)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Transform)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Transform)
add_mlir_dialect_tablegen_target(MLIRTransformDialectTransformsIncGen)

add_mlir_doc(Passes TransformPasses ./ -gen-pass-doc)
