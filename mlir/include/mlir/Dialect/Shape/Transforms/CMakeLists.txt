set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Shape)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Shape)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Shape)
add_mlir_dialect_tablegen_target(MLIRShapeTransformsIncGen)

add_mlir_doc(Passes ShapePasses ./ -gen-pass-doc)
