set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Shape)
add_mlir_dialect_tablegen_target(MLIRShapeTransformsIncGen)

add_mlir_doc(Passes ShapePasses ./ -gen-pass-doc)
