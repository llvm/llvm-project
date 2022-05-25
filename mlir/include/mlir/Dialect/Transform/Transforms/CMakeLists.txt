set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transform)
add_public_tablegen_target(MLIRTransformDialectTransformsIncGen)

add_mlir_doc(Passes TransformPasses ./ -gen-pass-doc)
