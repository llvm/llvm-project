set(LLVM_TARGET_DEFINITIONS VectorTransformsBase.td)
mlir_tablegen(VectorTransformsEnums.h.inc -gen-enum-decls)
mlir_tablegen(VectorTransformsEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Vector)
add_mlir_dialect_tablegen_target(MLIRVectorTransformsIncGen)

add_mlir_doc(Passes VectorPasses ./ -gen-pass-doc)
