set(LLVM_TARGET_DEFINITIONS VectorTransformsBase.td)
mlir_tablegen(VectorTransformsEnums.h.inc -gen-enum-decls)
mlir_tablegen(VectorTransformsEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Vector)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Vector)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Vector)
add_mlir_dialect_tablegen_target(MLIRVectorTransformsIncGen)

add_mlir_doc(Passes VectorPasses ./ -gen-pass-doc)
