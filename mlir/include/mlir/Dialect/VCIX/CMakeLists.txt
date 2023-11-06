add_mlir_dialect(VCIX vcix)
add_mlir_doc(VCIXOps VCIXOps Dialects/ -gen-dialect-doc -dialect=vcix)

set(LLVM_TARGET_DEFINITIONS VCIXAttrs.td)
mlir_tablegen(VCIXDialectEnums.h.inc -gen-enum-decls)
mlir_tablegen(VCIXDialectEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRVCIXDialectEnumIncGen)
add_dependencies(mlir-headers MLIRVCIXDialectEnumIncGen)
