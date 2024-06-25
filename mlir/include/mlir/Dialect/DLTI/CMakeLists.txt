add_mlir_dialect(DLTI dlti)
add_mlir_doc(DLTI DLTIDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS DLTIAttrs.td)
mlir_tablegen(DLTIAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=dlti)
mlir_tablegen(DLTIAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=dlti)
add_public_tablegen_target(MLIRDLTIAttrsIncGen)
add_dependencies(mlir-headers MLIRDLTIAttrsIncGen)
