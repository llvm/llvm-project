add_mlir_dialect(AGX agx)
add_mlir_doc(AGX AGX Dialects/ -gen-dialect-doc -dialect agx)

set(LLVM_TARGET_DEFINITIONS AGX.td)
mlir_tablegen(AGXEnums.h.inc -gen-enum-decls)
mlir_tablegen(AGXEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(AGXAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=agx)
mlir_tablegen(AGXAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=agx)
add_mlir_dialect_tablegen_target(MLIRAGXEnumsGen)
