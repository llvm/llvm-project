add_mlir_dialect(GENOps gen)
add_mlir_doc(GENDialect GENDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(GENOps GENOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS GENOps.td)
mlir_tablegen(GENOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(GENOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRGENOpsEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS GENAttrDefs.td)
mlir_tablegen(GENOpsAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(GENOpsAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRGENOpsAttrDefsIncGen)
