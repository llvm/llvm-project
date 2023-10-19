add_mlir_dialect(PartTensorOps part_tensor)
set(LLVM_TARGET_DEFINITIONS PartTensorAttrDefs.td)
mlir_tablegen(PartTensorAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(PartTensorAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRPartTensorOpsAttributesIncGen)
add_public_tablegen_target(MLIRPartTensorAttrDefsIncGen)

set(LLVM_TARGET_DEFINITIONS PartTensorTypes.td)
mlir_tablegen(PartTensorTypes.h.inc -gen-typedef-decls)
mlir_tablegen(PartTensorTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRPartTensorTypesIncGen)

add_mlir_doc(PartTensorOps PartTensorOps Dialects/ -gen-dialect-doc)
