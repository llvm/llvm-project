add_mlir_dialect(PartitionOps partition)
set(LLVM_TARGET_DEFINITIONS PartitionAttrDefs.td)
mlir_tablegen(PartitionAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(PartitionAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRPartitionAttrDefsIncGen)

# add_mlir_doc(SparseTensorOps SparseTensorOps Dialects/ -gen-dialect-doc)
