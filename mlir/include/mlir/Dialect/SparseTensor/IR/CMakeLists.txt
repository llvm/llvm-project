add_mlir_dialect(SparseTensorOps sparse_tensor)
add_mlir_doc(SparseTensorOps SparseTensorOps Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS SparseTensorAttrDefs.td)
mlir_tablegen(SparseTensorAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(SparseTensorAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRSparseTensorAttrDefsIncGen)
