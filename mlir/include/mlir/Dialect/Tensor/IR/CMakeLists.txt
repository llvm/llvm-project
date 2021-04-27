add_mlir_dialect(TensorOps tensor)
add_mlir_doc(TensorOps TensorOps Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS TensorAttrDefs.td)
mlir_tablegen(TensorAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TensorAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRTensorAttrDefsIncGen)
