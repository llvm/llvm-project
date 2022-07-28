add_mlir_dialect(ShapeOps shape)
add_mlir_doc(ShapeOps ShapeDialectOps Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS ShapeBase.td)
mlir_tablegen(ShapeOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(ShapeOpsAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRShapeOpsAttrIncGen)
