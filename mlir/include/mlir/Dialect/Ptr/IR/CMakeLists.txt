add_mlir_dialect(PtrOps ptr)
add_mlir_doc(PtrOps PtrOps Dialects/ -gen-dialect-doc -dialect=ptr)

set(LLVM_TARGET_DEFINITIONS PtrOps.td)
mlir_tablegen(PtrOpsAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=ptr)
mlir_tablegen(PtrOpsAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ptr)
add_public_tablegen_target(MLIRPtrOpsAttributesIncGen)
