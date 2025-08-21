add_mlir_dialect(MemRefOps memref)
add_mlir_doc(MemRefOps MemRefOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS MemRefOps.td)
mlir_tablegen(MemRefAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=memref)
mlir_tablegen(MemRefAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=memref)
add_public_tablegen_target(MLIRMemRefAttrsIncGen)
add_dependencies(mlir-headers MLIRMemRefAttrsIncGen)
