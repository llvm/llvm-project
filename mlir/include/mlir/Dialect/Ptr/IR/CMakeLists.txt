add_mlir_dialect(PtrOps ptr)
add_mlir_doc(PtrOps PtrOps Dialects/ -gen-dialect-doc -dialect=ptr)

set(LLVM_TARGET_DEFINITIONS PtrOps.td)
mlir_tablegen(PtrOpsAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=ptr)
mlir_tablegen(PtrOpsAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ptr)
add_mlir_dialect_tablegen_target(MLIRPtrOpsAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS MemorySpaceInterfaces.td)
mlir_tablegen(MemorySpaceAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(MemorySpaceAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_mlir_dialect_tablegen_target(MLIRPtrMemorySpaceInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS PtrOps.td)
mlir_tablegen(PtrOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(PtrOpsEnums.cpp.inc -gen-enum-defs)
add_mlir_dialect_tablegen_target(MLIRPtrOpsEnumsGen)
