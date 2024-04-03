add_mlir_dialect(PtrOps ptr)
add_mlir_doc(PtrOps PtrOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS MemorySpaceInterfaces.td)
mlir_tablegen(MemorySpaceInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(MemorySpaceInterfaces.cpp.inc -gen-op-interface-defs)
mlir_tablegen(MemorySpaceAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(MemorySpaceAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRPtrMemorySpaceInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS PtrInterfaces.td)
mlir_tablegen(PtrInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(PtrInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRPtrInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS PtrOps.td)
mlir_tablegen(PtrOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(PtrOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRPtrOpsEnumsGen)

set(LLVM_TARGET_DEFINITIONS PtrOps.td)
mlir_tablegen(PtrOpsAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=ptr)
mlir_tablegen(PtrOpsAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ptr)
add_public_tablegen_target(MLIRPtrOpsAttributesIncGen)
