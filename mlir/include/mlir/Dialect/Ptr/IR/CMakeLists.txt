add_mlir_dialect(PtrOps ptr)
add_mlir_doc(PtrOps PtrOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS MemorySpaceInterfaces.td)
mlir_tablegen(MemorySpaceInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(MemorySpaceInterfaces.cpp.inc -gen-op-interface-defs)
mlir_tablegen(MemorySpaceAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(MemorySpaceAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRPtrMemorySpaceInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS PtrOps.td)
mlir_tablegen(PtrOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(PtrOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRPtrOpsEnumsGen)
