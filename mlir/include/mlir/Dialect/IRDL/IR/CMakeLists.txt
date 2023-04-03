add_mlir_dialect(IRDL irdl)

# Add IRDL operations
set(LLVM_TARGET_DEFINITIONS IRDLOps.td)
mlir_tablegen(IRDLOps.h.inc -gen-op-decls)
mlir_tablegen(IRDLOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRIRDLOpsIncGen)
add_dependencies(mlir-generic-headers MLIRIRDLOpsIncGen)

# Add IRDL types
set(LLVM_TARGET_DEFINITIONS IRDLTypes.td)
mlir_tablegen(IRDLTypesGen.h.inc -gen-typedef-decls)
mlir_tablegen(IRDLTypesGen.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRIRDLTypesIncGen)
add_dependencies(mlir-generic-headers MLIRIRDLTypesIncGen)
