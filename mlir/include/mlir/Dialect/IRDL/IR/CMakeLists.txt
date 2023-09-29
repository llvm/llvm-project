add_mlir_dialect(IRDL irdl)
add_mlir_doc(IRDL IRDL Dialects/ -gen-dialect-doc)

# Add IRDL interfaces
set(LLVM_TARGET_DEFINITIONS IRDLInterfaces.td)
mlir_tablegen(IRDLInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(IRDLInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRIRDLInterfacesIncGen)
add_dependencies(mlir-generic-headers MLIRIRDLInterfacesIncGen)

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

# Add IRDL attributes
set(LLVM_TARGET_DEFINITIONS IRDLAttributes.td)
mlir_tablegen(IRDLEnums.h.inc -gen-enum-decls)
mlir_tablegen(IRDLEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(IRDLAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(IRDLAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRIRDLAttributesIncGen)
add_dependencies(mlir-generic-headers MLIRIRDLAttributesIncGen)
