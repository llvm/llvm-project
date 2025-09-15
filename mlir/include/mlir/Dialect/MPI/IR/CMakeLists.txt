add_mlir_dialect(MPI mpi)
add_mlir_doc(MPIOps MPI Dialects/ -gen-dialect-doc)

# Add MPI operations
set(LLVM_TARGET_DEFINITIONS MPIOps.td)
mlir_tablegen(MPIOps.h.inc -gen-op-decls)
mlir_tablegen(MPIOps.cpp.inc -gen-op-defs)
add_mlir_dialect_tablegen_target(MLIRMPIOpsIncGen)

# Add MPI types
set(LLVM_TARGET_DEFINITIONS MPITypes.td)
mlir_tablegen(MPITypesGen.h.inc -gen-typedef-decls)
mlir_tablegen(MPITypesGen.cpp.inc -gen-typedef-defs)
add_mlir_dialect_tablegen_target(MLIRMPITypesIncGen)

# Add MPI attributes
set(LLVM_TARGET_DEFINITIONS MPI.td)
mlir_tablegen(MPIEnums.h.inc -gen-enum-decls)
mlir_tablegen(MPIEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(MPIAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(MPIAttrDefs.cpp.inc -gen-attrdef-defs)
add_mlir_dialect_tablegen_target(MLIRMPIAttrsIncGen)
