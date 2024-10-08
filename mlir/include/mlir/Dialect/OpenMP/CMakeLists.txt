set(LLVM_TARGET_DEFINITIONS ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend/OpenMP/OMP.td)
mlir_tablegen(OmpCommon.td --gen-directive-decl --directives-dialect=OpenMP)
add_public_tablegen_target(omp_common_td)

set(LLVM_TARGET_DEFINITIONS OpenMPOps.td)

# The OpenMP verifier tablegen pseudo-backend does not produce any output, but
# mlir_tablegen expects an output file name to be passed. An empty "no-output"
# file is created by the statement below.
#
# This output will be added to the list of dependencies of the
# MLIROpenMPOpsIncGen target below, which results in triggering this
# verification pass every time OpenMPOps.td is modified and recompiled.
mlir_tablegen(no-output -verify-openmp-ops)

mlir_tablegen(OpenMPOpsDialect.h.inc -gen-dialect-decls -dialect=omp)
mlir_tablegen(OpenMPOpsDialect.cpp.inc -gen-dialect-defs -dialect=omp)
mlir_tablegen(OpenMPOps.h.inc -gen-op-decls)
mlir_tablegen(OpenMPOps.cpp.inc -gen-op-defs)
mlir_tablegen(OpenMPClauseOps.h.inc -gen-openmp-clause-ops)
mlir_tablegen(OpenMPOpsTypes.h.inc -gen-typedef-decls -typedefs-dialect=omp)
mlir_tablegen(OpenMPOpsTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=omp)
mlir_tablegen(OpenMPOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpenMPOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(OpenMPOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=omp)
mlir_tablegen(OpenMPOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=omp)
add_mlir_doc(OpenMPOps OpenMPDialect Dialects/ -gen-dialect-doc -dialect=omp)
add_public_tablegen_target(MLIROpenMPOpsIncGen)
add_dependencies(OpenMPDialectDocGen omp_common_td)
add_mlir_interface(OpenMPOpsInterfaces)

set(LLVM_TARGET_DEFINITIONS OpenMPTypeInterfaces.td)
mlir_tablegen(OpenMPTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(OpenMPTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIROpenMPTypeInterfacesIncGen)
add_dependencies(mlir-generic-headers MLIROpenMPTypeInterfacesIncGen)
