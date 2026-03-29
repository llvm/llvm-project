add_subdirectory(IR)
add_subdirectory(Transforms)
add_subdirectory(Utils)

add_mlir_dialect_library(ConflictMLIROpenMPDialect
  IR/OpenMPDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenMP

  DEPENDS
  omp_gen
  MLIROpenMPOpsIncGen
  MLIROpenMPOpsInterfacesIncGen
  MLIROpenMPTypeInterfacesIncGen

  LINK_COMPONENTS
  TargetParser

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRFuncDialect
  MLIROpenACCMPCommon
  )
