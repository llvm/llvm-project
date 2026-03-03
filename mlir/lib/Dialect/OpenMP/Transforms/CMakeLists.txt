add_mlir_dialect_library(MLIROpenMPTransforms
  OpenMPOffloadPrivatizationPrepare.cpp
  StackToShared.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenMP

  DEPENDS
  omp_gen
  MLIROpenMPPassIncGen
  MLIROpenMPOpsIncGen
  MLIROpenMPOpsInterfacesIncGen
  MLIROpenMPTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIROpenACCMPCommon
  MLIROpenMPDialect
  MLIROpenMPUtils
  MLIRPass
  MLIRSupport
  MLIRTransforms
  )
