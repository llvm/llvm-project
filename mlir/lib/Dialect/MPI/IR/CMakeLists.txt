add_mlir_dialect_library(MLIRMPIDialect
  MPIOps.cpp
  MPI.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MPI

  DEPENDS
  MLIRMPIIncGen
  MLIRMPIOpsIncGen
  MLIRMPITypesIncGen
  MLIRMPIAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  )
