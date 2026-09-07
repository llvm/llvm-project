add_mlir_dialect_library(MLIRArmNeonDialect
  ArmNeonDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ArmNeon

  DEPENDS
  MLIRArithOpsIncGen
  MLIRArmNeonIncGen
  MLIRVectorIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  )
