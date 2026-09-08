add_mlir_dialect_library(MLIRArmNeonDialect
  ArmNeonDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ArmNeon

  DEPENDS
  MLIRArmNeonIncGen
  MLIRArmNeonConversionsIncGen

  # ArmNeonDialect.cpp includes VectorOps.h, whose generated declarations
  # expose both Vector and Arith generated headers.
  HEADER_LIBS
  MLIRArithDialect
  MLIRVectorDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  )
