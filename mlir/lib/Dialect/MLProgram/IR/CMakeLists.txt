add_mlir_dialect_library(MLIRMLProgram
  MLProgramOps.cpp
  MLProgramDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MLProgram

  DEPENDS
  MLIRMLProgramOpsIncGen
  MLIRMLProgramAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRInferTypeOpInterface
  MLIRIR
  )
