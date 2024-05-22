add_mlir_dialect_library(MLIRMLProgramTransforms
  PipelineGlobalOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MLProgram/Transforms

  DEPENDS
  MLIRMLProgramPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMLProgramDialect
  MLIRPass
)
