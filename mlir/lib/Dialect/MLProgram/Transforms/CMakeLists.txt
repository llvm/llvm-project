add_mlir_dialect_library(MLIRMLProgramTransforms
  BufferizableOpInterfaceImpl.cpp
  ExternalizeLargeConstants.cpp
  PipelineGlobalOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MLProgram/Transforms

  DEPENDS
  MLIRMLProgramPassIncGen

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRBytecodeWriter
  MLIRIR
  MLIRMLProgramDialect
  MLIRParser
  MLIRPass
  MLIRSupport
)
