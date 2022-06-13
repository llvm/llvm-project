add_mlir_dialect_library(MLIRTensorTransforms
  BufferizableOpInterfaceImpl.cpp
  Bufferize.cpp
  SplitPadding.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tensor/Transforms

  DEPENDS
  MLIRTensorTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithmeticDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
  )
