add_mlir_dialect_library(MLIRSCFTransforms
  BufferizableOpInterfaceImpl.cpp
  Bufferize.cpp
  ForToWhile.cpp
  LoopCanonicalization.cpp
  LoopPipelining.cpp
  LoopRangeFolding.cpp
  LoopSpecialization.cpp
  ParallelLoopCollapsing.cpp
  ParallelLoopFusion.cpp
  ParallelLoopTiling.cpp
  StructuralTypeConversions.cpp
  TileUsingInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SCF

  DEPENDS
  MLIRSCFPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRArithmeticDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRDialectUtils
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRSupport
  MLIRTensorDialect
  MLIRTilingInterface
  MLIRTransforms
  MLIRTransformUtils
)
