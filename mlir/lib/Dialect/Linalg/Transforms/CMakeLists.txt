set(LLVM_OPTIONAL_SOURCES
  Bufferize.cpp
  BufferizableOpInterface.cpp
  CodegenStrategy.cpp
  ComprehensiveBufferize.cpp
  Detensorize.cpp
  Distribution.cpp
  DropUnitDims.cpp
  ElementwiseOpFusion.cpp
  ElementwiseToLinalg.cpp
  Fusion.cpp
  FusionOnTensors.cpp
  Generalization.cpp
  Hoisting.cpp
  HoistPadding.cpp
  InlineScalarOperands.cpp
  Interchange.cpp
  Loops.cpp
  LinalgStrategyPasses.cpp
  Promotion.cpp
  Tiling.cpp
  Transforms.cpp
  Vectorization.cpp
)

add_mlir_dialect_library(MLIRBufferizableOpInterface
  BufferizableOpInterface.cpp

  DEPENDS
  MLIRBufferizableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

add_mlir_dialect_library(MLIRLinalgTransforms
  Bufferize.cpp
  CodegenStrategy.cpp
  ComprehensiveBufferize.cpp
  Detensorize.cpp
  Distribution.cpp
  DropUnitDims.cpp
  ElementwiseOpFusion.cpp
  ElementwiseToLinalg.cpp
  Fusion.cpp
  FusionOnTensors.cpp
  Generalization.cpp
  Hoisting.cpp
  HoistPadding.cpp
  InlineScalarOperands.cpp
  Interchange.cpp
  Loops.cpp
  LinalgStrategyPasses.cpp
  Promotion.cpp
  Tiling.cpp
  Transforms.cpp
  Vectorization.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg

  DEPENDS
  MLIRLinalgPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAffineUtils
  MLIRAnalysis
  MLIRArithmetic
  MLIRBufferizableOpInterface
  MLIRComplex
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRMemRef
  MLIRLinalgAnalysis
  MLIRLinalg
  MLIRLinalgUtils
  MLIRSCF
  MLIRSCFTransforms
  MLIRPass
  MLIRStandard
  MLIRStandardOpsTransforms
  MLIRStandardToLLVM
  MLIRTensor
  MLIRTransforms
  MLIRTransformUtils
  MLIRVector
  MLIRVectorToSCF
)
