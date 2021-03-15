add_mlir_dialect_library(MLIRLinalgTransforms
  Bufferize.cpp
  CodegenStrategy.cpp
  Detensorize.cpp
  DropUnitDims.cpp
  ElementwiseToLinalg.cpp
  Fusion.cpp
  FusionOnTensors.cpp
  Generalization.cpp
  Hoisting.cpp
  Interchange.cpp
  Loops.cpp
  Promotion.cpp
  SparseLowering.cpp
  Sparsification.cpp
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
  MLIREDSC
  MLIRIR
  MLIRMemRef
  MLIRLinalgAnalysis
  MLIRLinalgEDSC
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
