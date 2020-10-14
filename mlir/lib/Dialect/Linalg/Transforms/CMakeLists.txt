add_mlir_dialect_library(MLIRLinalgTransforms
  CodegenStrategy.cpp
  DropUnitDims.cpp
  Fusion.cpp
  FusionOnTensors.cpp
  Hoisting.cpp
  Interchange.cpp
  Loops.cpp
  Promotion.cpp
  TensorsToBuffers.cpp
  Tiling.cpp
  Transforms.cpp
  Vectorization.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg

  DEPENDS
  MLIRLinalgPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAnalysis
  MLIREDSC
  MLIRIR
  MLIRLinalgAnalysis
  MLIRLinalgEDSC
  MLIRLinalg
  MLIRLinalgUtils
  MLIRSCF
  MLIRSCFTransforms
  MLIRPass
  MLIRStandard
  MLIRStandardToLLVM
  MLIRTransforms
  MLIRTransformUtils
  MLIRVector
  )
