add_mlir_dialect_library(MLIRSCFTransforms
  Bufferize.cpp
  ForToWhile.cpp
  LoopCanonicalization.cpp
  LoopPipelining.cpp
  LoopRangeFolding.cpp
  LoopSpecialization.cpp
  ParallelLoopFusion.cpp
  ParallelLoopTiling.cpp
  StructuralTypeConversions.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SCF

  DEPENDS
  MLIRSCFPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRArithmetic
  MLIRIR
  MLIRMemRef
  MLIRPass
  MLIRSCF
  MLIRStandard
  MLIRSupport
  MLIRTensor
  MLIRTransforms
  MLIRTransformUtils
)
