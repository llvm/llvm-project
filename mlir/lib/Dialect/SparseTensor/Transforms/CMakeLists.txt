add_mlir_dialect_library(MLIRSparseTensorTransforms
  CodegenUtils.cpp
  Sparsification.cpp
  SparseTensorConversion.cpp
  SparseTensorPasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SparseTensor

  DEPENDS
  MLIRSparseTensorPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRBufferization
  MLIRComplex
  MLIRFunc
  MLIRIR
  MLIRLLVMIR
  MLIRLinalg
  MLIRLinalgTransforms
  MLIRMemRef
  MLIRPass
  MLIRSCF
  MLIRSCFTransforms
  MLIRSCFUtils
  MLIRSparseTensor
  MLIRSparseTensorUtils
  MLIRTensor
  MLIRTransforms
  MLIRVector
)
