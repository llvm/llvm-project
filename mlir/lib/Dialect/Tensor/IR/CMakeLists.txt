set(LLVM_OPTIONAL_SOURCES
  TensorDialect.cpp
  TensorInferTypeOpInterfaceImpl.cpp
  TensorOps.cpp
  TensorTilingInterfaceImpl.cpp
)

add_mlir_dialect_library(MLIRTensor
  TensorDialect.cpp
  TensorOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mlir/Dialect/Tensor

  DEPENDS
  MLIRTensorOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRArithmeticUtils
  MLIRCastInterfaces
  MLIRComplex
  MLIRDialectUtils
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRViewLikeInterface
  )

add_mlir_dialect_library(MLIRTensorInferTypeOpInterfaceImpl
  TensorInferTypeOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tensor

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSupport
  MLIRTensor
  )

add_mlir_dialect_library(MLIRTensorTilingInterfaceImpl
  TensorTilingInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tensor

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRIR
  MLIRLinalg
  MLIRSCF
  MLIRSupport
  MLIRTensor
  MLIRTilingInterface
  )
