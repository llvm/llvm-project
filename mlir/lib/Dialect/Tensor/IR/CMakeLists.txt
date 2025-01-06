set(LLVM_OPTIONAL_SOURCES
  TensorDialect.cpp
  TensorInferTypeOpInterfaceImpl.cpp
  TensorOps.cpp
  TensorTilingInterfaceImpl.cpp
  ValueBoundsOpInterfaceImpl.cpp
)

add_mlir_dialect_library(MLIRTensorDialect
  TensorDialect.cpp
  TensorOps.cpp
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mlir/Dialect/Tensor

  DEPENDS
  MLIRTensorOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRArithUtils
  MLIRCastInterfaces
  MLIRComplexDialect
  MLIRDestinationStyleOpInterface
  MLIRDialectUtils
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRParallelCombiningOpInterface
  MLIRShapedOpInterfaces
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRValueBoundsOpInterface
  MLIRViewLikeInterface
  )

add_mlir_dialect_library(MLIRTensorInferTypeOpInterfaceImpl
  TensorInferTypeOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tensor

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSupport
  MLIRTensorDialect
  )

add_mlir_dialect_library(MLIRTensorTilingInterfaceImpl
  TensorTilingInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tensor

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRArithUtils
  MLIRDialectUtils
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTensorUtils
  MLIRTilingInterface
  MLIRValueBoundsOpInterface
  )
