add_mlir_dialect_library(MLIRShapeOpsTransforms
  BufferizableOpInterfaceImpl.cpp
  OutlineShapeComputation.cpp
  RemoveShapeConstraints.cpp
  ShapeToShapeLowering.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ShapeOps/Transforms

  DEPENDS
  MLIRShapeTransformsIncGen

  # The implementations include FuncOps.h and Tensor.h; Tensor.h includes
  # Dialect/Utils/ReshapeOpsUtils.h.
  HEADER_LIBS
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRTensorDialect

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRShapeDialect
  MLIRSupport
  MLIRTransforms
  )
