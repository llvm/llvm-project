add_mlir_dialect_library(MLIRShapeOpsTransforms
  BufferizableOpInterfaceImpl.cpp
  OutlineShapeComputation.cpp
  RemoveShapeConstraints.cpp
  ShapeToShapeLowering.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ShapeOps/Transforms

  DEPENDS
  MLIRShapeTransformsIncGen

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
