add_mlir_dialect_library(MLIRShapeOpsTransforms
  BufferizableOpInterfaceImpl.cpp
  OutlineShapeComputation.cpp
  RemoveShapeConstraints.cpp
  ShapeToShapeLowering.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ShapeOps/Transforms

  DEPENDS
  MLIRArithOpsIncGen
  MLIRBufferizationEnumsIncGen
  MLIRBufferizationOpsIncGen
  MLIRBufferizationTypeInterfacesIncGen
  MLIRDialectUtilsIncGen
  MLIRFuncOpsIncGen
  MLIRShapeOpsIncGen
  MLIRShapeTransformsIncGen
  MLIRTensorOpsIncGen
  )

target_link_libraries(MLIRShapeOpsTransforms
  PUBLIC
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
