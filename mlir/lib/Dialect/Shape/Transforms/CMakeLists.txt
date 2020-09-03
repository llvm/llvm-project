add_mlir_dialect_library(MLIRShapeOpsTransforms
  RemoveShapeConstraints.cpp
  ShapeToShapeLowering.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ShapeOps/Transforms

  DEPENDS
  MLIRShapeTransformsIncGen
  )

target_link_libraries(MLIRShapeOpsTransforms
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRShape
  MLIRSupport
  MLIRTransforms
  )
