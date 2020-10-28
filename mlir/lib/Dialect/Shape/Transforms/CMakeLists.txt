add_mlir_dialect_library(MLIRShapeOpsTransforms
  Bufferize.cpp
  RemoveShapeConstraints.cpp
  ShapeToShapeLowering.cpp
  StructuralTypeConversions.cpp

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
