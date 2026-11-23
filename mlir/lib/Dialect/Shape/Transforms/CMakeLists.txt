set(LLVM_OPTIONAL_SOURCES
  BufferizableOpInterfaceImpl.cpp
  )

add_mlir_dialect_library(MLIRShapeOpsTransforms
  OutlineShapeComputation.cpp
  RemoveShapeConstraints.cpp
  ShapeToShapeLowering.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ShapeOps/Transforms

  DEPENDS
  MLIRShapeTransformsIncGen
  )

target_link_libraries(MLIRShapeOpsTransforms
  PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRShapeDialect
  MLIRSupport
  MLIRTransforms
  )

add_mlir_dialect_library(MLIRShapeBufferizableOpInterfaceImpl
  BufferizableOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ShapeOps/Transforms

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRIR
  MLIRShapeDialect
  )
