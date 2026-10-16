# Keep the complete public library as an umbrella over its components.
add_mlir_library(MLIRComplexOps
  OBJECT
  ComplexOps.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Complex

  DEPENDS
  MLIRComplexOpsIncGen
  MLIRComplexAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRInferTypeOpInterface
  MLIRIRCore

  HEADER_LIBS
  MLIRDialect
  )

add_mlir_dialect_library(MLIRComplexDialect
  ComplexDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Complex

  DEPENDS
  MLIRComplexOpsIncGen
  MLIRComplexAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRComplexOps
  MLIRArithDialect
  MLIRDialect
  MLIRInferTypeOpInterface
  MLIRIR
  )

# Preserve the complete public archive. The component objects are compiled
# once; libMLIR still collects each object's defining library exactly once.
if(NOT BUILD_SHARED_LIBS)
  target_sources(MLIRComplexDialect PRIVATE $<TARGET_OBJECTS:obj.MLIRComplexOps>)
endif()
