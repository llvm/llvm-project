# Keep the complete public library as an umbrella over its components.
add_mlir_library(MLIRMathOps
  OBJECT
  MathOps.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Math

  DEPENDS
  MLIRMathOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIRCore
  MLIRUBDialect

  HEADER_LIBS
  MLIRDialect
  )

add_mlir_dialect_library(MLIRMathDialect
  MathDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Math

  DEPENDS
  MLIRMathOpsIncGen

  LINK_LIBS PUBLIC
  MLIRMathOps
  MLIRArithDialect
  MLIRDialect
  MLIRIR
  MLIRUBDialect
  )

# Preserve the complete public archive. The component objects are compiled
# once; libMLIR still collects each object's defining library exactly once.
if(NOT BUILD_SHARED_LIBS)
  target_sources(MLIRMathDialect PRIVATE $<TARGET_OBJECTS:obj.MLIRMathOps>)
endif()
