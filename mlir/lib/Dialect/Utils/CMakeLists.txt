# Keep the complete public library as an umbrella over its components.
add_mlir_library(MLIRDialectCoreUtils
  OBJECT
  ReshapeOpsUtils.cpp
  StaticValueUtils.cpp
  VerificationUtils.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRDialectUtilsIncGen
  MLIRArithOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIRCore
)

add_mlir_library(MLIRDialectUtils
  IndexingUtils.cpp
  StructuredOpsUtils.cpp

  DEPENDS
  MLIRDialectUtilsIncGen
  MLIRArithOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialectCoreUtils
  MLIRIR
)

# Preserve the complete public archive. The component objects are compiled
# once; libMLIR still collects each object's defining library exactly once.
if(NOT BUILD_SHARED_LIBS)
  target_sources(MLIRDialectUtils PRIVATE $<TARGET_OBJECTS:obj.MLIRDialectCoreUtils>)
endif()
