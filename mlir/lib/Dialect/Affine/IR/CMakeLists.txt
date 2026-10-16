# Keep the complete public library as an umbrella over its components.
add_mlir_library(MLIRAffineCore
  OBJECT
  AffineMemoryOpInterfaces.cpp
  AffineOps.cpp
  AffineValueMap.cpp
  InferIntDivisibilityOpInterfaceImpl.cpp
  InferIntRangeInterfaceImpls.cpp
  MemorySlot.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIRAffineMemoryOpInterfacesIncGen
  MLIRAffineOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectCoreUtils
  MLIRIRCore
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRLoopLikeInterface
  MLIRMemRefCore
  MLIRMemorySlotUtils
  MLIRShapedOpInterfaces
  MLIRSideEffectInterfaces
  MLIRUBDialect
  MLIRValueBoundsOpInterface

  HEADER_LIBS
  MLIRInferIntDivisibilityOpInterface
  )

add_mlir_dialect_library(MLIRAffineDialect
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIRAffineMemoryOpInterfacesIncGen
  MLIRAffineOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineCore
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRInferIntDivisibilityOpInterface
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRLoopLikeInterface
  MLIRMemRefDialect
  MLIRMemorySlotUtils
  MLIRShapedOpInterfaces
  MLIRSideEffectInterfaces
  MLIRUBDialect
  MLIRValueBoundsOpInterface
  )

# Preserve the complete public archive. The component objects are compiled
# once; libMLIR still collects each object's defining library exactly once.
if(NOT BUILD_SHARED_LIBS)
  target_sources(MLIRAffineDialect PRIVATE $<TARGET_OBJECTS:obj.MLIRAffineCore>)
endif()
