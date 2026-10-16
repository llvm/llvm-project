# Keep the complete public library as an umbrella over its components.
add_mlir_library(MLIRMemRefCore
  OBJECT
  MemRefDialect.cpp
  MemoryAccessOpInterfaces.cpp
  MemRefMemorySlot.cpp
  MemRefOps.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/inlude/mlir/Dialect/MemRef/IR

  DEPENDS
  MLIRMemRefOpsIncGen
  MLIRMemoryAccessOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRArithUtils
  MLIRControlFlowInterfaces
  MLIRDialectCoreUtils
  MLIRInferIntRangeCommon
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRIRCore
  MLIRMemOpInterfaces
  MLIRMemorySlotInterfaces
  MLIRShapedOpInterfaces
  MLIRSideEffectInterfaces
  MLIRUBDialect
  MLIRValueBoundsOpInterface
  MLIRViewLikeInterface

  HEADER_LIBS
  MLIRDialect
  MLIRInferStridedMetadataInterface
)

add_mlir_dialect_library(MLIRMemRefDialect
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/inlude/mlir/Dialect/MemRef/IR

  DEPENDS
  MLIRMemRefOpsIncGen
  MLIRMemoryAccessOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRMemRefCore
  MLIRArithDialect
  MLIRArithUtils
  MLIRControlFlowInterfaces
  MLIRDialect
  MLIRDialectUtils
  MLIRInferIntRangeCommon
  MLIRInferIntRangeInterface
  MLIRInferStridedMetadataInterface
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRMemOpInterfaces
  MLIRMemorySlotInterfaces
  MLIRShapedOpInterfaces
  MLIRSideEffectInterfaces
  MLIRUBDialect
  MLIRValueBoundsOpInterface
  MLIRViewLikeInterface
)

# Preserve the complete public archive. The component objects are compiled
# once; libMLIR still collects each object's defining library exactly once.
if(NOT BUILD_SHARED_LIBS)
  target_sources(MLIRMemRefDialect PRIVATE $<TARGET_OBJECTS:obj.MLIRMemRefCore>)
endif()
