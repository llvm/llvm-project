# Keep the complete public library as an umbrella over its components.
add_mlir_library(MLIRLLVMIRExportLegalization
  OBJECT
  DIExpressionLegalization.cpp
  DIExpressionRewriter.cpp
  LegalizeForExport.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRLLVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRIRCore
  MLIRLLVMDialect
  )

add_mlir_dialect_library(MLIRLLVMIRTransforms
  AddComdats.cpp
  DIScopeForLLVMFuncOp.cpp
  InlinerInterfaceImpl.cpp
  RequestCWrappers.cpp
  UseDefaultVisibilityPass.cpp

  DEPENDS
  MLIRLLVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMIRExportLegalization
  MLIRIR
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
  MLIROpenMPDialect
  )

# Preserve the complete public archive. The component objects are compiled
# once; libMLIR still collects each object's defining library exactly once.
if(NOT BUILD_SHARED_LIBS)
  target_sources(MLIRLLVMIRTransforms PRIVATE $<TARGET_OBJECTS:obj.MLIRLLVMIRExportLegalization>)
endif()
