# Keep the complete public library as an umbrella over its components.
add_mlir_library(MLIROpenMPCodeGenTransforms
  OBJECT
  HostOpFiltering.cpp
  MarkDeclareTarget.cpp
  StackToShared.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenMP

  DEPENDS
  omp_gen
  MLIROpenMPPassIncGen

  LINK_LIBS PUBLIC
  MLIRFunctionInterfaces
  MLIRIRCore
  MLIRLLVMDialect
  MLIROpenACCMPCommon
  MLIROpenMPDialect
  MLIROpenMPUtils
  MLIRPass
  MLIRSupportCore

  )

add_mlir_dialect_library(MLIROpenMPTransforms
  FunctionFiltering.cpp
  OpenMPOffloadPrivatizationPrepare.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenMP

  DEPENDS
  omp_gen
  MLIROpenMPPassIncGen

  LINK_LIBS PUBLIC
  MLIROpenMPCodeGenTransforms
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLLVMDialect
  MLIROpenACCMPCommon
  MLIROpenMPDialect
  MLIROpenMPUtils
  MLIRPass
  MLIRSupport
  MLIRTransforms
  )

# Preserve the complete public archive. The component objects are compiled
# once; libMLIR still collects each object's defining library exactly once.
if(NOT BUILD_SHARED_LIBS)
  target_sources(MLIROpenMPTransforms PRIVATE $<TARGET_OBJECTS:obj.MLIROpenMPCodeGenTransforms>)
endif()
