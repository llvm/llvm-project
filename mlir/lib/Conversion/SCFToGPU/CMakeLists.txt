add_mlir_conversion_library(MLIRSCFToGPU
  SCFToGPU.cpp
  SCFToGPUPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/SCFToGPU

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRArithDialect
  MLIRComplexDialect
  MLIRGPUTransforms
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRTransforms
  )

# Conversion patterns and static helpers conflict via 'using namespace mlir'.
set_source_files_properties(
  SCFToGPUPass.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
