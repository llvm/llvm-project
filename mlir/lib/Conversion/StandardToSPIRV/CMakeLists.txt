set(LLVM_TARGET_DEFINITIONS StandardToSPIRV.td)
mlir_tablegen(StandardToSPIRV.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRStandardToSPIRVIncGen)

add_mlir_conversion_library(MLIRStandardToSPIRVTransforms
  ConvertStandardToSPIRV.cpp
  ConvertStandardToSPIRVPass.cpp
  LegalizeStandardForSPIRV.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  LINK_LIBS
  MLIRIR
  MLIRPass
  MLIRSPIRV
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
  MLIRSPIRV
  MLIRStandardOps
  LLVMSupport
  )

add_dependencies(MLIRStandardToSPIRVTransforms
  MLIRStandardToSPIRVIncGen)
