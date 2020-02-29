add_mlir_conversion_library(MLIRLinalgToSPIRVTransforms
  LinalgToSPIRV.cpp
  LinalgToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  LINK_LIBS
  MLIRIR
  MLIRLinalgOps
  MLIRLinalgUtils
  MLIRPass
  MLIRSPIRV
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
  LLVMSupport
  )
