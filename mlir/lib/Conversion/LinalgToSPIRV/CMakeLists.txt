add_mlir_conversion_library(MLIRLinalgToSPIRVTransforms
  LinalgToSPIRV.cpp
  LinalgToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen
  )

target_link_libraries(MLIRLinalgToSPIRVTransforms
  PUBLIC
  MLIRIR
  MLIRLinalgOps
  MLIRLinalgUtils
  MLIRPass
  MLIRSPIRV
  MLIRSupport
  )
