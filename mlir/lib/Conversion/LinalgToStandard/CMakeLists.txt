add_mlir_conversion_library(MLIRLinalgToStandard
  LinalgToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LinalgToStandard

  DEPENDS
  MLIRConversionPassIncGen
)

target_link_libraries(MLIRLinalgToStandard
  PUBLIC
  MLIREDSC
  MLIRIR
  MLIRLinalgOps
  MLIRSCF
  LLVMCore
  LLVMSupport
  )
