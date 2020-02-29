add_llvm_library(MLIRSPIRVTestPasses
  TestAvailability.cpp

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
  LLVMSupport
  )
