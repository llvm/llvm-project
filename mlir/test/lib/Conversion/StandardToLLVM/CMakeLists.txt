# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestStandardToLLVM
  TestConvertCallOp.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRLLVMCommonConversion
  MLIRLLVMIR
  MLIRLLVMIRTransforms
  MLIRPass
  MLIRStandardToLLVM
  MLIRTestDialect
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../Dialect/Test)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../../Dialect/Test)
