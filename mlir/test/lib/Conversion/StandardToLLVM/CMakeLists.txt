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

target_include_directories(MLIRTestStandardToLLVM
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../../Dialect/Test
  )
