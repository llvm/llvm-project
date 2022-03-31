# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestFuncToLLVM
  TestConvertCallOp.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRFuncToLLVM
  MLIRLLVMCommonConversion
  MLIRLLVMIR
  MLIRLLVMIRTransforms
  MLIRPass
  MLIRTestDialect
  )

target_include_directories(MLIRTestFuncToLLVM
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../../Dialect/Test
  )
