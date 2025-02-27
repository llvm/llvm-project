# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestFuncToLLVM
  TestConvertCallOp.cpp
  TestConvertFuncOp.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRTestDialect
  )
mlir_target_link_libraries(MLIRTestFuncToLLVM PUBLIC
  MLIRFuncToLLVM
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLLVMIRTransforms
  MLIRPass
  )

target_include_directories(MLIRTestFuncToLLVM
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../../Dialect/Test
  )
