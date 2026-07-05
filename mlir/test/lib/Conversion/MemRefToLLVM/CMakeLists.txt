# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestMemRefToLLVMWithTransforms
  TestMemRefToLLVMWithTransforms.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRTestDialect
  )
mlir_target_link_libraries(MLIRTestMemRefToLLVMWithTransforms PUBLIC
  MLIRFuncToLLVM
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefTransforms
  MLIRPass
  )

target_include_directories(MLIRTestMemRefToLLVMWithTransforms
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../../Dialect/Test
  )
