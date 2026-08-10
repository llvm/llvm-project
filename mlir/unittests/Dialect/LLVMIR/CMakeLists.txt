add_mlir_unittest(MLIRLLVMIRTests
  LLVMTypeTest.cpp
)
mlir_target_link_libraries(MLIRLLVMIRTests
  PRIVATE
  MLIRLLVMDialect
  )
