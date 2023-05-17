add_mlir_unittest(MLIRLLVMIRTests
  LLVMTypeTest.cpp
)
target_link_libraries(MLIRLLVMIRTests
  PRIVATE
  MLIRLLVMDialect
  )
