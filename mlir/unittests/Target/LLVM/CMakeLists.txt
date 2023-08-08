add_mlir_unittest(MLIRTargetLLVMTests
  SerializeToLLVMBitcode.cpp
)

llvm_map_components_to_libnames(llvm_libs nativecodegen)

target_link_libraries(MLIRTargetLLVMTests
  PRIVATE
  MLIRTargetLLVM
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRBuiltinToLLVMIRTranslation
  ${llvm_libs}
)

if (DEFINED LLVM_NATIVE_TARGET)
  target_compile_definitions(MLIRTargetLLVMTests
    PRIVATE
    -DLLVM_NATIVE_TARGET_TEST_ENABLED=1
  )
else()
  target_compile_definitions(MLIRTargetLLVMTests
    PRIVATE
    -DLLVM_NATIVE_TARGET_TEST_ENABLED=0
  )
endif()
