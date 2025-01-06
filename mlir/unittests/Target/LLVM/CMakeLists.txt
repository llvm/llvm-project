set(LLVM_LINK_COMPONENTS nativecodegen)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_unittest(MLIRTargetLLVMTests
  SerializeNVVMTarget.cpp
  SerializeROCDLTarget.cpp
  SerializeToLLVMBitcode.cpp
DEPENDS
  ${dialect_libs}
)

mlir_target_link_libraries(MLIRTargetLLVMTests
  PRIVATE
  MLIRTargetLLVM
  MLIRNVVMTarget
  MLIRROCDLTarget
  MLIRGPUDialect
  MLIRNVVMDialect
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRBuiltinToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
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
