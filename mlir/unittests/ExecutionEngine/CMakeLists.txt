set(LLVM_LINK_COMPONENTS
  nativecodegen
  native
  orcjit
  support
)

add_mlir_unittest(MLIRExecutionEngineTests
  DynamicMemRef.cpp
  StridedMemRef.cpp
  Invoke.cpp
)

mlir_target_link_libraries(MLIRExecutionEngineTests
  PRIVATE
  MLIRArithToLLVM
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRRegisterAllDialects
)
target_link_libraries(MLIRExecutionEngineTests
  PRIVATE
  MLIRExecutionEngine
)
