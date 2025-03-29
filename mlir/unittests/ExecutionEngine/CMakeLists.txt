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
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

mlir_target_link_libraries(MLIRExecutionEngineTests
  PRIVATE
  MLIRArithToLLVM
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  ${dialect_libs}
)
target_link_libraries(MLIRExecutionEngineTests
  PRIVATE
  MLIRExecutionEngine
)
