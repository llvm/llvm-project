add_mlir_unittest(MLIRExecutionEngineTests
  DynamicMemRef.cpp
  StridedMemRef.cpp
  Invoke.cpp
)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_link_libraries(MLIRExecutionEngineTests
  PRIVATE
  MLIRArithToLLVM
  MLIRExecutionEngine
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  ${dialect_libs}

)
