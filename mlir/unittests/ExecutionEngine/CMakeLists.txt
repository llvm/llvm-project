add_mlir_unittest(MLIRExecutionEngineTests
  Invoke.cpp
)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_link_libraries(MLIRExecutionEngineTests
  PRIVATE
  MLIRArithmeticToLLVM
  MLIRExecutionEngine
  MLIRLinalgToLLVM
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  ${dialect_libs}

)
