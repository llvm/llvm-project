add_mlir_unittest(MLIRExecutionEngineTests
  Invoke.cpp
)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_link_libraries(MLIRExecutionEngineTests
  PRIVATE
  MLIRExecutionEngine
  MLIRLinalgToLLVM
  ${dialect_libs}

)
