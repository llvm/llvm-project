include(AddMLIRPythonExtension)
add_mlir_dialect_python_bindings(MLIRBindingsPythonTestOps
  TD_FILE python_test_ops.td
  DIALECT_NAME python_test)
