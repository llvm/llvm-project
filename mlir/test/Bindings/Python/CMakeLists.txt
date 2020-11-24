include(AddMLIRPythonExtension)
add_mlir_dialect_python_bindings(MLIRBindingsPythonTestOps
  python_test_ops.td
  python_test)
