include(AddMLIRPython)

################################################################################
# Generate dialect-specific bindings.
################################################################################

add_mlir_dialect_python_bindings(MLIRBindingsPythonAsyncOps
  TD_FILE AsyncOps.td
  DIALECT_NAME async_dialect)
add_dependencies(MLIRBindingsPythonSources MLIRBindingsPythonAsyncOps)

add_mlir_dialect_python_bindings(MLIRBindingsPythonBuiltinOps
  TD_FILE BuiltinOps.td
  DIALECT_NAME builtin)
add_dependencies(MLIRBindingsPythonSources MLIRBindingsPythonBuiltinOps)

add_mlir_dialect_python_bindings(MLIRBindingsPythonGPUOps
  TD_FILE GPUOps.td
  DIALECT_NAME gpu)
add_dependencies(MLIRBindingsPythonSources MLIRBindingsPythonGPUOps)

add_mlir_dialect_python_bindings(MLIRBindingsPythonLinalgOps
  TD_FILE LinalgOps.td
  DIALECT_NAME linalg
  DEPENDS LinalgOdsGen)
add_dependencies(MLIRBindingsPythonSources MLIRBindingsPythonLinalgOps)

add_mlir_dialect_python_bindings(MLIRBindingsPythonMathOps
  TD_FILE MathOps.td
  DIALECT_NAME math)
add_dependencies(MLIRBindingsPythonSources MLIRBindingsPythonMathOps)

add_mlir_dialect_python_bindings(MLIRBindingsPythonMemRefOps
  TD_FILE MemRefOps.td
  DIALECT_NAME memref)
add_dependencies(MLIRBindingsPythonSources MLIRBindingsPythonMemRefOps)

add_mlir_dialect_python_bindings(MLIRBindingsPythonShapeOps
  TD_FILE ShapeOps.td
  DIALECT_NAME shape)
add_dependencies(MLIRBindingsPythonSources MLIRBindingsPythonShapeOps)

add_mlir_dialect_python_bindings(MLIRBindingsPythonStandardOps
  TD_FILE StandardOps.td
  DIALECT_NAME std)
add_dependencies(MLIRBindingsPythonSources MLIRBindingsPythonStandardOps)

add_mlir_dialect_python_bindings(MLIRBindingsPythonTensorOps
  TD_FILE TensorOps.td
  DIALECT_NAME tensor)
add_dependencies(MLIRBindingsPythonSources MLIRBindingsPythonTensorOps)

add_mlir_dialect_python_bindings(MLIRBindingsPythonTosaOps
  TD_FILE TosaOps.td
  DIALECT_NAME tosa)
add_dependencies(MLIRBindingsPythonSources MLIRBindingsPythonTosaOps)

add_mlir_dialect_python_bindings(MLIRBindingsPythonVectorOps
  TD_FILE VectorOps.td
  DIALECT_NAME vector)
add_dependencies(MLIRBindingsPythonSources MLIRBindingsPythonVectorOps)

################################################################################
# Installation.
################################################################################

# Dialect sources are generated. Install separately.
# Note that __pycache__ directories may have been left by tests and other
# executions. And __init__.py is handled as a regular source file.
# TODO: Eliminate this glob install, instead adding INSTALL_COMPONENT to
# add_mlir_dialect_python_bindings and installing the precise file there.
install(
  DIRECTORY ${PROJECT_BINARY_DIR}/python/mlir/dialects
  DESTINATION python/mlir
  COMPONENT MLIRBindingsPythonDialects
  FILES_MATCHING PATTERN "_*_gen.py"
  PATTERN "__pycache__" EXCLUDE
  PATTERN "__init__.py" EXCLUDE
)

if (NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(
    install-MLIRBindingsPythonDialects
    DEPENDS MLIRBindingsPythonSources
    COMPONENT MLIRBindingsPythonDialects)
endif()
