################################################################################
# Build python extension
################################################################################

add_mlir_python_extension(MLIRConversionsBindingsPythonExtension _mlirConversions
  INSTALL_DIR
    python
  SOURCES
  Conversions.cpp
)
