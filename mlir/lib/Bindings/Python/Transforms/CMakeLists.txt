################################################################################
# Build python extension
################################################################################

add_mlir_python_extension(MLIRTransformsBindingsPythonExtension _mlirTransforms
  INSTALL_DIR
    python
  SOURCES
  Transforms.cpp
)