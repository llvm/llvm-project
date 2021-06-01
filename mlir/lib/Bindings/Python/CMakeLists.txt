include(AddMLIRPython)
add_custom_target(MLIRBindingsPythonExtension)

################################################################################
# Build core python extension
################################################################################
add_mlir_python_extension(MLIRCoreBindingsPythonExtension _mlir
  INSTALL_DIR
    python
  SOURCES
    DialectLinalg.cpp
    DialectSparseTensor.cpp
    MainModule.cpp
    IRAffine.cpp
    IRAttributes.cpp
    IRCore.cpp
    IRModule.cpp
    IRTypes.cpp
    PybindUtils.cpp
    Pass.cpp
    ExecutionEngine.cpp
)
add_dependencies(MLIRBindingsPythonExtension MLIRCoreBindingsPythonExtension)

add_subdirectory(Transforms)
add_subdirectory(Conversions)

add_mlir_python_extension(MLIRAllPassesRegistrationBindingsPythonExtension _mlirAllPassesRegistration
  INSTALL_DIR
    python
  SOURCES
    AllPassesRegistration.cpp
)
add_dependencies(MLIRBindingsPythonExtension MLIRAllPassesRegistrationBindingsPythonExtension)

add_mlir_python_extension(MLIRAsyncPassesBindingsPythonExtension _mlirAsyncPasses
  INSTALL_DIR
    python
  SOURCES
    AsyncPasses.cpp
)
add_dependencies(MLIRBindingsPythonExtension MLIRAsyncPassesBindingsPythonExtension)

add_mlir_python_extension(MLIRSparseTensorPassesBindingsPythonExtension _mlirSparseTensorPasses
  INSTALL_DIR
    python
  SOURCES
    SparseTensorPasses.cpp
)
add_dependencies(MLIRBindingsPythonExtension MLIRSparseTensorPassesBindingsPythonExtension)

add_mlir_python_extension(MLIRGPUPassesBindingsPythonExtension _mlirGPUPasses
  INSTALL_DIR
    python
  SOURCES
    GPUPasses.cpp
)
add_dependencies(MLIRBindingsPythonExtension MLIRGPUPassesBindingsPythonExtension)

add_mlir_python_extension(MLIRLinalgPassesBindingsPythonExtension _mlirLinalgPasses
  INSTALL_DIR
    python
  SOURCES
    LinalgPasses.cpp
)
add_dependencies(MLIRBindingsPythonExtension MLIRLinalgPassesBindingsPythonExtension)
