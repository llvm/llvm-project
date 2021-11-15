include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_standalone`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_standalone.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(StandalonePythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandalonePythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_standalone"
  TD_FILE dialects/StandaloneOps.td
  SOURCES
    dialects/standalone.py
  DIALECT_NAME standalone)

declare_mlir_python_extension(StandalonePythonSources.Extension
  MODULE_NAME _standaloneDialects
  ADD_TO_PARENT StandalonePythonSources
  SOURCES
    StandaloneExtension.cpp
  EMBED_CAPI_LINK_LIBS
    StandaloneCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(StandalonePythonCAPI
  INSTALL_COMPONENT StandalonePythonModules
  INSTALL_DESTINATION python_packages/standalone/mlir_standalone/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/standalone/mlir_standalone/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    StandalonePythonSources
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(StandalonePythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/standalone/mlir_standalone"
  INSTALL_PREFIX "python_packages/standalone/mlir_standalone"
  DECLARED_SOURCES
    StandalonePythonSources
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    StandalonePythonCAPI
  )
