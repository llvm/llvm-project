include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_standalone`
# top level package (the API has been embedded in a relocatable way).
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(StandalonePythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandalonePythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_standalone"
  TD_FILE dialects/StandaloneOps.td
  SOURCES
    dialects/standalone_pybind11.py
    dialects/standalone_nanobind.py
  DIALECT_NAME standalone)


declare_mlir_python_extension(StandalonePythonSources.Pybind11Extension
  MODULE_NAME _standaloneDialectsPybind11
  ADD_TO_PARENT StandalonePythonSources
  SOURCES
    StandaloneExtensionPybind11.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    StandaloneCAPI
  PYTHON_BINDINGS_LIBRARY pybind11
)

declare_mlir_python_extension(StandalonePythonSources.NanobindExtension
  MODULE_NAME _standaloneDialectsNanobind
  ADD_TO_PARENT StandalonePythonSources
  SOURCES
    StandaloneExtensionNanobind.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    StandaloneCAPI
  PYTHON_BINDINGS_LIBRARY nanobind
)


################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(StandalonePythonCAPI
  INSTALL_COMPONENT StandalonePythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    StandalonePythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
)

################################################################################
# Instantiation of all Python modules
################################################################################

set(StandalonePythonModules_ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")
add_mlir_python_modules(StandalonePythonModules
  ROOT_PREFIX "${StandalonePythonModules_ROOT_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES
    StandalonePythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
  COMMON_CAPI_LINK_LIBS
    StandalonePythonCAPI
)

# Everything here is very tightly coupled. See the ample descriptions at the bottom of
# mlir/python/CMakeLists.txt.
get_target_property(_standalone_extension_srcs StandalonePythonSources.NanobindExtension INTERFACE_SOURCES)
generate_type_stubs(
  MODULE_NAME mlir_standalone._mlir_libs._standaloneDialectsNanobind
  DEPENDS_TARGETS
    StandalonePythonModules.extension._mlir.dso
    StandalonePythonModules.extension._standaloneDialectsNanobind.dso
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_mlir_libs"
  OUTPUTS
    _standaloneDialectsNanobind/__init__.pyi
    _standaloneDialectsNanobind/standalone.pyi
  DEPENDS_TARGET_SRC_DEPS "${_standalone_extension_srcs}"
  IMPORT_PATHS "${StandalonePythonModules_ROOT_PREFIX}/.."
)
add_dependencies(StandalonePythonModules "${NB_STUBGEN_CUSTOM_TARGET}")

declare_mlir_python_sources(
  StandalonePythonSources.type_stub_gen
  ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
  ADD_TO_PARENT StandalonePythonSources
  SOURCES
    _mlir_libs/_standaloneDialectsNanobind/__init__.pyi
    _mlir_libs/_standaloneDialectsNanobind/standalone.pyi
)
