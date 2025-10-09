add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir.")

declare_mlir_python_sources(StandReallyAlonePythonSources)
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StandReallyAlonePythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_standreallyalone"
  SOURCES
    dialects/standalonereallyalone.py
    dialects/_ods_common.py
)

declare_mlir_python_extension(StandReallyAlonePythonSources.NanobindExtension
  MODULE_NAME _standReallyAloneDialectsNanobind
  ADD_TO_PARENT StandReallyAlonePythonSources
  SOURCES
    StandReallyAloneExtensionNanobind.cpp
  PRIVATE_LINK_LIBS
    StandaloneCAPI
  PYTHON_BINDINGS_LIBRARY nanobind
)

set(StandReallyAlonePythonModules_ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")
add_mlir_python_modules(StandReallyAlonePythonModules
  ROOT_PREFIX "${StandReallyAlonePythonModules_ROOT_PREFIX}/../mlir_standreallyalone"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/../mlir_standreallyalone"
  DECLARED_SOURCES
    StandReallyAlonePythonSources
    StandReallyAlonePythonSources.NanobindExtension
    StandalonePythonSources.standalone.ops_gen
    StandalonePythonSources.standalone.tablegen
    MLIRPythonSources.Dialects.builtin
)
