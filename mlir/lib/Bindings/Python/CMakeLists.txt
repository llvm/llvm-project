include(AddMLIRPythonExtension)
add_custom_target(MLIRBindingsPythonExtension)

################################################################################
# Copy python source tree.
################################################################################

file(GLOB_RECURSE PY_SRC_FILES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/mlir/*.py")

add_custom_target(MLIRBindingsPythonSources ALL
  DEPENDS ${PY_SRC_FILES}
)
add_dependencies(MLIRBindingsPythonExtension MLIRBindingsPythonSources)

foreach(PY_SRC_FILE ${PY_SRC_FILES})
  set(PY_DEST_FILE "${PROJECT_BINARY_DIR}/python/${PY_SRC_FILE}")
  get_filename_component(PY_DEST_DIR "${PY_DEST_FILE}" DIRECTORY)
  file(MAKE_DIRECTORY "${PY_DEST_DIR}")
  add_custom_command(
    TARGET MLIRBindingsPythonSources PRE_BUILD
    COMMENT "Copying python source ${PY_SRC_FILE} -> ${PY_DEST_FILE}"
    DEPENDS "${PY_SRC_FILE}"
    COMMAND "${CMAKE_COMMAND}" -E create_symlink
        "${CMAKE_CURRENT_SOURCE_DIR}/${PY_SRC_FILE}" "${PY_DEST_FILE}"
  )
endforeach()

################################################################################
# Generate dialect-specific bindings.
################################################################################

add_mlir_dialect_python_bindings(MLIRBindingsPythonBuiltinOps
  TD_FILE BuiltinOps.td
  DIALECT_NAME builtin)
add_dependencies(MLIRBindingsPythonSources MLIRBindingsPythonBuiltinOps)

add_mlir_dialect_python_bindings(MLIRBindingsPythonLinalgOps
  TD_FILE LinalgOps.td
  DIALECT_NAME linalg
  DEPENDS LinalgOdsGen)
add_dependencies(MLIRBindingsPythonSources MLIRBindingsPythonLinalgOps)

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

################################################################################
# Build core python extension
################################################################################
add_mlir_python_extension(MLIRCoreBindingsPythonExtension _mlir
  INSTALL_DIR
    python
  SOURCES
    MainModule.cpp
    IRAffine.cpp
    IRAttributes.cpp
    IRCore.cpp
    IRTypes.cpp
    PybindUtils.cpp
    Pass.cpp
    ExecutionEngine.cpp
)
add_dependencies(MLIRBindingsPythonExtension MLIRCoreBindingsPythonExtension)

# Note that we copy from the source tree just like for headers because
# it will not be polluted with py_cache runtime artifacts (from testing and
# such).
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mlir
  DESTINATION python
  COMPONENT MLIRBindingsPythonSources
  FILES_MATCHING PATTERN "*.py"
)

if (NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(
    install-MLIRBindingsPythonSources
    DEPENDS MLIRBindingsPythonSources
    COMPONENT MLIRBindingsPythonSources)
endif()

# Dialect sources are generated. Install separately.
# Note that __pycache__ directories may have been left by tests and other
# executions. And __init__.py is handled as a regular source file.
install(
  DIRECTORY ${PROJECT_BINARY_DIR}/python/mlir/dialects
  DESTINATION python/mlir
  COMPONENT MLIRBindingsPythonDialects
  FILES_MATCHING PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
  PATTERN "__init__.py" EXCLUDE
)

if (NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(
    install-MLIRBindingsPythonDialects
    DEPENDS MLIRBindingsPythonSources
    COMPONENT MLIRBindingsPythonDialects)
endif()

add_subdirectory(Transforms)
add_subdirectory(Conversions)

add_mlir_python_extension(MLIRLinalgPassesBindingsPythonExtension _mlirLinalgPasses
  INSTALL_DIR
    python
  SOURCES
    LinalgPasses.cpp
)
add_dependencies(MLIRBindingsPythonExtension MLIRLinalgPassesBindingsPythonExtension)
