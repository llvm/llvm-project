include(AddMLIRPythonExtension)
add_custom_target(MLIRBindingsPythonExtension)

################################################################################
# Generate dialect-specific bindings.
################################################################################

add_mlir_dialect_python_bindings(MLIRBindingsPythonStandardOps
  StandardOps.td
  std)
add_dependencies(MLIRBindingsPythonExtension MLIRBindingsPythonStandardOps)

################################################################################
# Copy python source tree.
################################################################################

set(PY_SRC_FILES
  mlir/__init__.py
  mlir/_dlloader.py
  mlir/ir.py
  mlir/dialects/__init__.py
  mlir/ir.py
  mlir/passmanager.py
  mlir/transforms/__init__.py
)

add_custom_target(MLIRBindingsPythonSources ALL
  DEPENDS ${PY_SRC_FILES}
)
add_dependencies(MLIRBindingsPythonExtension MLIRBindingsPythonSources)

foreach(PY_SRC_FILE ${PY_SRC_FILES})
  set(PY_DEST_FILE "${PROJECT_BINARY_DIR}/python/${PY_SRC_FILE}")
  add_custom_command(
    TARGET MLIRBindingsPythonSources PRE_BUILD
    COMMENT "Copying python source ${PY_SRC_FILE} -> ${PY_DEST_FILE}"
    DEPENDS "${PY_SRC_FILE}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/${PY_SRC_FILE}" "${PY_DEST_FILE}"
  )
endforeach()

################################################################################
# Build core python extension
################################################################################
add_mlir_python_extension(MLIRCoreBindingsPythonExtension _mlir
  INSTALL_DIR
    python
  SOURCES
    MainModule.cpp
    IRModules.cpp
    PybindUtils.cpp
    Pass.cpp
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

add_subdirectory(Transforms)
