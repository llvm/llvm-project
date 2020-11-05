if(NOT LLVM_BUILD_LLVM_DYLIB)
  message(FATAL_ERROR "Building the MLIR Python bindings require -DLLVM_BUILD_LLVM_DYLIB=ON")
endif()

include(AddMLIRPythonExtension)

################################################################################
# Copy python source tree.
################################################################################

set(PY_SRC_FILES
  mlir/__init__.py
  mlir/ir.py
  mlir/dialects/__init__.py
  mlir/dialects/std.py
)

add_custom_target(MLIRBindingsPythonSources ALL
  DEPENDS ${PY_SRC_FILES}
)

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
add_mlir_python_extension(MLIRBindingsPythonExtension _mlir
  INSTALL_DIR
    python
  SOURCES
    MainModule.cpp
    IRModules.cpp
    Pass.cpp
    PybindUtils.cpp
)
add_dependencies(MLIRBindingsPythonExtension MLIRBindingsPythonSources)

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
