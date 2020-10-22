################################################################################
# Copy python source tree.
################################################################################

set(PY_SRC_FILES
  mlir/__init__.py
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
# Build python extension
################################################################################

# Normally on unix-like platforms, extensions are built as "MODULE" libraries
# and do not explicitly link to the python shared object. This allows for
# some greater deployment flexibility since the extension will bind to
# symbols in the python interpreter on load. However, it also keeps the
# linker from erroring on undefined symbols, leaving this to (usually obtuse)
# runtime errors. Building in "SHARED" mode with an explicit link to the
# python libraries allows us to build with the expectation of no undefined
# symbols, which is better for development. Note that not all python
# configurations provide build-time libraries to link against, in which
# case, we fall back to MODULE linking.
if(PYTHON_LIBRARIES STREQUAL "" OR NOT MLIR_PYTHON_BINDINGS_VERSION_LOCKED)
  set(PYEXT_LINK_MODE MODULE)
  set(PYEXT_LIBADD)
else()
  set(PYEXT_LINK_MODE SHARED)
  set(PYEXT_LIBADD ${PYTHON_LIBRARIES})
endif()

# The actual extension library produces a shared-object or DLL and has
# sources that must be compiled in accordance with pybind11 needs (RTTI and
# exceptions).
# TODO: Link the libraries separately once a helper function is available
# to more generically add a pybind11 compliant library.
add_library(MLIRBindingsPythonExtension ${PYEXT_LINK_MODE}
  MainModule.cpp
  IRModules.cpp
  PybindUtils.cpp
)

target_include_directories(MLIRBindingsPythonExtension PRIVATE
  "${PYTHON_INCLUDE_DIRS}"
  "${pybind11_INCLUDE_DIRS}")

# The extension itself must be compiled with RTTI and exceptions enabled.
# Also, some warning classes triggered by pybind11 are disabled.
target_compile_options(MLIRBindingsPythonExtension PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    # Enable RTTI and exceptions.
    -frtti -fexceptions
    # Noisy pybind warnings
    -Wno-unused-value
    -Wno-covered-switch-default
  >
  $<$<CXX_COMPILER_ID:MSVC>:
    # Enable RTTI and exceptions.
    /EHsc /GR>
)

# Configure the output to match python expectations.
set_target_properties(
  MLIRBindingsPythonExtension PROPERTIES
    # Build-time RPath layouts require to be a directory one up from the
    # binary root.
    # TODO: Don't reference the LLVM_BINARY_DIR here: the invariant is that
    # the output directory must be at the same level of the lib directory
    # where libMLIR.so is installed. This is presently not optimal from a
    # project separation perspective and a discussion on how to better
    # segment MLIR libraries needs to happen.
    LIBRARY_OUTPUT_DIRECTORY ${LLVM_BINARY_DIR}/python
    OUTPUT_NAME "_mlir"
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_SUFFIX}${PYTHON_MODULE_EXTENSION}"
)

# pybind11 requires binding code to be compiled with -fvisibility=hidden
# For static linkage, better code can be generated if the entire project
# compiles that way, but that is not enforced here. Instead, include a linker
# script that explicitly hides anything but the PyInit_* symbols, allowing gc
# to take place.
# TODO: Add a Windows .def file and figure out the right thing to do on MacOS.
set_target_properties(
  MLIRBindingsPythonExtension PROPERTIES CXX_VISIBILITY_PRESET "hidden")

set(PYEXT_DEPS)
if(LLVM_BUILD_LLVM_DYLIB)
  list(APPEND PYEXT_DEPS
    # Depend on libMLIR.so first so that deps primarily come from the shared
    # library.
    MLIR
  )
endif()

# Full static dependencies are also added and will augment what is in the
# shared lib if needed (or in fully static builds, will result in mondo-built
# extension).
list(APPEND PYEXT_DEPS
  # Depend only on the MLIR C-API.
  MLIRCAPIIR
  MLIRCAPIRegistration
)

target_link_libraries(MLIRBindingsPythonExtension
  PRIVATE
  ${PYEXT_DEPS}
  ${PYEXT_LIBADD}
)

add_dependencies(MLIRBindingsPythonExtension MLIRBindingsPythonSources)
llvm_setup_rpath(MLIRBindingsPythonExtension)

################################################################################
# Install
################################################################################

install(TARGETS MLIRBindingsPythonExtension
  COMPONENT MLIRBindingsPythonExtension
  LIBRARY DESTINATION python
  ARCHIVE DESTINATION python
  # NOTE: Even on DLL-platforms, extensions go in the lib directory tree.
  RUNTIME DESTINATION python)

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
    install-MLIRBindingsPythonExtension
    DEPENDS MLIRBindingsPythonExtension
    COMPONENT MLIRBindingsPythonExtension)
  add_llvm_install_targets(
    install-MLIRBindingsPythonSources
    DEPENDS MLIRBindingsPythonSources
    COMPONENT MLIRBindingsPythonSources)
endif()
