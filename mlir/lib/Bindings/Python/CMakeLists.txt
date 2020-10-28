# Normally on unix-like platforms, extensions are built as "MODULE" libraries
# and do not explicitly link to the python shared object. This allows for
# some greater deployment flexibility since the extension will bind to
# symbols in the python interpreter on load. However, it also keeps the
# linker from erroring on undefined symbols, leaving this to (usually obtuse)
# runtime errors. Building in "SHARED" mode with an explicit link to the
# python libraries allows us to build with the expectation of no undefined
# symbols, which is better for development.
if(MLIR_PYTHON_BINDINGS_VERSION_LOCKED)
  set(PYEXT_LINK_MODE SHARED)
  set(PYEXT_LIBADD ${PYTHON_LIBRARIES})
else()
  set(PYEXT_LINK_MODE MODULE)
  set(PYEXT_LIBADD)
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
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
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
if(NOT MSVC AND NOT APPLE)
  set_target_properties(MLIRBindingsPythonExtension
    PROPERTIES LINK_FLAGS
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/unix_version.lds")
endif()

target_link_libraries(MLIRBindingsPythonExtension
  PRIVATE
  MLIRCAPIIR
  MLIRCAPIRegistration
  ${PYEXT_LIBADD}
)
