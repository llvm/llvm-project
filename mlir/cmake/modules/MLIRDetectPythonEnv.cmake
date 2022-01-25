# Macros and functions related to detecting details of the Python environment.

# Finds and configures python packages needed to build MLIR Python bindings.
macro(mlir_configure_python_dev_packages)
  if(CMAKE_VERSION VERSION_LESS "3.15.0")
  message(SEND_ERROR
      "Building MLIR Python bindings is known to rely on CMake features "
      "that require at least version 3.15. Recommend upgrading to 3.18+ "
      "for full support. Detected current version: ${CMAKE_VERSION}")
  endif()

  # After CMake 3.18, we are able to limit the scope of the search to just
  # Development.Module. Searching for Development will fail in situations where
  # the Python libraries are not available. When possible, limit to just
  # Development.Module.
  # See https://pybind11.readthedocs.io/en/stable/compiling.html#findpython-mode
  if(CMAKE_VERSION VERSION_LESS "3.18.0")
    message(WARNING
        "This version of CMake is not compatible with statically built Python "
        "installations. If Python fails to detect below this may apply to you. "
        "Recommend upgrading to at least CMake 3.18. "
        "Detected current version: ${CMAKE_VERSION}"
    )
    set(_python_development_component Development)
  else()
    # Prime the search for python to see if there is a full
    # development package. This seems to work around cmake bugs
    # searching only for Development.Module in some environments.
    find_package(Python3 ${LLVM_MINIMUM_PYTHON_VERSION}
      COMPONENTS Development)
    set(_python_development_component Development.Module)
  endif()
  find_package(Python3 ${LLVM_MINIMUM_PYTHON_VERSION}
    COMPONENTS Interpreter ${_python_development_component} NumPy REQUIRED)
  unset(_python_development_component)
  message(STATUS "Found python include dirs: ${Python3_INCLUDE_DIRS}")
  message(STATUS "Found python libraries: ${Python3_LIBRARIES}")
  message(STATUS "Found numpy v${Python3_NumPy_VERSION}: ${Python3_NumPy_INCLUDE_DIRS}")
  mlir_detect_pybind11_install()
  find_package(pybind11 2.8 CONFIG REQUIRED)
  message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIR}")
  message(STATUS "Python prefix = '${PYTHON_MODULE_PREFIX}', "
                 "suffix = '${PYTHON_MODULE_SUFFIX}', "
                 "extension = '${PYTHON_MODULE_EXTENSION}")
endmacro()

# Detects a pybind11 package installed in the current python environment
# and sets variables to allow it to be found. This allows pybind11 to be
# installed via pip, which typically yields a much more recent version than
# the OS install, which will be available otherwise.
function(mlir_detect_pybind11_install)
  if(pybind11_DIR)
    message(STATUS "Using explicit pybind11 cmake directory: ${pybind11_DIR} (-Dpybind11_DIR to change)")
  else()
    message(STATUS "Checking for pybind11 in python path...")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}"
      -c "import pybind11;print(pybind11.get_cmake_dir(), end='')"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE STATUS
      OUTPUT_VARIABLE PACKAGE_DIR
      ERROR_QUIET)
    if(NOT STATUS EQUAL "0")
      message(STATUS "not found (install via 'pip install pybind11' or set pybind11_DIR)")
      return()
    endif()
    message(STATUS "found (${PACKAGE_DIR})")
    set(pybind11_DIR "${PACKAGE_DIR}" PARENT_SCOPE)
  endif()
endfunction()
