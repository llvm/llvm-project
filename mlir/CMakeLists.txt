# MLIR project.

# Check if MLIR is built as a standalone project.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(mlir)
  cmake_minimum_required(VERSION 3.13.4)

  find_package(LLVM CONFIG REQUIRED)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${LLVM_CMAKE_DIR})
  include(HandleLLVMOptions)
  include(AddLLVM)
  include(TableGen)

  include_directories(${LLVM_INCLUDE_DIRS})

  set(LLVM_MAIN_SRC_DIR ${CMAKE_SOURCE_DIR}/../llvm CACHE PATH
      "Path to LLVM source tree")
  set(UNITTEST_DIR ${LLVM_MAIN_SRC_DIR}/utils/unittest)
  if(EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h)
    add_subdirectory(${UNITTEST_DIR} utils/unittest)
  endif()

  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
endif()

set(MLIR_MAIN_SRC_DIR     ${CMAKE_CURRENT_SOURCE_DIR}  )
set(MLIR_MAIN_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include )

set(MLIR_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR})
set(MLIR_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(MLIR_TOOLS_DIR   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include(AddMLIR)

# Forbid implicit function declaration: this may lead to subtle bugs and we
# don't have a reason to support this.
check_c_compiler_flag("-Werror=implicit-function-declaration" C_SUPPORTS_WERROR_IMPLICIT_FUNCTION_DECLARATION)
append_if(C_SUPPORTS_WERROR_IMPLICIT_FUNCTION_DECLARATION "-Werror=implicit-function-declaration" CMAKE_C_FLAGS)

# Installing the headers and docs needs to depend on generating any public
# tablegen'd targets.
# mlir-generic-headers are dialect-independent.
add_custom_target(mlir-generic-headers)
set_target_properties(mlir-generic-headers PROPERTIES FOLDER "Misc")
# mlir-headers may be dialect-dependent.
add_custom_target(mlir-headers)
set_target_properties(mlir-headers PROPERTIES FOLDER "Misc")
add_dependencies(mlir-headers mlir-generic-headers)
add_custom_target(mlir-doc)

# Build the CUDA conversions and run according tests if the NVPTX backend
# is available
if ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(MLIR_CUDA_CONVERSIONS_ENABLED 1)
else()
  set(MLIR_CUDA_CONVERSIONS_ENABLED 0)
endif()
# TODO: we should use a config.h file like LLVM does
add_definitions(-DMLIR_CUDA_CONVERSIONS_ENABLED=${MLIR_CUDA_CONVERSIONS_ENABLED})

# Build the ROCm conversions and run according tests if the AMDGPU backend
# is available
if ("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(MLIR_ROCM_CONVERSIONS_ENABLED 1)
else()
  set(MLIR_ROCM_CONVERSIONS_ENABLED 0)
endif()
add_definitions(-DMLIR_ROCM_CONVERSIONS_ENABLED=${MLIR_ROCM_CONVERSIONS_ENABLED})

set(MLIR_CUDA_RUNNER_ENABLED 0 CACHE BOOL "Enable building the mlir CUDA runner")
set(MLIR_ROCM_RUNNER_ENABLED 0 CACHE BOOL "Enable building the mlir ROCm runner")
set(MLIR_SPIRV_CPU_RUNNER_ENABLED 0 CACHE BOOL "Enable building the mlir SPIR-V cpu runner")
set(MLIR_VULKAN_RUNNER_ENABLED 0 CACHE BOOL "Enable building the mlir Vulkan runner")

option(MLIR_INCLUDE_TESTS
       "Generate build targets for the MLIR unit tests."
       ${LLVM_INCLUDE_TESTS})

option(MLIR_INCLUDE_INTEGRATION_TESTS
       "Generate build targets for the MLIR integration tests.")

#-------------------------------------------------------------------------------
# Python Bindings Configuration
# Requires:
#   The pybind11 library can be found (set with -DPYBIND_DIR=...)
#   The python executable is correct (set with -DPython3_EXECUTABLE=...)
#
# Version locking
# ---------------
# By default, python extensions are version locked to specific Python libraries.
# This linking mode is somewhat more consistent across platforms and surfaces
# undefined symbols at link time (vs runtime). It is suitable for development
# workflows but can be disabled for more flexible deployment by
# setting -DMLIR_PYTHON_BINDINGS_VERSION_LOCKED=OFF
#-------------------------------------------------------------------------------

set(MLIR_BINDINGS_PYTHON_ENABLED 0 CACHE BOOL
       "Enables building of Python bindings.")
set(MLIR_PYTHON_BINDINGS_VERSION_LOCKED 1 CACHE BOOL
       "Links to specific python libraries, resolving all symbols.")

if(MLIR_BINDINGS_PYTHON_ENABLED)
  include(MLIRDetectPythonEnv)
  find_package(Python3 ${LLVM_MINIMUM_PYTHON_VERSION}
    COMPONENTS Interpreter Development NumPy REQUIRED)
  message(STATUS "Found python include dirs: ${Python3_INCLUDE_DIRS}")
  message(STATUS "Found python libraries: ${Python3_LIBRARIES}")
  message(STATUS "Found numpy v${Python3_NumPy_VERSION}: ${Python3_NumPy_INCLUDE_DIRS}")
  mlir_detect_pybind11_install()
  find_package(pybind11 2.6 CONFIG REQUIRED)
  message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIR}")
  message(STATUS "Python prefix = '${PYTHON_MODULE_PREFIX}', "
                 "suffix = '${PYTHON_MODULE_SUFFIX}', "
                 "extension = '${PYTHON_MODULE_EXTENSION}")
endif()

include_directories( "include")
include_directories( ${MLIR_INCLUDE_DIR})

# Adding tools/mlir-tblgen here as calling add_tablegen sets some variables like
# MLIR_TABLEGEN_EXE in PARENT_SCOPE which gets lost if that folder is included
# from another directory like tools
add_subdirectory(tools/mlir-tblgen)
add_subdirectory(tools/mlir-linalg-ods-gen)

add_subdirectory(include/mlir)
add_subdirectory(lib)
# C API needs all dialects for registration, but should be built before tests.
add_subdirectory(lib/CAPI)
if (MLIR_INCLUDE_TESTS)
  add_definitions(-DMLIR_INCLUDE_TESTS)
  add_custom_target(MLIRUnitTests)
  if (EXISTS ${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include/gtest/gtest.h)
    add_subdirectory(unittests)
  else()
    message(WARNING "gtest not found, unittests will not be available")
  endif()
  add_subdirectory(test)
endif()
# Tools needs to come late to ensure that MLIR_ALL_LIBS is populated.
# Generally things after this point may depend on MLIR_ALL_LIBS or libMLIR.so.
add_subdirectory(tools)

if( LLVM_INCLUDE_EXAMPLES )
  add_subdirectory(examples)
endif()

option(MLIR_INCLUDE_DOCS "Generate build targets for the MLIR docs."
  ${LLVM_INCLUDE_DOCS})
if (MLIR_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY include/mlir include/mlir-c
    DESTINATION include
    COMPONENT mlir-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "LICENSE.TXT"
    )

  install(DIRECTORY ${MLIR_INCLUDE_DIR}/mlir ${MLIR_INCLUDE_DIR}/mlir-c
    DESTINATION include
    COMPONENT mlir-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.gen"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "config.h" EXCLUDE
    )

  if (NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-mlir-headers
                             DEPENDS mlir-headers
                             COMPONENT mlir-headers)
  endif()
endif()

add_subdirectory(cmake/modules)
