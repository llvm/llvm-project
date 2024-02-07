# MLIR project.
cmake_minimum_required(VERSION 3.20.0)

if(NOT DEFINED LLVM_COMMON_CMAKE_UTILS)
  set(LLVM_COMMON_CMAKE_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
endif()
include(${LLVM_COMMON_CMAKE_UTILS}/Modules/CMakePolicy.cmake
  NO_POLICY_SCOPE)

# Check if MLIR is built as a standalone project.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(mlir)
  set(MLIR_STANDALONE_BUILD TRUE)
endif()

# Must go below project(..)
include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 17)

if(MLIR_STANDALONE_BUILD)
  find_package(LLVM CONFIG REQUIRED)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${LLVM_CMAKE_DIR})
  include(HandleLLVMOptions)
  include(AddLLVM)
  include(TableGen)

  include_directories(${LLVM_INCLUDE_DIRS})

  set(UNITTEST_DIR ${LLVM_THIRD_PARTY_DIR}/unittest)
  if(EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h)
    add_subdirectory(${UNITTEST_DIR} third-party/unittest)
  endif()

  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")

  set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")
endif()

set(MLIR_TOOLS_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}" CACHE PATH
    "Path for binary subdirectory (defaults to '${CMAKE_INSTALL_BINDIR}')")
mark_as_advanced(MLIR_TOOLS_INSTALL_DIR)

set(MLIR_MAIN_SRC_DIR     ${CMAKE_CURRENT_SOURCE_DIR}  )
set(MLIR_MAIN_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include )

set(MLIR_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR})
set(MLIR_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(MLIR_TOOLS_DIR   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# Make sure that our source directory is on the current cmake module path so
# that we can include cmake files from this directory.
list(INSERT CMAKE_MODULE_PATH 0
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
  )

include(AddMLIR)

# -BSymbolic is incompatible with TypeID
if("${CMAKE_SHARED_LINKER_FLAGS}" MATCHES "-Bsymbolic[^-]")
  message(FATAL_ERROR " MLIR does not support `-Bsymbolic` (see http://llvm.org/pr51420 ),"
          " try `-Bsymbolic-functions` instead.")
endif()

# Forbid implicit function declaration: this may lead to subtle bugs and we
# don't have a reason to support this.
check_c_compiler_flag("-Werror=implicit-function-declaration" C_SUPPORTS_WERROR_IMPLICIT_FUNCTION_DECLARATION)
append_if(C_SUPPORTS_WERROR_IMPLICIT_FUNCTION_DECLARATION "-Werror=implicit-function-declaration" CMAKE_C_FLAGS)

# Forbid mismatch between declaration and definition for class vs struct. This is
# harmless on Unix systems, but it'll be a ticking bomb for MSVC/Windows systems
# where it creeps into the ABI.
check_c_compiler_flag("-Werror=mismatched-tags" C_SUPPORTS_WERROR_MISMATCHED_TAGS)
append_if(C_SUPPORTS_WERROR_MISMATCHED_TAGS "-Werror=mismatched-tags" CMAKE_C_FLAGS)
append_if(C_SUPPORTS_WERROR_MISMATCHED_TAGS "-Werror=mismatched-tags" CMAKE_CXX_FLAGS)

# Silence a false positive GCC -Wunused-but-set-parameter warning in constexpr
# cases, by marking SelectedCase as used. See
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85827 for details. The issue is
# fixed in GCC 10.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
  check_cxx_compiler_flag("-Wno-unused-but-set-parameter" CXX_SUPPORTS_WNO_UNUSED_BUT_SET_PARAMETER)
  append_if(CXX_SUPPORTS_WNO_UNUSED_BUT_SET_PARAMETER "-Wno-unused-but-set-parameter" CMAKE_CXX_FLAGS)
endif()

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

# Only enable execution engine if the native target is available.
if(${LLVM_NATIVE_ARCH} IN_LIST LLVM_TARGETS_TO_BUILD)
  set(MLIR_ENABLE_EXECUTION_ENGINE 1)
else()
  set(MLIR_ENABLE_EXECUTION_ENGINE 0)
endif()

# Build the CUDA conversions and run according tests if the NVPTX backend
# is available
if ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(MLIR_ENABLE_CUDA_CONVERSIONS 1)
else()
  set(MLIR_ENABLE_CUDA_CONVERSIONS 0)
endif()
# TODO: we should use a config.h file like LLVM does
add_definitions(-DMLIR_CUDA_CONVERSIONS_ENABLED=${MLIR_ENABLE_CUDA_CONVERSIONS})

# Build the ROCm conversions and run according tests if the AMDGPU backend
# is available.
if ("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(MLIR_ENABLE_ROCM_CONVERSIONS 1)
else()
  set(MLIR_ENABLE_ROCM_CONVERSIONS 0)
endif()
add_definitions(-DMLIR_ROCM_CONVERSIONS_ENABLED=${MLIR_ENABLE_ROCM_CONVERSIONS})

set(MLIR_ENABLE_DEPRECATED_GPU_SERIALIZATION 0 CACHE BOOL "Enable deprecated GPU serialization passes")
set(MLIR_ENABLE_CUDA_RUNNER 0 CACHE BOOL "Enable building the mlir CUDA runner")
set(MLIR_ENABLE_ROCM_RUNNER 0 CACHE BOOL "Enable building the mlir ROCm runner")
set(MLIR_ENABLE_SYCL_RUNNER 0 CACHE BOOL "Enable building the mlir Sycl runner")
set(MLIR_ENABLE_SPIRV_CPU_RUNNER 0 CACHE BOOL "Enable building the mlir SPIR-V cpu runner")
set(MLIR_ENABLE_VULKAN_RUNNER 0 CACHE BOOL "Enable building the mlir Vulkan runner")
set(MLIR_ENABLE_NVPTXCOMPILER 0 CACHE BOOL
    "Statically link the nvptxlibrary instead of calling ptxas as a subprocess \
    for compiling PTX to cubin")

set(MLIR_ENABLE_PDL_IN_PATTERNMATCH 1 CACHE BOOL "Enable PDL in PatternMatch")

option(MLIR_INCLUDE_TESTS
       "Generate build targets for the MLIR unit tests."
       ${LLVM_INCLUDE_TESTS})

option(MLIR_INCLUDE_INTEGRATION_TESTS
       "Generate build targets for the MLIR integration tests.")

set(MLIR_INSTALL_AGGREGATE_OBJECTS 1 CACHE BOOL
    "Installs object files needed for add_mlir_aggregate to work out of \
    tree. Package maintainers can disable this to exclude these assets if \
    not desired. Enabling this will result in object files being written \
    under lib/objects-{CMAKE_BUILD_TYPE}.")

set(MLIR_BUILD_MLIR_C_DYLIB 0 CACHE BOOL "Builds libMLIR-C shared library.")

configure_file(
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Config/mlir-config.h.cmake
  ${MLIR_INCLUDE_DIR}/mlir/Config/mlir-config.h)

#-------------------------------------------------------------------------------
# Python Bindings Configuration
# Requires:
#   The pybind11 library can be found (set with -DPYBIND_DIR=...)
#   The python executable is correct (set with -DPython3_EXECUTABLE=...)
#-------------------------------------------------------------------------------

set(MLIR_ENABLE_BINDINGS_PYTHON 0 CACHE BOOL
       "Enables building of Python bindings.")
set(MLIR_DETECT_PYTHON_ENV_PRIME_SEARCH 1 CACHE BOOL
       "Prime the python detection by searching for a full 'Development' \
       component first (temporary while diagnosing environment specific Python \
       detection issues)")
if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories( "include")
include_directories( ${MLIR_INCLUDE_DIR})

# Adding tools/mlir-tblgen here as calling add_tablegen sets some variables like
# MLIR_TABLEGEN_EXE in PARENT_SCOPE which gets lost if that folder is included
# from another directory like tools
add_subdirectory(tools/mlir-linalg-ods-gen)
add_subdirectory(tools/mlir-pdll)
add_subdirectory(tools/mlir-tblgen)
set(MLIR_TABLEGEN_EXE "${MLIR_TABLEGEN_EXE}" CACHE INTERNAL "")
set(MLIR_TABLEGEN_TARGET "${MLIR_TABLEGEN_TARGET}" CACHE INTERNAL "")
set(MLIR_PDLL_TABLEGEN_EXE "${MLIR_PDLL_TABLEGEN_EXE}" CACHE INTERNAL "")
set(MLIR_PDLL_TABLEGEN_TARGET "${MLIR_PDLL_TABLEGEN_TARGET}" CACHE INTERNAL "")

add_subdirectory(include/mlir)
add_subdirectory(lib)
# C API needs all dialects for registration, but should be built before tests.
add_subdirectory(lib/CAPI)

if (MLIR_INCLUDE_TESTS)
  add_definitions(-DMLIR_INCLUDE_TESTS)
  add_custom_target(MLIRUnitTests)
  if (EXISTS ${LLVM_THIRD_PARTY_DIR}/unittest/googletest/include/gtest/gtest.h)
    add_subdirectory(unittests)
  else()
    message(WARNING "gtest not found, unittests will not be available")
  endif()
  add_subdirectory(test)
endif()
# Tools needs to come late to ensure that MLIR_ALL_LIBS is populated.
# Generally things after this point may depend on MLIR_ALL_LIBS or libMLIR.so.
add_subdirectory(tools)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  # Python sources: built extensions come in via lib/Bindings/Python
  add_subdirectory(python)
endif()

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
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT mlir-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "LICENSE.TXT"
    )

  install(DIRECTORY ${MLIR_INCLUDE_DIR}/mlir ${MLIR_INCLUDE_DIR}/mlir-c
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
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

# Custom target to install all mlir libraries
add_custom_target(mlir-libraries)
set_target_properties(mlir-libraries PROPERTIES FOLDER "Misc")

if (NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-mlir-libraries
                           DEPENDS mlir-libraries
                           COMPONENT mlir-libraries)
endif()

get_property(MLIR_LIBS GLOBAL PROPERTY MLIR_ALL_LIBS)
if(MLIR_LIBS)
  list(REMOVE_DUPLICATES MLIR_LIBS)
  foreach(lib ${MLIR_LIBS})
    add_dependencies(mlir-libraries ${lib})
    if(NOT LLVM_ENABLE_IDE)
      add_dependencies(install-mlir-libraries install-${lib})
      add_dependencies(install-mlir-libraries-stripped install-${lib}-stripped)
    endif()
  endforeach()
endif()

add_subdirectory(cmake/modules)

if (MLIR_ENABLE_PYTHON_BENCHMARKS)
  add_subdirectory(utils/mbr)
endif()

if(MLIR_STANDALONE_BUILD)
  llvm_distribution_add_targets()
endif()
