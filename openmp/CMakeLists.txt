cmake_minimum_required(VERSION 3.20.0)

set(LLVM_COMMON_CMAKE_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

# Add path for custom modules
list(INSERT CMAKE_MODULE_PATH 0
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
  )

# llvm/runtimes/ will set OPENMP_STANDALONE_BUILD.
if (OPENMP_STANDALONE_BUILD OR "${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  set(OPENMP_STANDALONE_BUILD TRUE)
  project(openmp C CXX ASM)
endif()

# Must go below project(..)
include(GNUInstallDirs)

if (OPENMP_STANDALONE_BUILD)
  # CMAKE_BUILD_TYPE was not set, default to Release.
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()

  # Group common settings.
  set(OPENMP_ENABLE_WERROR FALSE CACHE BOOL
    "Enable -Werror flags to turn warnings into errors for supporting compilers.")
  set(OPENMP_LIBDIR_SUFFIX "" CACHE STRING
    "Suffix of lib installation directory, e.g. 64 => lib64")
  # Do not use OPENMP_LIBDIR_SUFFIX directly, use OPENMP_INSTALL_LIBDIR.
  set(OPENMP_INSTALL_LIBDIR "lib${OPENMP_LIBDIR_SUFFIX}")

  # Group test settings.
  set(OPENMP_TEST_C_COMPILER ${CMAKE_C_COMPILER} CACHE STRING
    "C compiler to use for testing OpenMP runtime libraries.")
  set(OPENMP_TEST_CXX_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING
    "C++ compiler to use for testing OpenMP runtime libraries.")
  set(OPENMP_TEST_Fortran_COMPILER ${CMAKE_Fortran_COMPILER} CACHE STRING
    "FORTRAN compiler to use for testing OpenMP runtime libraries.")
  set(OPENMP_LLVM_TOOLS_DIR "" CACHE PATH "Path to LLVM tools for testing.")

  set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
  set(CMAKE_CXX_STANDARD_REQUIRED NO)
  set(CMAKE_CXX_EXTENSIONS NO)
else()
  set(OPENMP_ENABLE_WERROR ${LLVM_ENABLE_WERROR})
  # If building in tree, we honor the same install suffix LLVM uses.
  set(OPENMP_INSTALL_LIBDIR "lib${LLVM_LIBDIR_SUFFIX}")

  if (NOT MSVC)
    set(OPENMP_TEST_C_COMPILER ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang)
    set(OPENMP_TEST_CXX_COMPILER ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang++)
  else()
    set(OPENMP_TEST_C_COMPILER ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang.exe)
    set(OPENMP_TEST_CXX_COMPILER ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang++.exe)
  endif()

  # Check for flang
  if (NOT MSVC)
    set(OPENMP_TEST_Fortran_COMPILER ${LLVM_RUNTIME_OUTPUT_INTDIR}/flang-new)
  else()
    set(OPENMP_TEST_Fortran_COMPILER ${LLVM_RUNTIME_OUTPUT_INTDIR}/flang-new.exe)
  endif()

  # Set fortran test compiler if flang is found
  if (EXISTS "${OPENMP_TEST_Fortran_COMPILER}")
    message("Using local flang build at ${OPENMP_TEST_Fortran_COMPILER}")
  else()
    unset(OPENMP_TEST_Fortran_COMPILER)
  endif()

  # If not standalone, set CMAKE_CXX_STANDARD but don't set the global cache value,
  # only set it locally for OpenMP.
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED NO)
  set(CMAKE_CXX_EXTENSIONS NO)
endif()

# Check and set up common compiler flags.
include(config-ix)
include(HandleOpenMPOptions)

# Set up testing infrastructure.
include(OpenMPTesting)

set(OPENMP_TEST_FLAGS "" CACHE STRING
  "Extra compiler flags to send to the test compiler.")
set(OPENMP_TEST_OPENMP_FLAGS ${OPENMP_TEST_COMPILER_OPENMP_FLAGS} CACHE STRING
  "OpenMP compiler flag to use for testing OpenMP runtime libraries.")

set(ENABLE_LIBOMPTARGET ON)
# Currently libomptarget cannot be compiled on Windows or MacOS X.
# Since the device plugins are only supported on Linux anyway,
# there is no point in trying to compile libomptarget on other OSes.
# 32-bit systems are not supported either.
if (APPLE OR WIN32 OR NOT "cxx_std_17" IN_LIST CMAKE_CXX_COMPILE_FEATURES OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(ENABLE_LIBOMPTARGET OFF)
endif()

option(OPENMP_ENABLE_LIBOMPTARGET "Enable building libomptarget for offloading."
       ${ENABLE_LIBOMPTARGET})
option(OPENMP_ENABLE_LIBOMP_PROFILING "Enable time profiling for libomp." OFF)

# Header install location
if(${OPENMP_STANDALONE_BUILD})
  set(LIBOMP_HEADERS_INSTALL_PATH "${CMAKE_INSTALL_INCLUDEDIR}")
else()
  include(GetClangResourceDir)
  get_clang_resource_dir(LIBOMP_HEADERS_INSTALL_PATH SUBDIR include)
endif()

# Build host runtime library, after LIBOMPTARGET variables are set since they are needed
# to enable time profiling support in the OpenMP runtime.
add_subdirectory(runtime)

if (OPENMP_ENABLE_LIBOMPTARGET)
  # Check that the library can actually be built.
  if (APPLE OR WIN32)
    message(FATAL_ERROR "libomptarget cannot be built on Windows and MacOS X!")
  elseif (NOT "cxx_std_17" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(FATAL_ERROR "Host compiler must support C++17 to build libomptarget!")
  elseif (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "libomptarget on 32-bit systems are not supported!")
  endif()

  add_subdirectory(libomptarget)
endif()

set(ENABLE_OMPT_TOOLS ON)
# Currently tools are not tested well on Windows or MacOS X.
if (APPLE OR WIN32)
  set(ENABLE_OMPT_TOOLS OFF)
endif()

option(OPENMP_ENABLE_OMPT_TOOLS "Enable building ompt based tools for OpenMP."
       ${ENABLE_OMPT_TOOLS})
if (OPENMP_ENABLE_OMPT_TOOLS)
  add_subdirectory(tools)
endif()

option(OPENMP_MSVC_NAME_SCHEME "Build dll with MSVC naming scheme." OFF)

# Build libompd.so
add_subdirectory(libompd)

# Build documentation
add_subdirectory(docs)

# Now that we have seen all testsuites, create the check-openmp target.
construct_check_openmp_target()
