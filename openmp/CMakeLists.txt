cmake_minimum_required(VERSION 3.13.4)

# Add cmake directory to search for custom cmake functions.
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# llvm/runtimes/ will set OPENMP_STANDALONE_BUILD.
if (OPENMP_STANDALONE_BUILD OR "${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  set(OPENMP_STANDALONE_BUILD TRUE)
  project(openmp C CXX)
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
  set(OPENMP_LLVM_TOOLS_DIR "" CACHE PATH "Path to LLVM tools for testing.")
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
if (APPLE OR WIN32 OR NOT OPENMP_HAVE_STD_CPP14_FLAG)
  set(ENABLE_LIBOMPTARGET OFF)
endif()

set(ENABLE_LIBOMPTARGET_PROFILING OFF)
if (ENABLE_LIBOMPTARGET AND NOT LLVM_RUNTIMES_BUILD)
  set(ENABLE_LIBOMPTARGET_PROFILING ON)
endif()

option(OPENMP_ENABLE_LIBOMPTARGET "Enable building libomptarget for offloading."
       ${ENABLE_LIBOMPTARGET})
option(OPENMP_ENABLE_LIBOMPTARGET_PROFILING "Enable time profiling for libomptarget."
       ${ENABLE_LIBOMPTARGET_PROFILING})
option(OPENMP_ENABLE_LIBOMP_PROFILING "Enable time profiling for libomp." OFF)

# Build host runtime library, after LIBOMPTARGET variables are set since they are needed
# to enable time profiling support in the OpenMP runtime.
add_subdirectory(runtime)

# Build libompd.so
add_subdirectory(libompd)

if (OPENMP_ENABLE_LIBOMPTARGET)
  # Check that the library can actually be built.
  if (APPLE OR WIN32)
    message(FATAL_ERROR "libomptarget cannot be built on Windows and MacOS X!")
  elseif (NOT OPENMP_HAVE_STD_CPP14_FLAG)
    message(FATAL_ERROR "Host compiler must support C++14 to build libomptarget!")
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

# Build documentation
add_subdirectory(docs)

# Now that we have seen all testsuites, create the check-openmp target.
construct_check_openmp_target()
