# See for https://openmp.llvm.org/SupportAndFAQ.html for instructions on how
# to build offload with CMake.

cmake_minimum_required(VERSION 3.20.0)
set(LLVM_SUBPROJECT_TITLE "liboffload")

if(DEFINED LIBOMP_SHARED_LINKER_FLAGS)
  set(CMAKE_SHARED_LINKER_FLAGS "${LIBOMP_SHARED_LINKER_FLAGS}")
endif()

if(DEFINED LIBOMP_INSTALL_RPATH)
  set(CMAKE_INSTALL_RPATH "${LIBOMP_INSTALL_RPATH}")
endif()

# Permit redefining OPENMP_STANDALONE_BUILD when doing a runtimes build.
if (OPENMP_STANDALONE_BUILD OR "${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  set(OPENMP_STANDALONE_BUILD TRUE)
  project(offload C CXX ASM)
else()
  set(OPENMP_STANDALONE_BUILD FALSE)
endif()

if(LIBOMPTARGET_NO_SANITIZER_AMDGPU)
  set(SANITIZER_AMDGPU FALSE)
endif()

# Check that the library can actually be built.
if(APPLE OR WIN32 OR WASM)
  message(WARNING "libomptarget cannot be built on Windows and MacOS X!")
  return()
elseif(NOT "cxx_std_17" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
  message(WARNING "Host compiler must support C++17 to build libomptarget!")
  return()
elseif(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(WARNING "libomptarget on 32-bit systems is not supported!")
  return()
endif()

set(OFFLOAD_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if(OPENMP_STANDALONE_BUILD)
  set(OFFLOAD_LIBDIR_SUFFIX "" CACHE STRING
    "Suffix of lib installation directory, e.g. 64 => lib64")
  set(OFFLOAD_INSTALL_LIBDIR "lib${OFFLOAD_LIBDIR_SUFFIX}" CACHE STRING
      "Path where built offload libraries should be installed.")
  include(GNUInstallDirs)
  set(LIBOMP_HEADERS_INSTALL_PATH "${CMAKE_INSTALL_INCLUDEDIR}")
else()
  include(GetClangResourceDir)
  get_clang_resource_dir(LIBOMP_HEADERS_INSTALL_PATH SUBDIR include)
  # When building in tree we install the runtime according to the LLVM settings.
  if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT APPLE)
    set(OFFLOAD_INSTALL_LIBDIR lib${LLVM_LIBDIR_SUFFIX}/${LLVM_DEFAULT_TARGET_TRIPLE} CACHE STRING
      "Path where built offload libraries should be installed.")
  else()
    set(OFFLOAD_INSTALL_LIBDIR "lib${LLVM_LIBDIR_SUFFIX}" CACHE STRING
      "Path where built offload libraries should be installed.")
  endif()
endif()

set(LLVM_COMMON_CMAKE_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

option(OFFLOAD_INCLUDE_TESTS "Generate and build offload tests." ${LLVM_INCLUDE_TESTS})

# Add path for custom modules
list(INSERT CMAKE_MODULE_PATH 0
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  "${CMAKE_CURRENT_SOURCE_DIR}/../runtimes/cmake/Modules"
  "${LLVM_COMMON_CMAKE_UTILS}"
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
  )

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

  # Used by llvm_add_tool() and tests.
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR})

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
    set(OPENMP_TEST_C_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang)
    set(OPENMP_TEST_CXX_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang++)
  else()
    set(OPENMP_TEST_C_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang.exe)
    set(OPENMP_TEST_CXX_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang++.exe)
  endif()

  # Check for flang
  if (NOT MSVC)
    set(OPENMP_TEST_Fortran_COMPILER ${LLVM_TOOLS_BINARY_DIR}/flang)
  else()
    set(OPENMP_TEST_Fortran_COMPILER ${LLVM_TOOLS_BINARY_DIR}/flang.exe)
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

# Emit a warning for people who haven't updated their build.
if(NOT "openmp" IN_LIST RUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES AND
   NOT "openmp" IN_LIST RUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES)
  message(WARNING "Building the offloading runtime with no device library. See "
                  "https://openmp.llvm.org/SupportAndFAQ.html#q-how-to-build-an-openmp-gpu-offload-capable-compiler.html "
                  "for more information.")
endif()

# Set the path of all resulting libraries to a unified location so that it can
# be used for testing.
set(LIBOMPTARGET_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LIBOMPTARGET_LIBRARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBOMPTARGET_LIBRARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LIBOMPTARGET_LIBRARY_DIR})

if(NOT LLVM_LIBRARY_OUTPUT_INTDIR)
  set(LIBOMPTARGET_INTDIR ${LIBOMPTARGET_LIBRARY_DIR})
else()
  set(LIBOMPTARGET_INTDIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
endif()

# Get dependencies for the different components of the project.
include(LibomptargetGetDependencies)

# Set up testing infrastructure.
include(OpenMPTesting)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-Werror=global-constructors OFFLOAD_HAVE_WERROR_CTOR)

# LLVM source tree is required at build time for libomptarget
if (NOT LIBOMPTARGET_LLVM_INCLUDE_DIRS)
  message(FATAL_ERROR "Missing definition for LIBOMPTARGET_LLVM_INCLUDE_DIRS")
endif()

if(DEFINED LIBOMPTARGET_BUILD_CUDA_PLUGIN OR
   DEFINED LIBOMPTARGET_BUILD_AMDGPU_PLUGIN)
  message(WARNING "Option removed, use 'LIBOMPTARGET_PLUGINS_TO_BUILD' instead")
endif()

set(LIBOMPTARGET_ALL_PLUGIN_TARGETS amdgpu cuda host)
set(LIBOMPTARGET_PLUGINS_TO_BUILD "all" CACHE STRING
    "Semicolon-separated list of plugins to use: cuda, amdgpu, host or \"all\".")

if(LIBOMPTARGET_PLUGINS_TO_BUILD STREQUAL "all")
  set(LIBOMPTARGET_PLUGINS_TO_BUILD ${LIBOMPTARGET_ALL_PLUGIN_TARGETS})
endif()

if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux" AND
   "host" IN_LIST LIBOMPTARGET_PLUGINS_TO_BUILD)
  message(STATUS "Not building host plugin: only Linux systems are supported")
  list(REMOVE_ITEM LIBOMPTARGET_PLUGINS_TO_BUILD "host")
endif()
if(NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)|(aarch64)$"
        AND CMAKE_SYSTEM_NAME MATCHES "Linux"))
  if("amdgpu" IN_LIST LIBOMPTARGET_PLUGINS_TO_BUILD)
    message(STATUS "Not building AMDGPU plugin: only support AMDGPU in "
                   "Linux x86_64, ppc64le, or aarch64 hosts")
    list(REMOVE_ITEM LIBOMPTARGET_PLUGINS_TO_BUILD "amdgpu")
  endif()
  if("cuda" IN_LIST LIBOMPTARGET_PLUGINS_TO_BUILD)
    message(STATUS "Not building CUDA plugin: only support CUDA in "
                   "Linux x86_64, ppc64le, or aarch64 hosts")
    list(REMOVE_ITEM LIBOMPTARGET_PLUGINS_TO_BUILD "cuda")
  endif()
endif()
message(STATUS "Building the offload library with support for "
               "the \"${LIBOMPTARGET_PLUGINS_TO_BUILD}\" plugins")

set(LIBOMPTARGET_DLOPEN_PLUGINS "${LIBOMPTARGET_PLUGINS_TO_BUILD}" CACHE STRING
    "Semicolon-separated list of plugins to use 'dlopen' for runtime linking")

set(LIBOMPTARGET_ENUM_PLUGIN_TARGETS "")
foreach(plugin IN LISTS LIBOMPTARGET_PLUGINS_TO_BUILD)
  set(LIBOMPTARGET_ENUM_PLUGIN_TARGETS
      "${LIBOMPTARGET_ENUM_PLUGIN_TARGETS}PLUGIN_TARGET(${plugin})\n")
endforeach()
string(STRIP ${LIBOMPTARGET_ENUM_PLUGIN_TARGETS} LIBOMPTARGET_ENUM_PLUGIN_TARGETS)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Shared/Targets.def.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/Shared/Targets.def
)

include_directories(${LIBOMPTARGET_LLVM_INCLUDE_DIRS})

# This is a list of all the targets that are supported/tested right now.
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} aarch64-unknown-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} aarch64-unknown-linux-gnu-oldDriver")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} aarch64-unknown-linux-gnu-LTO")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} amdgcn-amd-amdhsa")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64le-ibm-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64le-ibm-linux-gnu-oldDriver")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64le-ibm-linux-gnu-LTO")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64-ibm-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64-ibm-linux-gnu-oldDriver")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64-ibm-linux-gnu-LTO")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} x86_64-unknown-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} x86_64-unknown-linux-gnu-oldDriver")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} x86_64-unknown-linux-gnu-LTO")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} nvptx64-nvidia-cuda")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} nvptx64-nvidia-cuda-oldDriver")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} nvptx64-nvidia-cuda-LTO")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} nvptx64-nvidia-cuda-JIT-LTO")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} s390x-ibm-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} s390x-ibm-linux-gnu-LTO")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} riscv64-unknown-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} riscv64-unknown-linux-gnu-LTO")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} loongarch64-unknown-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} loongarch64-unknown-linux-gnu-LTO")

# Once the plugins for the different targets are validated, they will be added to
# the list of supported targets in the current system.
set (LIBOMPTARGET_SYSTEM_TARGETS "")
set (LIBOMPTARGET_TESTED_PLUGINS "")

# Check whether using debug mode. In debug mode, allow dumping progress
# messages at runtime by default. Otherwise, it can be enabled
# independently using the LIBOMPTARGET_ENABLE_DEBUG option.
string( TOLOWER "${CMAKE_BUILD_TYPE}" LIBOMPTARGET_CMAKE_BUILD_TYPE)
if(LIBOMPTARGET_CMAKE_BUILD_TYPE MATCHES debug)
  option(LIBOMPTARGET_ENABLE_DEBUG "Allow debug output with the environment variable LIBOMPTARGET_DEBUG=1" ON)
  add_definitions(-DDEBUG)
else()
  option(LIBOMPTARGET_ENABLE_DEBUG "Allow debug output with the environment variable LIBOMPTARGET_DEBUG=1" OFF)
endif()
if(LIBOMPTARGET_ENABLE_DEBUG)
  add_definitions(-DOMPTARGET_DEBUG)
endif()

# OMPD support for libomptarget (currently only with cuda)
set(LIBOMPTARGET_OMPD_SUPPORT FALSE CACHE BOOL "OMPD-support?")
if (LIBOMPTARGET_OMPD_SUPPORT)
  add_definitions(-DOMPD_SUPPORT=1)
endif()

# No exceptions and no RTTI, except if requested.
set(offload_compile_flags -fno-exceptions)
if(NOT LLVM_ENABLE_RTTI)
  set(offload_compile_flags ${offload_compile_flags} -fno-rtti)
endif()
if(OFFLOAD_HAVE_WERROR_CTOR)
#   list(APPEND offload_compile_flags -Werror=global-constructors)
endif()

# TODO: Consider enabling LTO by default if supported.
# https://cmake.org/cmake/help/latest/module/CheckIPOSupported.html can be used
# to test for working LTO. However, before CMake 3.24 this will test the
# default linker and ignore options such as LLVM_ENABLE_LLD. As a result, CMake
# would test whether LTO works with the default linker but build with another one.
# In a typical scenario, libomptarget is compiled with the in-tree Clang, but
# linked with ld.gold, which requires the LLVMgold plugin, when it actually
# would work with the lld linker (or also fail because the system lld is too old
# to understand opaque pointers). Using gcc as the compiler would pass the test, but fail
# when linking with lld since does not understand gcc's LTO format.
set(LIBOMPTARGET_USE_LTO FALSE CACHE BOOL "Use LTO for the offload runtimes if available")
if (LIBOMPTARGET_USE_LTO)
  # CMake sets CMAKE_CXX_COMPILE_OPTIONS_IPO depending on the compiler and is
  # also what CheckIPOSupported uses to test support.
  list(APPEND offload_compile_flags ${CMAKE_CXX_COMPILE_OPTIONS_IPO})
  list(APPEND offload_link_flags ${CMAKE_CXX_COMPILE_OPTIONS_IPO})
endif()

if(OPENMP_STANDALONE_BUILD)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    execute_process(
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND ${CMAKE_CXX_COMPILER} --print-resource-dir
      RESULT_VARIABLE COMMAND_RETURN_CODE
      OUTPUT_VARIABLE COMPILER_RESOURCE_DIR
    )
  endif()

  set(LIBOMP_HAVE_OMPT_SUPPORT FALSE)
  set(LIBOMP_OMPT_SUPPORT FALSE)

  find_path (
    LIBOMP_OMP_TOOLS_INCLUDE_DIR
    NAMES
      omp-tools.h
    HINTS
    ${COMPILER_RESOURCE_DIR}/include
    ${CMAKE_INSTALL_PREFIX}/include
  )

  if(LIBOMP_OMP_TOOLS_INCLUDE_DIR)
    set(LIBOMP_HAVE_OMPT_SUPPORT TRUE)
    set(LIBOMP_OMPT_SUPPORT TRUE)
  endif()

  # LLVM_LIBRARY_DIRS set by find_package(LLVM) in LibomptargetGetDependencies
  find_library (
    LIBOMP_STANDALONE
    NAMES
      omp
    HINTS
      ${CMAKE_INSTALL_PREFIX}/lib
      ${LLVM_LIBRARY_DIRS}
    REQUIRED
  )

  find_path (
    LIBOMP_INCLUDE_DIR
    NAMES
      omp.h
    HINTS
    ${COMPILER_RESOURCE_DIR}/include
    ${CMAKE_INSTALL_PREFIX}/include
  )

  get_filename_component(LIBOMP_LIBRARY_DIR ${LIBOMP_STANDALONE} DIRECTORY)

  set(OPENMP_TEST_FLAGS "" CACHE STRING
    "Extra compiler flags to send to the test compiler.")
  set(OPENMP_TEST_OPENMP_FLAGS ${OPENMP_TEST_COMPILER_OPENMP_FLAGS} CACHE STRING
    "OpenMP compiler flag to use for testing OpenMP runtime libraries.")
  set(LIBOMPTARGET_OPENMP_HEADER_FOLDER "${LIBOMP_INCLUDE_DIR}" CACHE STRING
    "Path to folder containing omp.h")
  set(LIBOMPTARGET_OPENMP_HOST_RTL_FOLDER "${LIBOMP_LIBRARY_DIR}" CACHE STRING
    "Path to folder containing libomp.so, and libLLVMSupport.so with profiling enabled")
endif()

macro(pythonize_bool var)
if (${var})
  set(${var} True)
else()
  set(${var} False)
endif()
endmacro()

if(OPENMP_STANDALONE_BUILD OR TARGET omp)
  # Check LIBOMP_HAVE_VERSION_SCRIPT_FLAG
  include(LLVMCheckCompilerLinkerFlag)
  if(NOT APPLE)
    llvm_check_compiler_linker_flag(C "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/../openmp/runtime/src/exports_test_so.txt" LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
  endif()
endif()

# OMPT support for libomptarget
# Follow host OMPT support and check if host support has been requested.
# LIBOMP_HAVE_OMPT_SUPPORT indicates whether host OMPT support has been implemented.
# LIBOMP_OMPT_SUPPORT indicates whether host OMPT support has been requested (default is ON).
# LIBOMPTARGET_OMPT_SUPPORT indicates whether target OMPT support has been requested (default is ON).
set(OMPT_TARGET_DEFAULT FALSE)
if ((LIBOMP_HAVE_OMPT_SUPPORT) AND (NOT WIN32))
  set (OMPT_TARGET_DEFAULT TRUE)
endif()
set(LIBOMPTARGET_OMPT_SUPPORT ${OMPT_TARGET_DEFAULT} CACHE BOOL "OMPT-target-support?")
if (LIBOMPTARGET_OMPT_SUPPORT)
  add_definitions(-DOMPT_SUPPORT=1)
  message(STATUS "OMPT target enabled")
else()
  set(LIBOMPTARGET_OMPT_SUPPORT FALSE)
  message(STATUS "OMPT target disabled")
endif()

include_directories(include)
pythonize_bool(LIBOMPTARGET_OMPT_SUPPORT)

if(${LLVM_LIBC_GPU_BUILD})
  set(LIBOMPTARGET_HAS_LIBC TRUE)
else()
  set(LIBOMPTARGET_HAS_LIBC FALSE)
endif()
set(LIBOMPTARGET_GPU_LIBC_SUPPORT ${LIBOMPTARGET_HAS_LIBC} CACHE BOOL
    "Libomptarget support for the GPU libc")
pythonize_bool(LIBOMPTARGET_GPU_LIBC_SUPPORT)

set(LIBOMPTARGET_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Various LLVM_TOOLS are needed to build libomptarget
find_package(LLVM QUIET CONFIG PATHS
   ${LLVM_INSTALL_PREFIX}
   ${LIBOMPTARGET_NVPTX_CUDA_COMPILER_DIR}
   ${LIBOMPTARGET_NVPTX_CUDA_LINKER_DIR}
   ${CMAKE_CXX_COMPILER_DIR}
   NO_DEFAULT_PATH)
if(LLVM_DIR)
  message(" -- LLVM found at ${LLVM_DIR}")
  find_program(CLANG_TOOL clang PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  find_program(CLANG_OFFLOAD_BUNDLER_TOOL clang-offload-bundler PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  find_program(AR_TOOL llvm-ar PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  find_program(LINK_TOOL llvm-link PATHS ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)
  find_program(OPT_TOOL opt PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
elseif (LLVM_TOOL_CLANG_BUILD AND NOT CMAKE_CROSSCOMPILING AND NOT OPENMP_STANDALONE_BUILD)
  # LLVM in-tree builds may use CMake target names to discover the tools.
  set(CLANG_TOOL $<TARGET_FILE:clang>)
  set(CLANG_OFFLOAD_BUNDLER_TOOL $<TARGET_FILE:clang-offload-bundler>)
  set(AR_TOOL $<TARGET_FILE:llvm-ar>)
  set(LINK_TOOL $<TARGET_FILE:llvm-link>)
  set(OPT_TOOL $<TARGET_FILE:opt>)
else()
  message(" ====== WARNING! no LLVM found! some libomptarget components may be skipped")
endif()

set(LIBOMPTARGET_BINARY_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
message(STATUS "OpenMP tools dir in libomptarget: ${LIBOMP_OMP_TOOLS_INCLUDE_DIR}")
if(LIBOMP_OMP_TOOLS_INCLUDE_DIR)
  include_directories(${LIBOMP_OMP_TOOLS_INCLUDE_DIR})
endif()

set(LIBOMPTARGET_LLVM_LIBRARY_DIR "${LLVM_LIBRARY_DIR}" CACHE STRING
  "Path to folder containing llvm library libomptarget.so")
set(LIBOMPTARGET_LLVM_LIBRARY_INTDIR "${LIBOMPTARGET_INTDIR}" CACHE STRING
  "Path to folder where intermediate libraries will be output")

if(SANITIZER_AMDGPU)
  add_definitions(-DSANITIZER_AMDGPU=1)
  # Check for COMGr package , ASan requires COMGr with minimum version 2.4
  find_package(amd_comgr QUIET 2.4.0 HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
endif()

# An Emissary API is a subset of Host APIs that are executed from Offload devices
# using the offload RPC mechanism such as FORTRAN IO runtime and MPI.
option(OFFLOAD_ENABLE_EMISSARY_APIS "Enable build of GPU Emissary APIs" ON)
if(OFFLOAD_ENABLE_EMISSARY_APIS)
  # Header install location
  add_definitions(-DOFFLOAD_ENABLE_EMISSARY_APIS)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../openmp/device/include/EmissaryIds.h
    DESTINATION ${LIBOMP_HEADERS_INSTALL_PATH})
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../openmp/device/include/EmissaryMPI.h
    DESTINATION ${LIBOMP_HEADERS_INSTALL_PATH})
endif()

add_subdirectory(tools/offload-tblgen)

# Build offloading plugins and device RTLs if they are available.
add_subdirectory(plugins-nextgen)
add_subdirectory(tools)
add_subdirectory(docs)

# Build target agnostic offloading library.
add_subdirectory(libomptarget)

add_subdirectory(liboffload)

# Add tests.
if(OFFLOAD_INCLUDE_TESTS)
  add_subdirectory(test)
  add_subdirectory(unittests)
endif()

add_subdirectory(utils)
