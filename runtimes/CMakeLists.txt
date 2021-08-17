# This file handles building LLVM runtime sub-projects.
cmake_minimum_required(VERSION 3.13.4)
project(Runtimes C CXX ASM)

set(LLVM_ALL_RUNTIMES "compiler-rt;libc;libcxx;libcxxabi;libunwind;openmp")
set(LLVM_ENABLE_RUNTIMES "" CACHE STRING
  "Semicolon-separated list of runtimes to build (${LLVM_ALL_RUNTIMES}), or \"all\".")
if(LLVM_ENABLE_RUNTIMES STREQUAL "all" )
  set(LLVM_ENABLE_RUNTIMES ${LLVM_ALL_RUNTIMES})
endif()

foreach(proj ${LLVM_ENABLE_RUNTIMES})
  set(proj_dir "${CMAKE_CURRENT_SOURCE_DIR}/../${proj}")
  if(IS_DIRECTORY ${proj_dir} AND EXISTS ${proj_dir}/CMakeLists.txt)
    list(APPEND runtimes ${proj_dir})
  else()
    message(FATAL_ERROR "LLVM_ENABLE_RUNTIMES requests ${proj} but directory not found: ${proj_dir}")
  endif()
  string(TOUPPER "${proj}" canon_name)
  STRING(REGEX REPLACE "-" "_" canon_name ${canon_name})
  set(LLVM_EXTERNAL_${canon_name}_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../${proj}")
endforeach()

function(runtime_register_component name)
  set_property(GLOBAL APPEND PROPERTY SUB_COMPONENTS ${name})
endfunction()

find_package(LLVM PATHS "${LLVM_BINARY_DIR}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
find_package(Clang PATHS "${LLVM_BINARY_DIR}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

set(LLVM_COMMON_CMAKE_UTILS "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

# Add path for custom and the LLVM build's modules to the CMake module path.
list(INSERT CMAKE_MODULE_PATH 0
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  "${LLVM_COMMON_CMAKE_UTILS}"
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
  "${CMAKE_CURRENT_SOURCE_DIR}/../llvm/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/../llvm/cmake/modules"
)

set(LLVM_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third-party")

function(get_compiler_rt_path path)
  foreach(entry ${runtimes})
    get_filename_component(projName ${entry} NAME)
    if("${projName}" MATCHES "compiler-rt")
      set(${path} ${entry} PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

# Some of the runtimes will conditionally use the compiler-rt sanitizers
# to make this work smoothly we ensure that compiler-rt is added first in
# the list of sub-projects. This allows other sub-projects to have checks
# like `if(TARGET asan)` to enable building with asan.
get_compiler_rt_path(compiler_rt_path)
if(compiler_rt_path)
  list(REMOVE_ITEM runtimes ${compiler_rt_path})
  if(NOT DEFINED LLVM_BUILD_COMPILER_RT OR LLVM_BUILD_COMPILER_RT)
    list(INSERT runtimes 0 ${compiler_rt_path})
  endif()
endif()

# If building standalone by pointing CMake at this runtimes directory,
# LLVM_BINARY_DIR isn't set, find_package(LLVM) will fail and these
# intermediate paths are unset.
if (NOT LLVM_BINARY_DIR)
  set(LLVM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
endif()
if (NOT LLVM_FOUND)
  set(LLVM_TOOLS_BINARY_DIR ${LLVM_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib)
endif()

# Setting these variables will allow the sub-build to put their outputs into
# the library and bin directories of the top-level build.
set(LLVM_LIBRARY_OUTPUT_INTDIR ${LLVM_LIBRARY_DIR})
set(LLVM_RUNTIME_OUTPUT_INTDIR ${LLVM_TOOLS_BINARY_DIR})

# This variable makes sure that e.g. llvm-lit is found.
set(LLVM_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../llvm)
set(LLVM_CMAKE_DIR ${LLVM_MAIN_SRC_DIR}/cmake/modules)

# This variable is used by individual runtimes to locate LLVM files.
set(LLVM_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../llvm)

include(CheckLibraryExists)
include(LLVMCheckCompilerLinkerFlag)
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)


check_c_compiler_flag("" LLVM_RUNTIMES_LINKING_WORKS)
if (NOT LLVM_RUNTIMES_LINKING_WORKS)
  # The compiler driver may be implicitly trying to link against libunwind,
  # which might not work if libunwind doesn't exist yet. Try to check if
  # --unwindlib=none is supported, and use that if possible.
  # Don't add this if not necessary to fix linking, as it can break using
  # e.g. ASAN/TSAN.
  llvm_check_compiler_linker_flag(C "--unwindlib=none" LLVM_RUNTIMES_SUPPORT_UNWINDLIB_NONE_FLAG)
  if (LLVM_RUNTIMES_SUPPORT_UNWINDLIB_NONE_FLAG)
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} --unwindlib=none")
  endif()
endif()

# Disable use of the installed C++ standard library when building runtimes.
# Check for -nostdlib++ first; if there's no C++ standard library yet,
# all check_cxx_compiler_flag commands will fail until we add -nostdlib++
# (or -nodefaultlibs).
llvm_check_compiler_linker_flag(C "-nostdlib++" LLVM_RUNTIMES_SUPPORT_NOSTDLIBXX_FLAG)
if (LLVM_RUNTIMES_SUPPORT_NOSTDLIBXX_FLAG)
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -nostdlib++")
endif()
check_cxx_compiler_flag(-nostdinc++ LLVM_RUNTIMES_SUPPORT_NOSTDINCXX_FLAG)
if (LLVM_RUNTIMES_SUPPORT_NOSTDINCXX_FLAG)
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -nostdinc++")
endif()

# Avoid checking whether the compiler is working.
set(LLVM_COMPILER_CHECKED ON)

# Handle common options used by all runtimes.
include(AddLLVM)
include(HandleLLVMOptions)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Host triple is used by tests to check if they are running natively.
include(GetHostTriple)
get_host_triple(LLVM_HOST_TRIPLE)
set(LLVM_DEFAULT_TARGET_TRIPLE "${LLVM_HOST_TRIPLE}" CACHE STRING
  "Default target for which the runtimes will be built.")

option(LLVM_INCLUDE_TESTS "Generate build targets for the runtimes unit tests." ON)
option(LLVM_INCLUDE_DOCS "Generate build targets for the runtimes documentation." ON)
option(LLVM_ENABLE_SPHINX "Use Sphinx to generate the runtimes documentation." OFF)

# Use libtool instead of ar if you are both on an Apple host, and targeting Apple.
if(CMAKE_HOST_APPLE AND APPLE)
  include(UseLibtool)
endif()

# This can be used to detect whether we're in the runtimes build.
set(LLVM_RUNTIMES_BUILD ON)

foreach(entry ${runtimes})
  get_filename_component(projName ${entry} NAME)

  # TODO: Clean this up as part of an interface standardization
  string(REPLACE "-" "_" canon_name ${projName})
  string(TOUPPER ${canon_name} canon_name)

  # TODO: compiler-rt has to use standalone build for now. We tried to remove
  # this in D57992 but this broke the build because compiler-rt assumes that
  # LLVM and Clang are configured in the same build to set up dependencies. We
  # should clean up the compiler-rt build and remove this eventually.
  if ("${canon_name}" STREQUAL "COMPILER_RT")
    set(${canon_name}_STANDALONE_BUILD ON)
  endif()

  if(LLVM_RUNTIMES_LIBDIR_SUBDIR)
    set(${canon_name}_LIBDIR_SUBDIR "${LLVM_RUNTIMES_LIBDIR_SUBDIR}" CACHE STRING "" FORCE)
  endif()

  # Setting a variable to let sub-projects detect which other projects
  # will be included under here.
  set(HAVE_${canon_name} ON)
endforeach()

if(LLVM_INCLUDE_TESTS)
  set(LIT_ARGS_DEFAULT "-sv --show-xfail --show-unsupported")
  if (MSVC OR XCODE)
    set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
  endif()
  set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")
endif()

# llvm-libgcc incorporates both compiler-rt and libunwind as subprojects with very
# specific flags, which causes clashes when they're independently built too.
if("llvm-libgcc" IN_LIST runtimes)
  if("compiler-rt" IN_LIST runtimes OR "compiler-rt" IN_LIST LLVM_ENABLE_PROJECTS)
    message(FATAL_ERROR
      "Attempting to build both compiler-rt and llvm-libgcc will cause irreconcilable "
      "target clashes. Please choose one or the other, but not both.")
  endif()

  if("libunwind" IN_LIST runtimes)
    message(
      FATAL_ERROR
      "Attempting to build both libunwind and llvm-libgcc will cause irreconcilable "
      "target clashes. Please choose one or the other, but not both.")
  endif()
endif()

# We do this in two loops so that HAVE_* is set for each runtime before the
# other runtimes are added.
foreach(entry ${runtimes})
  get_filename_component(projName ${entry} NAME)

  # Between each sub-project we want to cache and clear the LIT properties
  set_property(GLOBAL PROPERTY LLVM_LIT_TESTSUITES)
  set_property(GLOBAL PROPERTY LLVM_LIT_PARAMS)
  set_property(GLOBAL PROPERTY LLVM_LIT_DEPENDS)
  set_property(GLOBAL PROPERTY LLVM_LIT_EXTRA_ARGS)

  add_subdirectory(${entry} ${projName})

  get_property(LLVM_LIT_TESTSUITES GLOBAL PROPERTY LLVM_LIT_TESTSUITES)
  get_property(LLVM_LIT_PARAMS GLOBAL PROPERTY LLVM_LIT_PARAMS)
  get_property(LLVM_LIT_DEPENDS GLOBAL PROPERTY LLVM_LIT_DEPENDS)
  get_property(LLVM_LIT_EXTRA_ARGS GLOBAL PROPERTY LLVM_LIT_EXTRA_ARGS)

  list(APPEND RUNTIMES_LIT_TESTSUITES ${LLVM_LIT_TESTSUITES})
  list(APPEND RUNTIMES_LIT_PARAMS ${LLVM_LIT_PARAMS})
  list(APPEND RUNTIMES_LIT_DEPENDS ${LLVM_LIT_DEPENDS})
  list(APPEND RUNTIMES_LIT_EXTRA_ARGS ${LLVM_LIT_EXTRA_ARGS})
endforeach()

if(LLVM_INCLUDE_TESTS)
  # Add a global check rule now that all subdirectories have been traversed
  # and we know the total set of lit testsuites.
  add_lit_target(check-runtimes
    "Running all regression tests"
    ${RUNTIMES_LIT_TESTSUITES}
    PARAMS ${RUNTIMES_LIT_PARAMS}
    DEPENDS ${RUNTIMES_LIT_DEPENDS}
    ARGS ${RUNTIMES_LIT_EXTRA_ARGS}
    )
  add_custom_target(runtimes-test-depends DEPENDS ${RUNTIMES_LIT_DEPENDS})

  if (NOT HAVE_LLVM_LIT)
    # If built by manually invoking cmake on this directory, we don't have
    # llvm-lit. If invoked via llvm/runtimes, the toplevel llvm cmake
    # invocation already generated the llvm-lit script.
    add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/llvm-lit
                     ${CMAKE_CURRENT_BINARY_DIR}/llvm-lit)
  endif()
endif()

get_property(SUB_COMPONENTS GLOBAL PROPERTY SUB_COMPONENTS)
if(SUB_COMPONENTS)
  list(REMOVE_DUPLICATES SUB_COMPONENTS)
  foreach(component ${SUB_COMPONENTS})
    if(NOT TARGET ${component})
      message(SEND_ERROR "Missing target for runtime component ${component}!")
      continue()
    endif()

    if(TARGET check-${component})
      list(APPEND SUB_CHECK_TARGETS check-${component})
    endif()

    if(TARGET install-${component})
      list(APPEND SUB_INSTALL_TARGETS install-${component})
    endif()
    if(TARGET install-${component}-stripped)
      list(APPEND SUB_INSTALL_TARGETS install-${component}-stripped)
    endif()
  endforeach()

  if(LLVM_RUNTIMES_TARGET)
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/Components.cmake.in
      ${LLVM_BINARY_DIR}/runtimes/${LLVM_RUNTIMES_TARGET}/Components.cmake)
  else()
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/Components.cmake.in
      ${LLVM_BINARY_DIR}/runtimes/Components.cmake)
  endif()
endif()
