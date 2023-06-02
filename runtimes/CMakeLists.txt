# This file handles building LLVM runtime sub-projects.
cmake_minimum_required(VERSION 3.20.0)

# Add path for custom and the LLVM build's modules to the CMake module path.
set(LLVM_COMMON_CMAKE_UTILS "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(${LLVM_COMMON_CMAKE_UTILS}/Modules/CMakePolicy.cmake
  NO_POLICY_SCOPE)

project(Runtimes C CXX ASM)

list(INSERT CMAKE_MODULE_PATH 0
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  "${LLVM_COMMON_CMAKE_UTILS}"
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
  "${CMAKE_CURRENT_SOURCE_DIR}/../llvm/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/../llvm/cmake/modules"
)

# We order libraries to mirror roughly how they are layered, except that compiler-rt can depend
# on libc++, so we put it after.
set(LLVM_DEFAULT_RUNTIMES "libc;libunwind;libcxxabi;pstl;libcxx;compiler-rt;openmp")
set(LLVM_SUPPORTED_RUNTIMES "${LLVM_DEFAULT_RUNTIMES};llvm-libgcc")
set(LLVM_ENABLE_RUNTIMES "" CACHE STRING
  "Semicolon-separated list of runtimes to build, or \"all\" (${LLVM_DEFAULT_RUNTIMES}). Supported runtimes are ${LLVM_SUPPORTED_RUNTIMES}.")
if(LLVM_ENABLE_RUNTIMES STREQUAL "all" )
  set(LLVM_ENABLE_RUNTIMES ${LLVM_DEFAULT_RUNTIMES})
endif()
include(SortSubset)
sort_subset("${LLVM_SUPPORTED_RUNTIMES}" "${LLVM_ENABLE_RUNTIMES}" LLVM_ENABLE_RUNTIMES)

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

set(LLVM_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third-party")

# If building standalone by pointing CMake at this runtimes directory,
# LLVM_BINARY_DIR isn't set, find_package(LLVM) will fail and these
# intermediate paths are unset.
if (NOT LLVM_BINARY_DIR)
  set(LLVM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
endif()
if (NOT LLVM_FOUND)
  set(LLVM_TOOLS_BINARY_DIR ${LLVM_BINARY_DIR}/bin)
  set(LLVM_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
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

# CMake omits default compiler include paths, but in runtimes build, we use
# -nostdinc and -nostdinc++ and control include paths manually so this behavior
# is undesirable. Filtering CMAKE_{LANG}_IMPLICIT_INCLUDE_DIRECTORIES to remove
# paths that are inside the build directory disables this behavior.
#
# See https://gitlab.kitware.com/cmake/cmake/-/issues/19227 for further details.

function(filter_prefixed list prefix outvar)
  foreach(str ${list})
    string(FIND "${str}" "${prefix}" out)
    if(NOT "${out}" EQUAL 0)
      list(APPEND result ${str})
    endif()
  endforeach()
  set(${outvar} ${result} PARENT_SCOPE)
endfunction()

filter_prefixed("${CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES}" ${LLVM_BINARY_DIR} CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES)
filter_prefixed("${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}" ${LLVM_BINARY_DIR} CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
filter_prefixed("${CMAKE_ASM_IMPLICIT_INCLUDE_DIRECTORIES}" ${LLVM_BINARY_DIR} CMAKE_ASM_IMPLICIT_INCLUDE_DIRECTORIES)

# The compiler driver may be implicitly trying to link against libunwind,
# which might not work if libunwind doesn't exist yet. Try to check if
# --unwindlib=none is supported, and use that if possible.
#
# TODO: Note that this is problematic when LLVM_USE_SANITIZER is used
# because some sanitizers require the unwinder and so the combination of
# -fsanitize=... --unwindlib=none will always result in a linking error.
# Currently, we counteract this issue by adding -fno-sanitize=all flag in
# the project specific code within */cmake/config-ix.cmake files but that's
# brittle. We should ideally move this to runtimes/CMakeLists.txt.
llvm_check_compiler_linker_flag(C "--unwindlib=none" CXX_SUPPORTS_UNWINDLIB_EQ_NONE_FLAG)
if (CXX_SUPPORTS_UNWINDLIB_EQ_NONE_FLAG)
  set(ORIG_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} --unwindlib=none")
  # TODO: When we can require CMake 3.14, we should use
  # CMAKE_REQUIRED_LINK_OPTIONS here. Until then, we need a workaround:
  # When using CMAKE_REQUIRED_FLAGS, this option gets added both to
  # compilation and linking commands. That causes warnings in the
  # compilation commands during cmake tests. This is normally benign, but
  # when testing whether -Werror works, that test fails (due to the
  # preexisting warning).
  #
  # Therefore, before we can use CMAKE_REQUIRED_LINK_OPTIONS, check if we
  # can use --start-no-unused-arguments to silence the warnings about
  # --unwindlib=none during compilation.
  #
  # We must first add --unwindlib=none to CMAKE_REQUIRED_FLAGS above, to
  # allow this subsequent test to succeed, then rewrite CMAKE_REQUIRED_FLAGS
  # below.
  check_c_compiler_flag("--start-no-unused-arguments" C_SUPPORTS_START_NO_UNUSED_ARGUMENTS)
  if (C_SUPPORTS_START_NO_UNUSED_ARGUMENTS)
    set(CMAKE_REQUIRED_FLAGS "${ORIG_CMAKE_REQUIRED_FLAGS} --start-no-unused-arguments --unwindlib=none --end-no-unused-arguments")
  endif()
endif()

# Disable use of the installed C++ standard library when building runtimes.
# Check for -nostdlib++ first; if there's no C++ standard library yet,
# all check_cxx_compiler_flag commands will fail until we add -nostdlib++
# (or -nodefaultlibs).
llvm_check_compiler_linker_flag(CXX "-nostdlib++" CXX_SUPPORTS_NOSTDLIBXX_FLAG)
if (CXX_SUPPORTS_NOSTDLIBXX_FLAG)
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -nostdlib++")
endif()
check_cxx_compiler_flag(-nostdinc++ CXX_SUPPORTS_NOSTDINCXX_FLAG)
if (CXX_SUPPORTS_NOSTDINCXX_FLAG)
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

include(SetTargetTriple)
set_llvm_target_triple()

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

  umbrella_lit_testsuite_begin(check-runtimes)
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

  add_subdirectory(${entry} ${projName})
endforeach()

if(LLVM_INCLUDE_TESTS)
  # LLVM_RUNTIMES_LIT_DEPENDS is populated when lit tests are added between
  # umbrella_list_testsuite begin and end. The bootstrap runtimes builds
  # currently assumes this target exists.
  get_property(LLVM_RUNTIMES_LIT_DEPENDS GLOBAL PROPERTY LLVM_RUNTIMES_LIT_DEPENDS)
  add_custom_target(runtimes-test-depends)
  if(LLVM_RUNTIMES_LIT_DEPENDS)
    # add_dependencies complains if called with no dependencies
    add_dependencies(runtimes-test-depends ${LLVM_RUNTIMES_LIT_DEPENDS})
  endif()
  # Add a global check rule now that all subdirectories have been traversed
  # and we know the total set of lit testsuites.
  umbrella_lit_testsuite_end(check-runtimes)

  if (NOT HAVE_LLVM_LIT)
    # If built by manually invoking cmake on this directory, we don't have
    # llvm-lit. If invoked via llvm/runtimes, the toplevel llvm cmake
    # invocation already generated the llvm-lit script.
    add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/llvm-lit
                     ${CMAKE_CURRENT_BINARY_DIR}/llvm-lit)
  endif()

  get_property(LLVM_RUNTIMES_LIT_TESTSUITES GLOBAL PROPERTY LLVM_RUNTIMES_LIT_TESTSUITES)
  string(REPLACE ";" "\n" LLVM_RUNTIMES_LIT_TESTSUITES "${LLVM_RUNTIMES_LIT_TESTSUITES}")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit.tests ${LLVM_RUNTIMES_LIT_TESTSUITES})
else()
  # Create empty files so the parent build can use these unconditionally.
  file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/lit.tests)
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
