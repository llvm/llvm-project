# This file handles building LLVM runtime sub-projects.

# Runtimes are different from tools or other drop-in projects because runtimes
# should be built with the LLVM toolchain from the build directory. This file is
# a first step to formalizing runtime build interfaces.

# In the current state this file only works with compiler-rt, other runtimes
# will work as the runtime build interface standardizes.

# Find all subdirectories containing CMake projects
file(GLOB entries *)
foreach(entry ${entries})
  if(IS_DIRECTORY ${entry} AND EXISTS ${entry}/CMakeLists.txt)
    list(APPEND runtimes ${entry})
  endif()
endforeach()

# If this file is acting as a top-level CMake invocation, this code path is
# triggered by the external project call for the runtimes target below.
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})

  cmake_minimum_required(VERSION 3.4.3)

  # Add the root project's CMake modules, and the LLVM build's modules to the
  # CMake module path.
  list(INSERT CMAKE_MODULE_PATH 0
    "${CMAKE_CURRENT_SOURCE_DIR}/../cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules"
    "${LLVM_BINARY_DIR}/lib/cmake/llvm"
  )

  # LLVMConfig.cmake contains a bunch of CMake variables from the LLVM build.
  # This file is installed as part of LLVM distributions, so this can be used
  # either from a build directory or an installed LLVM.
  include(LLVMConfig)

  # Setting these variables will allow the sub-build to put their outputs into
  # the library and bin directories of the top-level build.
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${LLVM_LIBRARY_DIR})
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${LLVM_BINARY_DIR})

  foreach(entry ${runtimes})
    get_filename_component(projName ${entry} NAME)

    # TODO: Clean this up as part of an interface standardization
    string(REPLACE "-" "_" canon_name ${projName})
    string(TOUPPER ${canon_name} canon_name)
    # The subdirectories need to treat this as standalone builds
    set(${canon_name}_STANDALONE_BUILD On)

    add_subdirectory(${projName})
  endforeach()

else() # if this is included from LLVM's CMake
  include(LLVMExternalProjectUtils)

  # If compiler-rt is present we need to build the builtin libraries first. This
  # is required because the other runtimes need the builtin libraries present
  # before the just-built compiler can pass the configuration tests.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/compiler-rt)
    llvm_ExternalProject_Add(builtins
                             ${CMAKE_CURRENT_SOURCE_DIR}/compiler-rt/lib/builtins
                             PASSTHROUGH_PREFIXES COMPILER_RT
                             USE_TOOLCHAIN)
    set(deps builtins)
  endif()

  # We create a list the names of all the runtime projects in all uppercase and
  # with dashes turned to underscores. This gives us the CMake variable prefixes
  # for all variables that will apply to runtimes.
  foreach(entry ${runtimes})
    get_filename_component(projName ${entry} NAME)
    string(REPLACE "-" "_" canon_name ${projName})
    string(TOUPPER ${canon_name} canon_name)
    list(APPEND prefixes ${canon_name})
  endforeach()

  if(runtimes)
    # Create a runtimes target that uses this file as its top-level CMake file.
    # The runtimes target is a configuration of all the runtime libraries
    # together in a single CMake invocaiton.
    llvm_ExternalProject_Add(runtimes
                             ${CMAKE_CURRENT_SOURCE_DIR}
                             DEPENDS ${deps} llvm-config
                             # Builtins were built separately above
                             CMAKE_ARGS -DCOMPILER_RT_BUILD_BUILTINS=Off
                             PASSTHROUGH_PREFIXES ${prefixes}
                             USE_TOOLCHAIN)
    if(LLVM_INCLUDE_TESTS)
      set_property(GLOBAL APPEND PROPERTY LLVM_ADDITIONAL_TEST_DEPENDS runtimes-test-depends)
      set_property(GLOBAL APPEND PROPERTY LLVM_ADDITIONAL_TEST_TARGETS check-runtimes)
    endif()
  endif()
endif()
