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

  function(runtime_register_component name)
    set_property(GLOBAL APPEND PROPERTY SUB_COMPONENTS ${name})
  endfunction()

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
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${LLVM_TOOLS_BINARY_DIR})

  foreach(entry ${runtimes})
    get_filename_component(projName ${entry} NAME)

    # TODO: Clean this up as part of an interface standardization
    string(REPLACE "-" "_" canon_name ${projName})
    string(TOUPPER ${canon_name} canon_name)
    # The subdirectories need to treat this as standalone builds
    set(${canon_name}_STANDALONE_BUILD On)

    # Setting a variable to let sub-projects detect which other projects
    # will be included under here.
    set(HAVE_${canon_name} On)

    # Between each sub-project we want to cache and clear the LIT properties
    set_property(GLOBAL PROPERTY LLVM_LIT_TESTSUITES)
    set_property(GLOBAL PROPERTY LLVM_LIT_PARAMS)
    set_property(GLOBAL PROPERTY LLVM_LIT_DEPENDS)
    set_property(GLOBAL PROPERTY LLVM_LIT_EXTRA_ARGS)

    add_subdirectory(${projName})

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
  endif()

  get_property(SUB_COMPONENTS GLOBAL PROPERTY SUB_COMPONENTS)
  if(SUB_COMPONENTS)
    list(REMOVE_DUPLICATES SUB_COMPONENTS)
    foreach(component ${SUB_COMPONENTS})
      if(NOT TARGET ${component})
        message(SEND_ERROR "Missing target for runtime component ${component}!")
        continue()
      endif()
      if(LLVM_INCLUDE_TESTS AND NOT TARGET check-${component})
        message(SEND_ERROR "Missing check target for runtime component ${component}!")
        continue()
      endif()

      if(TARGET install-${component})
        list(APPEND SUB_INSTALL_TARGETS install-${component})
      endif()
    endforeach()

    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/Components.cmake.in
      ${LLVM_BINARY_DIR}/runtimes/Components.cmake)
  endif()

else() # if this is included from LLVM's CMake
  include(${LLVM_BINARY_DIR}/runtimes/Components.cmake OPTIONAL)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${LLVM_BINARY_DIR}/runtimes/Components.cmake)
  include(LLVMExternalProjectUtils)

  # If compiler-rt is present we need to build the builtin libraries first. This
  # is required because the other runtimes need the builtin libraries present
  # before the just-built compiler can pass the configuration tests.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/compiler-rt)
    llvm_ExternalProject_Add(builtins
                             ${CMAKE_CURRENT_SOURCE_DIR}/compiler-rt/lib/builtins
                             CMAKE_ARGS -DLLVM_LIBRARY_OUTPUT_INTDIR=${LLVM_LIBRARY_DIR}
                                        -DLLVM_RUNTIME_OUTPUT_INTDIR=${LLVM_TOOLS_BINARY_DIR}
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

    string(FIND ${projName} "lib" LIB_IDX)
    if(LIB_IDX EQUAL 0)
      string(SUBSTRING ${projName} 3 -1 projName)
    endif()
    list(APPEND runtime_names ${projName})
  endforeach()

  if(runtimes)

    foreach(runtime_name ${runtime_names})
      list(APPEND extra_targets
        ${runtime_name}
        install-${runtime_name}
        check-${runtime_name})
    endforeach()

    if(LLVM_INCLUDE_TESTS)
      set(test_targets runtimes-test-depends check-runtimes)
      foreach(component ${SUB_COMPONENTS})
        list(APPEND SUB_COMPONENT_CHECK_TARGETS check-${component})
      endforeach()
    endif()

    # Create a runtimes target that uses this file as its top-level CMake file.
    # The runtimes target is a configuration of all the runtime libraries
    # together in a single CMake invocaiton.
    llvm_ExternalProject_Add(runtimes
                             ${CMAKE_CURRENT_SOURCE_DIR}
                             DEPENDS ${deps}
                             # Builtins were built separately above
                             CMAKE_ARGS -DCOMPILER_RT_BUILD_BUILTINS=Off
                                        -DLLVM_INCLUDE_TESTS=${LLVM_INCLUDE_TESTS}
                             PASSTHROUGH_PREFIXES ${prefixes}
                             EXTRA_TARGETS ${extra_targets}
                                            ${test_targets}
                                            ${SUB_COMPONENTS}
                                            ${SUB_COMPONENT_CHECK_TARGETS}
                                            ${SUB_INSTALL_TARGETS}
                             USE_TOOLCHAIN)
    if(LLVM_INCLUDE_TESTS)
      set_property(GLOBAL APPEND PROPERTY LLVM_LIT_DEPENDS runtimes-test-depends)
    endif()
  endif()
endif()
