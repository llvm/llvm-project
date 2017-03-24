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

  # Some of the runtimes will conditionally use the compiler-rt sanitizers
  # to make this work smoothly we ensure that compiler-rt is added first in
  # the list of sub-projects. This allows other sub-projects to have checks
  # like `if(TARGET asan)` to enable building with asan.
  foreach(entry ${runtimes})
    if("${entry}" MATCHES "compiler-rt")
      set(compiler_rt_path ${entry})
      break()
    endif()
  endforeach()
  if(compiler_rt_path)
    list(REMOVE_ITEM runtimes ${compiler_rt_path})
    list(INSERT runtimes 0 ${compiler_rt_path})
  endif()

  # LLVMConfig.cmake contains a bunch of CMake variables from the LLVM build.
  # This file is installed as part of LLVM distributions, so this can be used
  # either from a build directory or an installed LLVM.
  include(LLVMConfig)

  # Setting these variables will allow the sub-build to put their outputs into
  # the library and bin directories of the top-level build.
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${LLVM_LIBRARY_DIR})
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${LLVM_TOOLS_BINARY_DIR})

  # This variable makes sure that e.g. llvm-lit is found.
  set(LLVM_MAIN_SRC_DIR ${LLVM_BUILD_MAIN_SRC_DIR})

  if(APPLE)
   set(LLVM_ENABLE_LIBCXX ON CACHE BOOL "")
  endif()

  # Handle common options used by all runtimes.
  include(AddLLVM)
  include(HandleLLVMOptions)

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
  endforeach()

  # We do this in two loops so that HAVE_* is set for each runtime before the
  # other runtimes are added.
  foreach(entry ${runtimes})
    get_filename_component(projName ${entry} NAME)
    
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

  if(NOT LLVM_BUILD_RUNTIMES)
    set(EXTRA_ARGS EXCLUDE_FROM_ALL)
  endif()

  # If compiler-rt is present we need to build the builtin libraries first. This
  # is required because the other runtimes need the builtin libraries present
  # before the just-built compiler can pass the configuration tests.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/compiler-rt)
    if(NOT LLVM_BUILTIN_TARGETS)
      llvm_ExternalProject_Add(builtins
                               ${CMAKE_CURRENT_SOURCE_DIR}/compiler-rt/lib/builtins
                               CMAKE_ARGS -DLLVM_LIBRARY_OUTPUT_INTDIR=${LLVM_LIBRARY_DIR}
                                          -DLLVM_RUNTIME_OUTPUT_INTDIR=${LLVM_TOOLS_BINARY_DIR}
                                          -DCMAKE_C_COMPILER_TARGET=${TARGET_TRIPLE}
                                          -DCMAKE_ASM_COMPILER_TARGET=${TARGET_TRIPLE}
                               PASSTHROUGH_PREFIXES COMPILER_RT
                               USE_TOOLCHAIN
                               ${EXTRA_ARGS})
    else()
      get_cmake_property(variableNames VARIABLES)
      add_custom_target(builtins)
      foreach(target ${LLVM_BUILTIN_TARGETS})
        string(REPLACE "-" ";" builtin_target_list ${target})
        foreach(item ${builtin_target_list})
          string(TOLOWER "${item}" item_lower)
          if(item_lower MATCHES "darwin")
            message(FATAL_ERROR "LLVM_BUILTIN_TARGETS isn't implemented for Darwin platform!")
          endif()
        endforeach()

        foreach(variableName ${variableNames})
          if(variableName MATCHES "^BUILTINS_${target}")
            string(REPLACE "BUILTINS_${target}_" "" new_name ${variableName})
            list(APPEND ${target}_extra_args "-D${new_name}=${${variableName}}")
          endif()
        endforeach()
        llvm_ExternalProject_Add(builtins-${target}
                               ${CMAKE_CURRENT_SOURCE_DIR}/compiler-rt/lib/builtins
                               CMAKE_ARGS -DLLVM_LIBRARY_OUTPUT_INTDIR=${LLVM_LIBRARY_DIR}
                                          -DLLVM_RUNTIME_OUTPUT_INTDIR=${LLVM_TOOLS_BINARY_DIR}
                                          -DCMAKE_C_COMPILER_TARGET=${target}
                                          -DCMAKE_ASM_COMPILER_TARGET=${target}
                                          -DCMAKE_C_COMPILER_WORKS=On
                                          -DCMAKE_ASM_COMPILER_WORKS=On
                                          -DCOMPILER_RT_DEFAULT_TARGET_ONLY=On
                                          ${${target}_extra_args}
                               TOOLCHAIN_TOOLS clang lld llvm-ar llvm-ranlib
                               PASSTHROUGH_PREFIXES COMPILER_RT
                               USE_TOOLCHAIN
                               ${EXTRA_ARGS})
        add_dependencies(builtins builtins-${target})
      endforeach()
    endif()
    set(deps builtins)
    # We don't need to depend on the builtins if we're building instrumented
    # because the next stage will use the same compiler used to build this stage.
    if(NOT LLVM_BUILD_INSTRUMENTED AND CLANG_ENABLE_BOOTSTRAP)
      add_dependencies(clang-bootstrap-deps builtins)
    endif()
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
                             USE_TOOLCHAIN
                             ${EXTRA_ARGS})
    
    # TODO: This is a hack needed because the libcxx headers are copied into the
    # build directory during configuration. Without that step the clang in the
    # build directory cannot find the C++ headers in certain configurations.
    # I need to build a mechanism for runtime projects to provide CMake code
    # that executes at LLVM configuration time to handle this case.
    if(NOT LLVM_BUILD_INSTRUMENTED AND CLANG_ENABLE_BOOTSTRAP)
      add_dependencies(clang-bootstrap-deps runtimes-configure)
    endif()

    if(LLVM_INCLUDE_TESTS)
      set_property(GLOBAL APPEND PROPERTY LLVM_ADDITIONAL_TEST_DEPENDS runtimes-test-depends)
      set_property(GLOBAL APPEND PROPERTY LLVM_ADDITIONAL_TEST_TARGETS check-runtimes)
    endif()
  endif()
endif()
