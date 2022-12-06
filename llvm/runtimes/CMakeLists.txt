# TODO: This file assumes the Clang toolchain so it'd be better if it lived in
# Clang, except there already is clang/runtime directory which contains
# similar although simpler functionality. We should figure out how to merge
# the two files.

set(COMMON_CMAKE_ARGS "-DHAVE_LLVM_LIT=ON")
foreach(proj ${LLVM_ENABLE_RUNTIMES})
  set(proj_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../${proj}")
  if(IS_DIRECTORY ${proj_dir} AND EXISTS ${proj_dir}/CMakeLists.txt)
    list(APPEND runtimes ${proj_dir})
  else()
    message(FATAL_ERROR "LLVM_ENABLE_RUNTIMES requests ${proj} but directory not found: ${proj_dir}")
  endif()
  string(TOUPPER "${proj}" canon_name)
  STRING(REGEX REPLACE "-" "_" canon_name ${canon_name})
  set(LLVM_EXTERNAL_${canon_name}_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../${proj}")
endforeach()

function(get_compiler_rt_path path)
  foreach(entry ${runtimes})
    get_filename_component(projName ${entry} NAME)
    if("${projName}" MATCHES "compiler-rt")
      set(${path} ${entry} PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

include(LLVMExternalProjectUtils)

if(NOT LLVM_BUILD_RUNTIMES)
  set(EXTRA_ARGS EXCLUDE_FROM_ALL)
endif()

function(check_apple_target triple builtin_or_runtime)
  set(error "\
compiler-rt for Darwin builds for all platforms and architectures using a \
single configuration. Specify only a single darwin triple (e.g. x86_64-apple-darwin) \
in your targets list (and not a triple for a specific platform such as macos). \
You can use variables such as COMPILER_RT_ENABLE_IOS and DARWIN_ios_ARCHS to \
control the specific platforms and architectures to build.")

  set(seen_property ${builtin_or_runtime}_darwin_triple_seen)
  string(REPLACE "-" ";" triple_components ${triple})
  foreach(component ${triple_components})
    string(TOLOWER "${component}" component_lower)
    if(component_lower MATCHES "^darwin")
      get_property(darwin_triple_seen GLOBAL PROPERTY ${seen_property})
      if(darwin_triple_seen)
        message(FATAL_ERROR "${error}")
      endif()
      set_property(GLOBAL PROPERTY ${seen_property} YES)
      if(NOT RUNTIMES_BUILD_ALLOW_DARWIN)
        message(FATAL_ERROR "\
${error} Set RUNTIMES_BUILD_ALLOW_DARWIN to allow a single darwin triple.")
      endif()
    elseif(component_lower MATCHES "^ios|^macos|^tvos|^watchos")
      message(FATAL_ERROR "${error}")
    endif()
  endforeach()
endfunction()

function(builtin_default_target compiler_rt_path)
  cmake_parse_arguments(ARG "" "" "DEPENDS" ${ARGN})

  set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR_default ON)
  # AIX should fold 32-bit & 64-bit arch libraries into a single archive.
  if (LLVM_TARGET_TRIPLE MATCHES "aix")
    set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR_default OFF)
  endif()

  llvm_ExternalProject_Add(builtins
                           ${compiler_rt_path}/lib/builtins
                           DEPENDS ${ARG_DEPENDS}
                           CMAKE_ARGS -DLLVM_LIBRARY_OUTPUT_INTDIR=${LLVM_LIBRARY_DIR}
                                      -DLLVM_RUNTIME_OUTPUT_INTDIR=${LLVM_TOOLS_BINARY_DIR}
                                      -DLLVM_DEFAULT_TARGET_TRIPLE=${LLVM_TARGET_TRIPLE}
                                      -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=${LLVM_ENABLE_PER_TARGET_RUNTIME_DIR_default}
                                      -DCMAKE_C_COMPILER_WORKS=ON
                                      -DCMAKE_ASM_COMPILER_WORKS=ON
                                      ${COMMON_CMAKE_ARGS}
                                      ${BUILTINS_CMAKE_ARGS}
                           PASSTHROUGH_PREFIXES COMPILER_RT
                                                DARWIN
                                                SANITIZER
                           USE_TOOLCHAIN
                           TARGET_TRIPLE ${LLVM_TARGET_TRIPLE}
                           ${EXTRA_ARGS})
endfunction()

function(builtin_register_target compiler_rt_path target)
  cmake_parse_arguments(ARG "" "" "DEPENDS" ${ARGN})

  check_apple_target(${target} builtin)

  get_cmake_property(variableNames VARIABLES)
  foreach(variableName ${variableNames})
    string(FIND "${variableName}" "BUILTINS_${target}" out)
    if("${out}" EQUAL 0)
      string(REPLACE "BUILTINS_${target}_" "" new_name ${variableName})
      string(REPLACE ";" "|" new_value "${${variableName}}")
      list(APPEND ${target}_extra_args "-D${new_name}=${new_value}")
    endif()
  endforeach()

  llvm_ExternalProject_Add(builtins-${target}
                           ${compiler_rt_path}/lib/builtins
                           DEPENDS ${ARG_DEPENDS}
                           CMAKE_ARGS -DLLVM_LIBRARY_OUTPUT_INTDIR=${LLVM_LIBRARY_DIR}
                                      -DLLVM_RUNTIME_OUTPUT_INTDIR=${LLVM_TOOLS_BINARY_DIR}
                                      -DLLVM_DEFAULT_TARGET_TRIPLE=${target}
                                      -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON
                                      -DCMAKE_C_COMPILER_WORKS=ON
                                      -DCMAKE_ASM_COMPILER_WORKS=ON
                                      -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON
                                      ${COMMON_CMAKE_ARGS}
                                      ${${target}_extra_args}
                           USE_TOOLCHAIN
                           TARGET_TRIPLE ${target}
                           ${EXTRA_ARGS})
endfunction()

# If compiler-rt is present we need to build the builtin libraries first. This
# is required because the other runtimes need the builtin libraries present
# before the just-built compiler can pass the configuration tests.
get_compiler_rt_path(compiler_rt_path)
if(compiler_rt_path)
  if(NOT LLVM_BUILTIN_TARGETS)
    builtin_default_target(${compiler_rt_path}
      DEPENDS clang-resource-headers)
  else()
    if("default" IN_LIST LLVM_BUILTIN_TARGETS)
      builtin_default_target(${compiler_rt_path}
        DEPENDS clang-resource-headers)
      list(REMOVE_ITEM LLVM_BUILTIN_TARGETS "default")
    else()
      add_custom_target(builtins)
      add_custom_target(install-builtins)
      add_custom_target(install-builtins-stripped)
    endif()

    foreach(target ${LLVM_BUILTIN_TARGETS})
      builtin_register_target(${compiler_rt_path} ${target}
        DEPENDS clang-resource-headers)

      add_dependencies(builtins builtins-${target})
      add_dependencies(install-builtins install-builtins-${target})
      add_dependencies(install-builtins-stripped install-builtins-${target}-stripped)
    endforeach()
  endif()
  set(deps builtins)
  # We don't need to depend on the builtins if we're building instrumented
  # because the next stage will use the same compiler used to build this stage.
  if(NOT LLVM_BUILD_INSTRUMENTED AND CLANG_ENABLE_BOOTSTRAP)
    add_dependencies(clang-bootstrap-deps builtins)
  endif()
endif()

# Create a list with the names of all the runtime projects in all uppercase and
# with dashes turned to underscores. This gives us the CMake variable `prefixes`
# for all variables that will apply to runtimes.
foreach(entry ${runtimes})
  get_filename_component(projName ${entry} NAME)
  if(projName STREQUAL "libc")
    # For now, we will use the name "llvmlibc" for the libc project as it is
    # not a full libc yet. Also, if we leave it as is, the "lib" prefix gets
    # stripped below and the targets endup having the name "c", "check-c" etc.
    set(projName "llvmlibc")
  endif()
  string(REPLACE "-" "_" canon_name ${projName})
  string(TOUPPER ${canon_name} canon_name)
  list(APPEND prefixes ${canon_name})
  if (${canon_name} STREQUAL "OPENMP")
    list(APPEND prefixes "LIBOMP" "LIBOMPTARGET")
  endif()
  # Many compiler-rt options start with SANITIZER_ and DARWIN_ rather than
  # COMPILER_RT_, so when compiler-rt is enabled, consider both.
  if(canon_name STREQUAL "COMPILER_RT")
    list(APPEND prefixes SANITIZER DARWIN)
  endif()
  if(canon_name STREQUAL "LLVMLIBC")
    list(APPEND prefixes "LLVM_LIBC")
  endif()

  string(FIND ${projName} "lib" LIB_IDX)
  if(LIB_IDX EQUAL 0)
    string(SUBSTRING ${projName} 3 -1 projName)
  endif()
  list(APPEND runtime_names ${projName})
endforeach()

function(runtime_default_target)
  cmake_parse_arguments(ARG "" "" "DEPENDS;PREFIXES" ${ARGN})

  include(${LLVM_BINARY_DIR}/runtimes/Components.cmake OPTIONAL)
  set(SUB_CHECK_TARGETS ${SUB_CHECK_TARGETS} PARENT_SCOPE)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${LLVM_BINARY_DIR}/runtimes/Components.cmake)

  foreach(runtime_name ${runtime_names})
    list(APPEND extra_targets
      ${runtime_name}
      install-${runtime_name}
      install-${runtime_name}-stripped)
    if(LLVM_INCLUDE_TESTS)
      list(APPEND test_targets check-${runtime_name})
    endif()
  endforeach()
  foreach(component ${LLVM_RUNTIME_DISTRIBUTION_COMPONENTS})
    if(NOT ${component} IN_LIST SUB_COMPONENTS)
      list(APPEND extra_targets install-${component} install-${component}-stripped)
    endif()
  endforeach()

  if(LLVM_INCLUDE_TESTS)
    set_property(GLOBAL APPEND PROPERTY LLVM_ALL_LIT_TESTSUITES "@${LLVM_BINARY_DIR}/runtimes/runtimes-bins/lit.tests")
    list(APPEND test_targets runtimes-test-depends check-runtimes)
  endif()

  set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR_default ON)
  # AIX should fold 32-bit & 64-bit arch libraries into a single archive.
  if (LLVM_TARGET_TRIPLE MATCHES "aix")
    set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR_default OFF)
  endif()

  llvm_ExternalProject_Add(runtimes
                           ${CMAKE_CURRENT_SOURCE_DIR}/../../runtimes
                           DEPENDS ${ARG_DEPENDS}
                           # Builtins were built separately above
                           CMAKE_ARGS -DCOMPILER_RT_BUILD_BUILTINS=Off
                                      -DLLVM_INCLUDE_TESTS=${LLVM_INCLUDE_TESTS}
                                      -DLLVM_DEFAULT_TARGET_TRIPLE=${LLVM_TARGET_TRIPLE}
                                      -DLLVM_ENABLE_PROJECTS_USED=${LLVM_ENABLE_PROJECTS_USED}
                                      -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=${LLVM_ENABLE_PER_TARGET_RUNTIME_DIR_default}
                                      -DLLVM_BUILD_TOOLS=${LLVM_BUILD_TOOLS}
                                      -DCMAKE_C_COMPILER_WORKS=ON
                                      -DCMAKE_CXX_COMPILER_WORKS=ON
                                      -DCMAKE_ASM_COMPILER_WORKS=ON
                                      ${COMMON_CMAKE_ARGS}
                                      ${RUNTIMES_CMAKE_ARGS}
                           PASSTHROUGH_PREFIXES LLVM_ENABLE_RUNTIMES
                                                LLVM_USE_LINKER
                                                ${ARG_PREFIXES}
                           EXTRA_TARGETS ${extra_targets}
                                         ${test_targets}
                                         ${SUB_COMPONENTS}
                                         ${SUB_CHECK_TARGETS}
                                         ${SUB_INSTALL_TARGETS}
                           USE_TOOLCHAIN
                           TARGET_TRIPLE ${LLVM_TARGET_TRIPLE}
                           ${EXTRA_ARGS})
endfunction()

# runtime_register_target(target)
#   Utility function to register external runtime target.
function(runtime_register_target name target)
  cmake_parse_arguments(ARG "" "" "DEPENDS;CMAKE_ARGS" ${ARGN})
  include(${LLVM_BINARY_DIR}/runtimes/${name}/Components.cmake OPTIONAL)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${LLVM_BINARY_DIR}/runtimes/${name}/Components.cmake)

  check_apple_target(${target} runtime)

  set(${name}_deps ${ARG_DEPENDS})
  if(NOT name STREQUAL target)
    list(APPEND ${name}_deps runtimes-${target})
  endif()

  foreach(runtime_name ${runtime_names})
    set(${runtime_name}-${name} ${runtime_name})
    set(install-${runtime_name}-${name} install-${runtime_name})
    set(install-${runtime_name}-${name}-stripped install-${runtime_name}-stripped)
    list(APPEND ${name}_extra_targets ${runtime_name}-${name} install-${runtime_name}-${name} install-${runtime_name}-${name}-stripped)
    if(LLVM_INCLUDE_TESTS)
      set(check-${runtime_name}-${name} check-${runtime_name} )
      list(APPEND ${name}_test_targets check-${runtime_name}-${name})
    endif()
  endforeach()

  foreach(target_name IN LISTS SUB_COMPONENTS SUB_INSTALL_TARGETS)
    set(${target_name}-${name} ${target_name})
    list(APPEND ${name}_extra_targets ${target_name}-${name})
  endforeach()

  foreach(component ${LLVM_RUNTIME_DISTRIBUTION_COMPONENTS})
    set(${component}-${name} ${component})
    set(install-${component}-${name} install-${component})
    set(install-${component}-${name}-stripped install-${component}-stripped)
    list(APPEND ${name}_extra_targets ${component}-${name} install-${component}-${name} install-${component}-${name}-stripped)
  endforeach()

  if(LLVM_INCLUDE_TESTS)
    set_property(GLOBAL APPEND PROPERTY LLVM_ALL_LIT_TESTSUITES "@${LLVM_BINARY_DIR}/runtimes/runtimes-${name}-bins/lit.tests")
    set(runtimes-test-depends-${name} runtimes-test-depends)
    set(check-runtimes-${name} check-runtimes)
    list(APPEND ${name}_test_targets runtimes-test-depends-${name} check-runtimes-${name})
    list(APPEND test_targets ${${name}_test_targets})
    foreach(target_name IN LISTS SUB_CHECK_TARGETS)
      set(${target_name}-${name} ${target_name})
      list(APPEND ${name}_test_targets ${target_name}-${name})
      list(APPEND test_targets ${target_name}-${name})
    endforeach()
    set(test_targets "${test_targets}" PARENT_SCOPE)
  endif()

  set(${name}_extra_args ${ARG_CMAKE_ARGS})
  get_cmake_property(variableNames VARIABLES)
  foreach(variableName ${variableNames})
    string(FIND "${variableName}" "RUNTIMES_${target}_" out)
    if("${out}" EQUAL 0)
      string(REPLACE "RUNTIMES_${target}_" "" new_name ${variableName})
      string(REPLACE ";" "|" new_value "${${variableName}}")
      list(APPEND ${name}_extra_args "-D${new_name}=${new_value}")
    endif()
  endforeach()
  if(NOT "${name}" STREQUAL "${target}")
    foreach(variableName ${variableNames})
      string(FIND "${variableName}" "RUNTIMES_${name}_" out)
      if("${out}" EQUAL 0)
        string(REPLACE "RUNTIMES_${name}_" "" new_name ${variableName})
        string(REPLACE ";" "|" new_value "${${variableName}}")
        list(APPEND ${name}_extra_args "-D${new_name}=${new_value}")
      endif()
    endforeach()
  endif()

  if(NOT RUNTIMES_${name}_LLVM_ENABLE_RUNTIMES AND NOT RUNTIMES_${target}_LLVM_ENABLE_RUNTIMES)
    string(REPLACE ";" "|" LLVM_ENABLE_RUNTIMES_PASSTHROUGH "${LLVM_ENABLE_RUNTIMES}")
    list(APPEND ${name}_extra_args -DLLVM_ENABLE_RUNTIMES=${LLVM_ENABLE_RUNTIMES_PASSTHROUGH})
  endif()

  if(NOT RUNTIMES_${name}_LLVM_USE_LINKER AND NOT RUNTIMES_${target}_LLVM_USE_LINKER)
    list(APPEND ${name}_extra_args -DLLVM_USE_LINKER=${LLVM_USE_LINKER})
  endif()

  llvm_ExternalProject_Add(runtimes-${name}
                           ${CMAKE_CURRENT_SOURCE_DIR}/../../runtimes
                           DEPENDS ${${name}_deps}
                           # Builtins were built separately above
                           CMAKE_ARGS -DCOMPILER_RT_BUILD_BUILTINS=Off
                                      -DLLVM_INCLUDE_TESTS=${LLVM_INCLUDE_TESTS}
                                      -DLLVM_DEFAULT_TARGET_TRIPLE=${target}
                                      -DLLVM_ENABLE_PROJECTS_USED=${LLVM_ENABLE_PROJECTS_USED}
                                      -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON
                                      -DCMAKE_C_COMPILER_WORKS=ON
                                      -DCMAKE_CXX_COMPILER_WORKS=ON
                                      -DCMAKE_ASM_COMPILER_WORKS=ON
                                      -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON
                                      -DLLVM_RUNTIMES_TARGET=${name}
                                      ${COMMON_CMAKE_ARGS}
                                      ${${name}_extra_args}
                           EXTRA_TARGETS ${${name}_extra_targets}
                                         ${${name}_test_targets}
                           USE_TOOLCHAIN
                           TARGET_TRIPLE ${target}
                           ${EXTRA_ARGS})
endfunction()

if(runtimes)
  # Create a runtimes target that uses this file as its top-level CMake file.
  # The runtimes target is a configuration of all the runtime libraries
  # together in a single CMake invocation.
  set(extra_deps "")
  if("openmp" IN_LIST LLVM_ENABLE_RUNTIMES)
    if(TARGET opt)
      list(APPEND extra_deps opt)
    endif()
    if(TARGET llvm-link)
      list(APPEND extra_deps llvm-link)
    endif()
  endif()
  if(NOT LLVM_RUNTIME_TARGETS)
    runtime_default_target(
      DEPENDS ${deps} ${extra_deps}
      PREFIXES ${prefixes})
    set(test_targets check-runtimes)
  else()
    if("default" IN_LIST LLVM_RUNTIME_TARGETS)
      runtime_default_target(
        DEPENDS ${deps} ${extra_deps}
        PREFIXES ${prefixes})
      list(REMOVE_ITEM LLVM_RUNTIME_TARGETS "default")
    else()
      add_custom_target(runtimes)
      add_custom_target(runtimes-configure)
      add_custom_target(install-runtimes)
      add_custom_target(install-runtimes-stripped)
      if(LLVM_INCLUDE_TESTS)
        add_custom_target(check-runtimes)
        add_custom_target(runtimes-test-depends)
        set(test_targets "")
      endif()
      foreach(runtime_name ${runtime_names})
        add_custom_target(${runtime_name})
        add_custom_target(install-${runtime_name})
        add_custom_target(install-${runtime_name}-stripped)
      endforeach()
      if(LLVM_RUNTIME_DISTRIBUTION_COMPONENTS)
        foreach(component ${LLVM_RUNTIME_DISTRIBUTION_COMPONENTS})
          add_custom_target(${component})
          add_custom_target(install-${component})
          add_custom_target(install-${component}-stripped)
        endforeach()
      endif()
    endif()

    foreach(name ${LLVM_RUNTIME_TARGETS})
      runtime_register_target(${name} ${name}
        DEPENDS ${deps})

      add_dependencies(runtimes runtimes-${name})
      add_dependencies(runtimes-configure runtimes-${name}-configure)
      add_dependencies(install-runtimes install-runtimes-${name})
      add_dependencies(install-runtimes-stripped install-runtimes-${name}-stripped)
      if(LLVM_INCLUDE_TESTS)
        add_dependencies(check-runtimes check-runtimes-${name})
        add_dependencies(runtimes-test-depends runtimes-test-depends-${name})
      endif()
      foreach(runtime_name ${runtime_names})
        add_dependencies(${runtime_name} ${runtime_name}-${name})
        add_dependencies(install-${runtime_name} install-${runtime_name}-${name})
        add_dependencies(install-${runtime_name}-stripped install-${runtime_name}-${name}-stripped)
      endforeach()
      foreach(component ${LLVM_RUNTIME_DISTRIBUTION_COMPONENTS})
        add_dependencies(${component} ${component}-${name})
        add_dependencies(install-${component} install-${component}-${name})
        add_dependencies(install-${component}-stripped install-${component}-${name}-stripped)
      endforeach()
    endforeach()

    foreach(multilib ${LLVM_RUNTIME_MULTILIBS})
      foreach(name ${LLVM_RUNTIME_MULTILIB_${multilib}_TARGETS})
        runtime_register_target(${name}+${multilib} ${name}
          DEPENDS runtimes-${name}
          CMAKE_ARGS -DLLVM_RUNTIMES_PREFIX=${name}/
                     -DLLVM_RUNTIMES_LIBDIR_SUBDIR=${multilib})
        add_dependencies(runtimes runtimes-${name}+${multilib})
        add_dependencies(runtimes-configure runtimes-${name}+${multilib}-configure)
        add_dependencies(install-runtimes install-runtimes-${name}+${multilib})
        add_dependencies(install-runtimes-stripped install-runtimes-${name}+${multilib}-stripped)
        foreach(runtime_name ${runtime_names})
          add_dependencies(${runtime_name} ${runtime_name}-${name}+${multilib})
          add_dependencies(install-${runtime_name} install-${runtime_name}-${name}+${multilib})
          add_dependencies(install-${runtime_name}-stripped install-${runtime_name}-${name}+${multilib}-stripped)
        endforeach()
        foreach(component ${LLVM_RUNTIME_DISTRIBUTION_COMPONENTS})
          add_dependencies(${component} ${component}-${name}+${multilib})
          add_dependencies(install-${component} install-${component}-${name}+${multilib})
          add_dependencies(install-${component}-stripped install-${component}-${name}+${multilib}-stripped)
        endforeach()
      endforeach()
    endforeach()
  endif()

  if(NOT LLVM_BUILD_INSTRUMENTED AND CLANG_ENABLE_BOOTSTRAP)
    # TODO: This is a hack needed because the libcxx headers are copied into the
    # build directory during configuration. Without that step the clang in the
    # build directory cannot find the C++ headers in certain configurations.
    # I need to build a mechanism for runtime projects to provide CMake code
    # that executes at LLVM configuration time to handle this case.
    add_dependencies(clang-bootstrap-deps runtimes-configure)
    # We need to add the runtimes as a dependency because compiler-rt can be
    # built as part of runtimes and we need the profile runtime for PGO
    add_dependencies(clang-bootstrap-deps runtimes)
  endif()

  if(LLVM_INCLUDE_TESTS)
    set_property(GLOBAL APPEND PROPERTY LLVM_ALL_ADDITIONAL_TEST_DEPENDS runtimes-test-depends)

    set(RUNTIMES_TEST_DEPENDS
        FileCheck
        count
        llvm-nm
        llvm-objdump
        llvm-xray
        not
        obj2yaml
        sancov
        sanstats
        llvm_gtest_main
        llvm_gtest
      )
    foreach(target ${test_targets} ${SUB_CHECK_TARGETS})
      add_dependencies(${target} ${RUNTIMES_TEST_DEPENDS})
    endforeach()

    set_property(GLOBAL APPEND PROPERTY LLVM_ALL_ADDITIONAL_TEST_TARGETS runtimes ${RUNTIMES_TEST_DEPENDS})
  endif()
endif()
