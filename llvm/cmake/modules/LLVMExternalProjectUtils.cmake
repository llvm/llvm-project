include(ExternalProject)

# llvm_ExternalProject_BuildCmd(out_var target)
#   Utility function for constructing command lines for external project targets
function(llvm_ExternalProject_BuildCmd out_var target)
  if (CMAKE_GENERATOR MATCHES "Make")
    # Use special command for Makefiles to support parallelism.
    set(${out_var} "$(MAKE)" "${target}" PARENT_SCOPE)
  else()
    set(${out_var} ${CMAKE_COMMAND} --build . --target ${target}
                                    --config $<CONFIGURATION> PARENT_SCOPE)
  endif()
endfunction()

# llvm_ExternalProject_Add(name source_dir ...
#   USE_TOOLCHAIN
#     Use just-built tools (see TOOLCHAIN_TOOLS)
#   EXCLUDE_FROM_ALL
#     Exclude this project from the all target
#   NO_INSTALL
#     Don't generate install targets for this project
#   CMAKE_ARGS arguments...
#     Optional cmake arguments to pass when configuring the project
#   TOOLCHAIN_TOOLS targets...
#     Targets for toolchain tools (defaults to clang;lld)
#   DEPENDS targets...
#     Targets that this project depends on
#   EXTRA_TARGETS targets...
#     Extra targets in the subproject to generate targets for
#   )
function(llvm_ExternalProject_Add name source_dir)
  cmake_parse_arguments(ARG "USE_TOOLCHAIN;EXCLUDE_FROM_ALL;NO_INSTALL"
    "SOURCE_DIR"
    "CMAKE_ARGS;TOOLCHAIN_TOOLS;RUNTIME_LIBRARIES;DEPENDS;EXTRA_TARGETS" ${ARGN})
  canonicalize_tool_name(${name} nameCanon)
  if(NOT ARG_TOOLCHAIN_TOOLS)
    set(ARG_TOOLCHAIN_TOOLS clang lld)
  endif()
  foreach(tool ${ARG_TOOLCHAIN_TOOLS})
    if(TARGET ${tool})
      list(APPEND TOOLCHAIN_TOOLS ${tool})
      list(APPEND TOOLCHAIN_BINS $<TARGET_FILE:${tool}>)
    endif()
  endforeach()

  if(NOT ARG_RUNTIME_LIBRARIES)
    set(ARG_RUNTIME_LIBRARIES compiler-rt libcxx)
  endif()
  foreach(lib ${ARG_RUNTIME_LIBRARIES})
    if(TARGET ${lib})
      list(APPEND RUNTIME_LIBRARIES ${lib})
    endif()
  endforeach()

  list(FIND TOOLCHAIN_TOOLS clang FOUND_CLANG)
  if(FOUND_CLANG GREATER -1)
    set(CLANG_IN_TOOLCHAIN On)
  endif()

  if(RUNTIME_LIBRARIES AND CLANG_IN_TOOLCHAIN)
    list(APPEND TOOLCHAIN_BINS ${RUNTIME_LIBRARIES})
  endif()

  if(CMAKE_VERSION VERSION_GREATER 3.1.0)
    set(cmake_3_1_EXCLUDE_FROM_ALL EXCLUDE_FROM_ALL 1)
  endif()

  if(CMAKE_VERSION VERSION_GREATER 3.3.20150708)
    set(cmake_3_4_USES_TERMINAL_OPTIONS
      USES_TERMINAL_CONFIGURE 1
      USES_TERMINAL_BUILD 1
      USES_TERMINAL_INSTALL 1
      )
    set(cmake_3_4_USES_TERMINAL USES_TERMINAL 1)
  endif()

  if(CMAKE_VERSION VERSION_GREATER 3.1.20141116)
    set(cmake_3_2_USES_TERMINAL USES_TERMINAL)
  endif()

  set(STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/${name}-stamps/)
  set(BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${name}-bins/)

  add_custom_target(${name}-clear
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${STAMP_DIR}
    COMMENT "Clobbering ${name} build and stamp directories"
    ${cmake_3_2_USES_TERMINAL}
    )

  # Find all variables that start with COMPILER_RT and populate a variable with
  # them.
  get_cmake_property(variableNames VARIABLES)
  foreach(varaibleName ${variableNames})
    if(${varaibleName} MATCHES "^${nameCanon}")
      list(APPEND PASSTHROUGH_VARIABLES
        -D${varaibleName}=${${varaibleName}})
    endif()
  endforeach()

  if(ARG_USE_TOOLCHAIN)
    if(CLANG_IN_TOOLCHAIN)
      set(compiler_args -DCMAKE_C_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang
                        -DCMAKE_CXX_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang++)
    endif()
    list(APPEND ARG_DEPENDS ${TOOLCHAIN_TOOLS})
  endif()

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${name}-clobber-stamp
    DEPENDS ${ARG_DEPENDS}
    COMMAND ${CMAKE_COMMAND} -E touch ${BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E touch ${STAMP_DIR}/${name}-mkdir
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/${name}-clobber-stamp
    COMMENT "Clobbering bootstrap build and stamp directories"
    )

  add_custom_target(${name}-clobber
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${name}-clobber-stamp)

  if(ARG_EXCLUDE_FROM_ALL)
    set(exclude ${cmake_3_1_EXCLUDE_FROM_ALL})
  endif()

  ExternalProject_Add(${name}
    DEPENDS ${ARG_DEPENDS}
    ${name}-clobber
    PREFIX ${CMAKE_BINARY_DIR}/projects/${name}
    SOURCE_DIR ${source_dir}
    STAMP_DIR ${STAMP_DIR}
    BINARY_DIR ${BINARY_DIR}
    ${exclude}
    CMAKE_ARGS ${${nameCanon}_CMAKE_ARGS}
               ${compiler_args}
               -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
               ${ARG_CMAKE_ARGS}
               ${PASSTHROUGH_VARIABLES}
    INSTALL_COMMAND ""
    STEP_TARGETS configure build
    ${cmake_3_4_USES_TERMINAL_OPTIONS}
    )

  if(ARG_USE_TOOLCHAIN)
    ExternalProject_Add_Step(${name} force-rebuild
      COMMENT "Forcing rebuild becaues tools have changed"
      DEPENDERS configure
      DEPENDS ${TOOLCHAIN_BINS}
      ${cmake_3_4_USES_TERMINAL} )
  endif()

  if(ARG_USE_TOOLCHAIN)
    set(force_deps DEPENDS ${TOOLCHAIN_BINS})
  endif()

  llvm_ExternalProject_BuildCmd(run_clean clean)
  ExternalProject_Add_Step(${name} clean
    COMMAND ${run_clean}
    COMMENT "Cleaning ${name}..."
    DEPENDEES configure
    ${force_deps}
    WORKING_DIRECTORY ${BINARY_DIR}
    ${cmake_3_4_USES_TERMINAL}
    )
  ExternalProject_Add_StepTargets(${name} clean)

  if(ARG_USE_TOOLCHAIN)
    add_dependencies(${name}-clean ${name}-clobber)
    set_target_properties(${name}-clean PROPERTIES
      SOURCES ${CMAKE_CURRENT_BINARY_DIR}/${name}-clobber-stamp)
  endif()

  if(NOT ARG_NO_INSTALL)
    install(CODE "execute_process\(COMMAND \${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=\${CMAKE_INSTALL_PREFIX} -P ${BINARY_DIR}/cmake_install.cmake \)"
      COMPONENT ${name})

    add_custom_target(install-${name}
                      DEPENDS ${name}
                      COMMAND "${CMAKE_COMMAND}"
                               -DCMAKE_INSTALL_COMPONENT=${name}
                               -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
                      ${cmake_3_2_USES_TERMINAL})
  endif()

  # Add top-level targets
  foreach(target ${ARG_EXTRA_TARGETS})
    llvm_ExternalProject_BuildCmd(build_runtime_cmd ${target})
    add_custom_target(${target}
      COMMAND ${build_runtime_cmd}
      DEPENDS ${name}-configure
      WORKING_DIRECTORY ${BINARY_DIR}
      VERBATIM
      ${cmake_3_2_USES_TERMINAL})
  endforeach()
endfunction()
