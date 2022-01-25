# LLVM_TARGET_DEFINITIONS must contain the name of the .td file to process,
# while LLVM_TARGET_DEPENDS may contain additional file dependencies.
# Extra parameters for `tblgen' may come after `ofn' parameter.
# Adds the name of the generated file to TABLEGEN_OUTPUT.

function(tablegen project ofn)
  # Validate calling context.
  if(NOT ${project}_TABLEGEN_EXE)
    message(FATAL_ERROR "${project}_TABLEGEN_EXE not set")
  endif()

  # Use depfile instead of globbing arbitrary *.td(s) for Ninja.
  if(CMAKE_GENERATOR MATCHES "Ninja")
    # Make output path relative to build.ninja, assuming located on
    # ${CMAKE_BINARY_DIR}.
    # CMake emits build targets as relative paths but Ninja doesn't identify
    # absolute path (in *.d) as relative path (in build.ninja)
    # Note that tblgen is executed on ${CMAKE_BINARY_DIR} as working directory.
    file(RELATIVE_PATH ofn_rel
      ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${ofn})
    set(additional_cmdline
      -o ${ofn_rel}
      -d ${ofn_rel}.d
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      DEPFILE ${CMAKE_CURRENT_BINARY_DIR}/${ofn}.d
      )
    set(local_tds)
    set(global_tds)
  else()
    file(GLOB local_tds "*.td")
    file(GLOB_RECURSE global_tds "${LLVM_MAIN_INCLUDE_DIR}/llvm/*.td")
    set(additional_cmdline
      -o ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      )
  endif()

  if (IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
  else()
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE
      ${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_TARGET_DEFINITIONS})
  endif()
  if (LLVM_ENABLE_DAGISEL_COV)
    list(FIND ARGN "-gen-dag-isel" idx)
    if( NOT idx EQUAL -1 )
      list(APPEND LLVM_TABLEGEN_FLAGS "-instrument-coverage")
    endif()
  endif()
  if (LLVM_ENABLE_GISEL_COV)
    list(FIND ARGN "-gen-global-isel" idx)
    if( NOT idx EQUAL -1 )
      list(APPEND LLVM_TABLEGEN_FLAGS "-instrument-gisel-coverage")
      list(APPEND LLVM_TABLEGEN_FLAGS "-gisel-coverage-file=${LLVM_GISEL_COV_PREFIX}all")
    endif()
  endif()
  if (LLVM_OMIT_DAGISEL_COMMENTS)
    list(FIND ARGN "-gen-dag-isel" idx)
    if (NOT idx EQUAL -1)
      list(APPEND LLVM_TABLEGEN_FLAGS "-omit-comments")
    endif()
  endif()

  # MSVC can't support long string literals ("long" > 65534 bytes)[1], so if there's
  # a possibility of generated tables being consumed by MSVC, generate arrays of
  # char literals, instead. If we're cross-compiling, then conservatively assume
  # that the source might be consumed by MSVC.
  # [1] https://docs.microsoft.com/en-us/cpp/cpp/compiler-limits?view=vs-2017
  if (MSVC AND project STREQUAL LLVM)
    list(APPEND LLVM_TABLEGEN_FLAGS "--long-string-literals=0")
  endif()
  if (CMAKE_GENERATOR MATCHES "Visual Studio")
    # Visual Studio has problems with llvm-tblgen's native --write-if-changed
    # behavior. Since it doesn't do restat optimizations anyway, just don't
    # pass --write-if-changed there.
    set(tblgen_change_flag)
  else()
    set(tblgen_change_flag "--write-if-changed")
  endif()

  if (NOT LLVM_ENABLE_WARNINGS)
    list(APPEND LLVM_TABLEGEN_FLAGS "-no-warn-on-unused-template-args")
  endif()

  # We need both _TABLEGEN_TARGET and _TABLEGEN_EXE in the  DEPENDS list
  # (both the target and the file) to have .inc files rebuilt on
  # a tablegen change, as cmake does not propagate file-level dependencies
  # of custom targets. See the following ticket for more information:
  # https://cmake.org/Bug/view.php?id=15858
  # The dependency on both, the target and the file, produces the same
  # dependency twice in the result file when
  # ("${${project}_TABLEGEN_TARGET}" STREQUAL "${${project}_TABLEGEN_EXE}")
  # but lets us having smaller and cleaner code here.
  get_directory_property(tblgen_includes INCLUDE_DIRECTORIES)
  list(TRANSFORM tblgen_includes PREPEND -I)

  set(tablegen_exe ${${project}_TABLEGEN_EXE})
  set(tablegen_depends ${${project}_TABLEGEN_TARGET} ${tablegen_exe})

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
    COMMAND ${tablegen_exe} ${ARGN} -I ${CMAKE_CURRENT_SOURCE_DIR}
    ${tblgen_includes}
    ${LLVM_TABLEGEN_FLAGS}
    ${LLVM_TARGET_DEFINITIONS_ABSOLUTE}
    ${tblgen_change_flag}
    ${additional_cmdline}
    # The file in LLVM_TARGET_DEFINITIONS may be not in the current
    # directory and local_tds may not contain it, so we must
    # explicitly list it here:
    DEPENDS ${tablegen_depends}
      ${local_tds} ${global_tds}
    ${LLVM_TARGET_DEFINITIONS_ABSOLUTE}
    ${LLVM_TARGET_DEPENDS}
    COMMENT "Building ${ofn}..."
    )

  # `make clean' must remove all those generated files:
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${ofn})

  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn} PARENT_SCOPE)
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${ofn} PROPERTIES
    GENERATED 1)
endfunction()

# Creates a target for publicly exporting tablegen dependencies.
function(add_public_tablegen_target target)
  if(NOT TABLEGEN_OUTPUT)
    message(FATAL_ERROR "Requires tablegen() definitions as TABLEGEN_OUTPUT.")
  endif()
  add_custom_target(${target}
    DEPENDS ${TABLEGEN_OUTPUT})
  if(LLVM_COMMON_DEPENDS)
    add_dependencies(${target} ${LLVM_COMMON_DEPENDS})
  endif()
  set_target_properties(${target} PROPERTIES FOLDER "Tablegenning")
  set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} ${target} PARENT_SCOPE)
endfunction()

macro(add_tablegen target project)
  set(${target}_OLD_LLVM_LINK_COMPONENTS ${LLVM_LINK_COMPONENTS})
  set(LLVM_LINK_COMPONENTS ${LLVM_LINK_COMPONENTS} TableGen)

  # CMake doesn't let compilation units depend on their dependent libraries on some generators.
  if(NOT CMAKE_GENERATOR MATCHES "Ninja" AND NOT XCODE)
    # FIXME: It leaks to user, callee of add_tablegen.
    set(LLVM_ENABLE_OBJLIB ON)
  endif()

  add_llvm_executable(${target} DISABLE_LLVM_LINK_LLVM_DYLIB ${ARGN})
  set(LLVM_LINK_COMPONENTS ${${target}_OLD_LLVM_LINK_COMPONENTS})

  set(${project}_TABLEGEN "${target}" CACHE
      STRING "Native TableGen executable. Saves building one when cross-compiling.")

  # Effective tblgen executable to be used:
  set(${project}_TABLEGEN_EXE ${${project}_TABLEGEN} PARENT_SCOPE)
  set(${project}_TABLEGEN_TARGET ${${project}_TABLEGEN} PARENT_SCOPE)

  if(LLVM_USE_HOST_TOOLS)
    if( ${${project}_TABLEGEN} STREQUAL "${target}" )
      # The NATIVE tablegen executable *must* depend on the current target one
      # otherwise the native one won't get rebuilt when the tablgen sources
      # change, and we end up with incorrect builds.
      build_native_tool(${target} ${project}_TABLEGEN_EXE DEPENDS ${target})
      set(${project}_TABLEGEN_EXE ${${project}_TABLEGEN_EXE} PARENT_SCOPE)

      add_custom_target(${project}-tablegen-host DEPENDS ${${project}_TABLEGEN_EXE})
      set(${project}_TABLEGEN_TARGET ${project}-tablegen-host PARENT_SCOPE)

      # Create an artificial dependency between tablegen projects, because they
      # compile the same dependencies, thus using the same build folders.
      # FIXME: A proper fix requires sequentially chaining tablegens.
      if (NOT ${project} STREQUAL LLVM AND TARGET ${project}-tablegen-host AND
          TARGET LLVM-tablegen-host)
        add_dependencies(${project}-tablegen-host LLVM-tablegen-host)
      endif()

      # If we're using the host tablegen, and utils were not requested, we have no
      # need to build this tablegen.
      if ( NOT LLVM_BUILD_UTILS )
        set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL ON)
      endif()
    endif()
  endif()

  if ((${project} STREQUAL LLVM OR ${project} STREQUAL MLIR) AND NOT LLVM_INSTALL_TOOLCHAIN_ONLY AND LLVM_BUILD_UTILS)
    set(export_to_llvmexports)
    if(${target} IN_LIST LLVM_DISTRIBUTION_COMPONENTS OR
        NOT LLVM_DISTRIBUTION_COMPONENTS)
      set(export_to_llvmexports EXPORT LLVMExports)
    endif()

    install(TARGETS ${target}
            ${export_to_llvmexports}
            COMPONENT ${target}
            RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR})
    if(NOT LLVM_ENABLE_IDE)
      add_llvm_install_targets("install-${target}"
                               DEPENDS ${target}
                               COMPONENT ${target})
    endif()
  endif()
  set_property(GLOBAL APPEND PROPERTY LLVM_EXPORTS ${target})
endmacro()
