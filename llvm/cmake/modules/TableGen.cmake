# LLVM_TARGET_DEFINITIONS must contain the name of the .td file to process.
# Extra parameters for `tblgen' may come after `ofn' parameter.
# Adds the name of the generated file to TABLEGEN_OUTPUT.

function(tablegen project ofn)
  # Validate calling context.
  foreach(v
      ${project}_TABLEGEN_EXE
      LLVM_MAIN_SRC_DIR
      LLVM_MAIN_INCLUDE_DIR
      )
    if(NOT ${v})
      message(FATAL_ERROR "${v} not set")
    endif()
  endforeach()

  file(GLOB local_tds "*.td")
  file(GLOB_RECURSE global_tds "${LLVM_MAIN_INCLUDE_DIR}/llvm/*.td")

  if (IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
  else()
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE
      ${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_TARGET_DEFINITIONS})
  endif()
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ofn}.tmp
    # Generate tablegen output in a temporary file.
    COMMAND ${${project}_TABLEGEN_EXE} ${ARGN} -I ${CMAKE_CURRENT_SOURCE_DIR}
    -I ${LLVM_MAIN_SRC_DIR}/lib/Target -I ${LLVM_MAIN_INCLUDE_DIR}
    ${LLVM_TARGET_DEFINITIONS_ABSOLUTE}
    -o ${CMAKE_CURRENT_BINARY_DIR}/${ofn}.tmp
    # The file in LLVM_TARGET_DEFINITIONS may be not in the current
    # directory and local_tds may not contain it, so we must
    # explicitly list it here:
    DEPENDS ${${project}_TABLEGEN_TARGET} ${local_tds} ${global_tds}
    ${LLVM_TARGET_DEFINITIONS_ABSOLUTE}
    COMMENT "Building ${ofn}..."
    )
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
    # Only update the real output file if there are any differences.
    # This prevents recompilation of all the files depending on it if there
    # aren't any.
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_BINARY_DIR}/${ofn}.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${ofn}.tmp
    COMMENT "Updating ${ofn}..."
    )

  # `make clean' must remove all those generated files:
  set_property(DIRECTORY APPEND
    PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${ofn}.tmp ${ofn})

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

if(LLVM_USE_HOST_TOOLS)
  add_custom_command(OUTPUT LIB_LLVMSUPPORT
      COMMAND ${CMAKE_COMMAND} --build . --target LLVMSupport --config Release
      DEPENDS CONFIGURE_LLVM_NATIVE
      WORKING_DIRECTORY ${LLVM_NATIVE_BUILD}
      COMMENT "Building libLLVMSupport for native TableGen...")
  add_custom_target(NATIVE_LIB_LLVMSUPPORT DEPENDS LIB_LLVMSUPPORT)

  add_custom_command(OUTPUT LIB_LLVMTABLEGEN
      COMMAND ${CMAKE_COMMAND} --build . --target LLVMTableGen --config Release
      DEPENDS CONFIGURE_LLVM_NATIVE
      WORKING_DIRECTORY ${LLVM_NATIVE_BUILD}
      COMMENT "Building libLLVMTableGen for native TableGen...")
  add_custom_target(NATIVE_LIB_LLVMTABLEGEN DEPENDS LIB_LLVMTABLEGEN)
endif(LLVM_USE_HOST_TOOLS)

macro(add_tablegen target project)
  set(${target}_OLD_LLVM_LINK_COMPONENTS ${LLVM_LINK_COMPONENTS})
  set(LLVM_LINK_COMPONENTS ${LLVM_LINK_COMPONENTS} TableGen)

  if(NOT XCODE)
    # FIXME: It leaks to user, callee of add_tablegen.
    set(LLVM_ENABLE_OBJLIB ON)
  endif()

  add_llvm_utility(${target} ${ARGN})
  set(LLVM_LINK_COMPONENTS ${${target}_OLD_LLVM_LINK_COMPONENTS})

  set(${project}_TABLEGEN "${target}" CACHE
      STRING "Native TableGen executable. Saves building one when cross-compiling.")

  # Upgrade existing LLVM_TABLEGEN setting.
  if(${project} STREQUAL LLVM)
    if(${LLVM_TABLEGEN} STREQUAL tblgen)
      set(LLVM_TABLEGEN "${target}" CACHE
          STRING "Native TableGen executable. Saves building one when cross-compiling."
          FORCE)
    endif()
  endif()

  # Effective tblgen executable to be used:
  set(${project}_TABLEGEN_EXE ${${project}_TABLEGEN} PARENT_SCOPE)
  set(${project}_TABLEGEN_TARGET ${${project}_TABLEGEN} PARENT_SCOPE)

  if(LLVM_USE_HOST_TOOLS)
    if( ${${project}_TABLEGEN} STREQUAL "${target}" )
      if (NOT CMAKE_CONFIGURATION_TYPES)
        set(${project}_TABLEGEN_EXE "${LLVM_NATIVE_BUILD}/bin/${target}")
      else()
        set(${project}_TABLEGEN_EXE "${LLVM_NATIVE_BUILD}/Release/bin/${target}")
      endif()
      set(${project}_TABLEGEN_EXE ${${project}_TABLEGEN_EXE} PARENT_SCOPE)

      add_custom_command(OUTPUT ${${project}_TABLEGEN_EXE}
        COMMAND ${CMAKE_COMMAND} --build . --target ${target} --config Release
        DEPENDS ${target} NATIVE_LIB_LLVMSUPPORT NATIVE_LIB_LLVMTABLEGEN
        WORKING_DIRECTORY ${LLVM_NATIVE_BUILD}
        COMMENT "Building native TableGen...")
      add_custom_target(${project}-tablegen-host DEPENDS ${${project}_TABLEGEN_EXE})
      set(${project}_TABLEGEN_TARGET ${project}-tablegen-host PARENT_SCOPE)
    endif()
  endif()

  if( MINGW )
    if(CMAKE_SIZEOF_VOID_P MATCHES "8")
      set_target_properties(${target} PROPERTIES LINK_FLAGS -Wl,--stack,16777216)
    endif(CMAKE_SIZEOF_VOID_P MATCHES "8")
  endif( MINGW )
  if (${project} STREQUAL LLVM AND NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
    install(TARGETS ${target}
            EXPORT LLVMExports
            RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR})
  endif()
  set_property(GLOBAL APPEND PROPERTY LLVM_EXPORTS ${target})
endmacro()
