# This tool creates a shared library from the LLVM libraries. Generating this
# library is enabled by setting LLVM_BUILD_LLVM_DYLIB=yes on the CMake
# commandline. By default the shared library only exports the LLVM C API.

set(SOURCES
  libllvm.cpp
  )

if(LLVM_LINK_LLVM_DYLIB AND LLVM_DYLIB_EXPORTED_SYMBOL_FILE)
  message(WARNING "Using LLVM_LINK_LLVM_DYLIB with LLVM_DYLIB_EXPORTED_SYMBOL_FILE may not work. Use at your own risk.")
endif()

if(LLVM_BUILD_LLVM_DYLIB)
  if(MSVC)
    message(FATAL_ERROR "Generating libLLVM is not supported on MSVC")
  endif()

  llvm_map_components_to_libnames(LIB_NAMES ${LLVM_DYLIB_COMPONENTS})

  # Exclude libLLVMTableGen for the following reasons:
  #  - it is only used by internal *-tblgen utilities;
  #  - it pollutes the global options space.
  list(REMOVE_ITEM LIB_NAMES "LLVMTableGen")

  if(LLVM_DYLIB_EXPORTED_SYMBOL_FILE)
    set(LLVM_EXPORTED_SYMBOL_FILE ${LLVM_DYLIB_EXPORTED_SYMBOL_FILE})
    add_custom_target(libLLVMExports DEPENDS ${LLVM_EXPORTED_SYMBOL_FILE})
  endif()

  if (LLVM_LINK_LLVM_DYLIB)
    set(INSTALL_WITH_TOOLCHAIN INSTALL_WITH_TOOLCHAIN)
  endif()
  add_llvm_library(LLVM SHARED DISABLE_LLVM_LINK_LLVM_DYLIB SONAME ${INSTALL_WITH_TOOLCHAIN} ${SOURCES})

  list(REMOVE_DUPLICATES LIB_NAMES)
  if(("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux") OR (MINGW) OR (HAIKU)
     OR ("${CMAKE_SYSTEM_NAME}" STREQUAL "FreeBSD")
     OR ("${CMAKE_SYSTEM_NAME}" STREQUAL "GNU")
     OR ("${CMAKE_SYSTEM_NAME}" STREQUAL "OpenBSD")
     OR ("${CMAKE_SYSTEM_NAME}" STREQUAL "Fuchsia")
     OR ("${CMAKE_SYSTEM_NAME}" STREQUAL "DragonFly")
     OR ("${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS")) # FIXME: It should be "GNU ld for elf"
    configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_version_script.map.in
    ${LLVM_LIBRARY_DIR}/tools/llvm-shlib/simple_version_script.map)

    # GNU ld doesn't resolve symbols in the version script.
    set(LIB_NAMES -Wl,--whole-archive ${LIB_NAMES} -Wl,--no-whole-archive)
    if (NOT LLVM_LINKER_IS_SOLARISLD AND NOT MINGW)
      # Solaris ld does not accept global: *; so there is no way to version *all* global symbols
      set(LIB_NAMES -Wl,--version-script,${LLVM_LIBRARY_DIR}/tools/llvm-shlib/simple_version_script.map ${LIB_NAMES})
    endif()
    if (NOT MINGW)
      # Optimize function calls for default visibility definitions to avoid PLT and
      # reduce dynamic relocations.
      # Note: for -fno-pic default, the address of a function may be different from
      # inside and outside libLLVM.so.
      target_link_options(LLVM PRIVATE LINKER:-Bsymbolic-functions)
    endif()
  elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
    set(LIB_NAMES -Wl,-all_load ${LIB_NAMES})
  endif()

  target_link_libraries(LLVM PRIVATE ${LIB_NAMES})

  if (APPLE)
    set_property(TARGET LLVM APPEND_STRING PROPERTY
                LINK_FLAGS
                " -compatibility_version 1 -current_version ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
  endif()

  if(TARGET libLLVMExports)
    add_dependencies(LLVM libLLVMExports)
  endif()
endif()

if(LLVM_BUILD_LLVM_C_DYLIB AND NOT MSVC)
  if(NOT APPLE)
    message(FATAL_ERROR "Generating libLLVM-c is only supported on Darwin")
  endif()

  if(NOT LLVM_BUILD_LLVM_DYLIB)
    message(FATAL_ERROR "Generating libLLVM-c requires LLVM_BUILD_LLVM_C_DYLIB on Darwin")
  endif()

  # To get the export list for a single llvm library:
  # nm ${LIB_PATH} | awk "/T _LLVM/ { print $3 }" | sort -u | sed -e "s/^_//g" > ${LIB_PATH}.exports

  set(LLVM_EXPORTED_SYMBOL_FILE ${LLVM_BINARY_DIR}/libllvm-c.exports)

  set(LIB_DIR ${LLVM_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX})
  set(LIB_NAME ${LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}LLVM)
  set(LIB_PATH ${LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(LIB_EXPORTS_PATH ${LIB_NAME}.exports)
  list(APPEND LLVM_DYLIB_REQUIRED_EXPORTS ${LIB_EXPORTS_PATH})

  add_custom_command(OUTPUT ${LLVM_EXPORTED_SYMBOL_FILE}
    COMMAND nm ${LIB_PATH} | awk "/T _LLVM/ || /T LLVM/ { print $3 }" | sort -u | sed -e "s/^_//g" > ${LLVM_EXPORTED_SYMBOL_FILE}
    WORKING_DIRECTORY ${LIB_DIR}
    DEPENDS LLVM
    COMMENT "Generating Export list for LLVM..."
    VERBATIM )

  add_custom_target(libLLVMCExports DEPENDS ${LLVM_EXPORTED_SYMBOL_FILE})

  add_llvm_library(LLVM-C SHARED ${SOURCES} INSTALL_WITH_TOOLCHAIN)

  target_link_libraries(LLVM-C PUBLIC LLVM)
  add_dependencies(LLVM-C libLLVMCExports)

  set_property(TARGET LLVM-C APPEND_STRING PROPERTY
              LINK_FLAGS
              " -compatibility_version 1 -current_version ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH} -Wl,-reexport_library ${LIB_PATH}")
endif()

if(LLVM_BUILD_LLVM_C_DYLIB AND MSVC)
  # Build the LLVM-C.dll library that exports the C API.

  set(LLVM_LINK_COMPONENTS
    ${LLVM_DYLIB_COMPONENTS}
    )

  llvm_map_components_to_libnames(LIB_NAMES ${LLVM_DYLIB_COMPONENTS})
  list(REMOVE_DUPLICATES LIB_NAMES)

  # The python script needs to know whether symbols are prefixed with underscores or not.
  if(LLVM_HOST_TRIPLE MATCHES "i?86-.*win.*")
    set(GEN_UNDERSCORE "--underscore")
  else()
    set(GEN_UNDERSCORE "")
  endif()

  # Set this name here, not used in multi conf loop,
  # but add script will pick the right one.
  set(LIBSFILE ${LLVM_BINARY_DIR}/${CMAKE_CFG_INTDIR}/libllvm-c.args)

  # Get the full name to the libs so the python script understands them.
  foreach(lib ${LIB_NAMES})
    list(APPEND FULL_LIB_NAMES ${LLVM_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/${lib}.lib)
  endforeach()

  # Need to separate lib names with newlines.
  string(REPLACE ";" "\n" FILE_CONTENT "${FULL_LIB_NAMES}")

  if(NOT "${CMAKE_CFG_INTDIR}" STREQUAL ".")
    foreach(BUILD_MODE ${CMAKE_CONFIGURATION_TYPES})
      # Replace the special string with a per config directory.
      string(REPLACE ${CMAKE_CFG_INTDIR} ${BUILD_MODE} PER_CONF_CONTENT "${FILE_CONTENT}")

      # Write out the full lib names into file to be read by the python script.
      # One libsfile per build, the add_custom_command should expand
      # ${CMAKE_CFG_INTDIR} correctly and select the right one.
      file(WRITE ${LLVM_BINARY_DIR}/${BUILD_MODE}/libllvm-c.args "${PER_CONF_CONTENT}")
    endforeach()
  else()
    # Write out the full lib names into file to be read by the python script.
    file(WRITE ${LIBSFILE} "${FILE_CONTENT}")
  endif()

  # Generate the exports file dynamically.
  set(GEN_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/gen-msvc-exports.py)

  set(LLVM_EXPORTED_SYMBOL_FILE ${LLVM_BINARY_DIR}/${CMAKE_CFG_INTDIR}/libllvm-c.exports)
  if(NOT LLVM_NM)
    if(CMAKE_CROSSCOMPILING)
      build_native_tool(llvm-nm llvm_nm)
      set(llvm_nm_target "${llvm_nm}")
    else()
      set(llvm_nm $<TARGET_FILE:llvm-nm>)
      set(llvm_nm_target llvm-nm)
    endif()
  else()
    set(llvm_nm ${LLVM_NM})
    set(llvm_nm_target "")
  endif()

  add_custom_command(OUTPUT ${LLVM_EXPORTED_SYMBOL_FILE}
    COMMAND "${Python3_EXECUTABLE}" ${GEN_SCRIPT} --libsfile ${LIBSFILE} ${GEN_UNDERSCORE} --nm "${llvm_nm}" -o ${LLVM_EXPORTED_SYMBOL_FILE}
    DEPENDS ${LIB_NAMES} ${llvm_nm_target}
    COMMENT "Generating export list for LLVM-C"
    VERBATIM )

  # Finally link the target.
  add_llvm_library(LLVM-C SHARED INSTALL_WITH_TOOLCHAIN ${SOURCES} DEPENDS intrinsics_gen)

  if (LLVM_INTEGRATED_CRT_ALLOC AND MSVC)
    # Make sure we search LLVMSupport first, before the CRT libs
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -INCLUDE:malloc")
  endif()
  
endif()
