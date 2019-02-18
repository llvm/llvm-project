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

  # libLLVM.so should not have any dependencies on any other LLVM
  # shared libraries. When using the "all" pseudo-component,
  # LLVM_AVAILABLE_LIBS is added to the dependencies, which may
  # contain shared libraries (e.g. libLTO).
  #
  # Also exclude libLLVMTableGen for the following reasons:
  #  - it is only used by internal *-tblgen utilities;
  #  - it pollutes the global options space.
  foreach(lib ${LIB_NAMES})
    get_target_property(t ${lib} TYPE)
    if("${lib}" STREQUAL "LLVMTableGen")
    elseif("x${t}" STREQUAL "xSTATIC_LIBRARY")
      list(APPEND FILTERED_LIB_NAMES ${lib})
    endif()
  endforeach()
  set(LIB_NAMES ${FILTERED_LIB_NAMES})

  if(LLVM_DYLIB_EXPORTED_SYMBOL_FILE)
    set(LLVM_EXPORTED_SYMBOL_FILE ${LLVM_DYLIB_EXPORTED_SYMBOL_FILE})
    add_custom_target(libLLVMExports DEPENDS ${LLVM_EXPORTED_SYMBOL_FILE})
  endif()

  add_llvm_library(LLVM SHARED DISABLE_LLVM_LINK_LLVM_DYLIB SONAME ${SOURCES})

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
    if (NOT LLVM_LINKER_IS_SOLARISLD)
      # Solaris ld does not accept global: *; so there is no way to version *all* global symbols
      set(LIB_NAMES -Wl,--version-script,${LLVM_LIBRARY_DIR}/tools/llvm-shlib/simple_version_script.map ${LIB_NAMES})
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

  set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_BINARY_DIR}/libllvm-c.exports)

  set(LIB_DIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX})
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

  add_llvm_library(LLVM-C SHARED ${SOURCES})
  
  target_link_libraries(LLVM-C PUBLIC LLVM)
  add_dependencies(LLVM-C libLLVMCExports)
  
  set_property(TARGET LLVM-C APPEND_STRING PROPERTY
              LINK_FLAGS
              " -compatibility_version 1 -current_version ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH} -Wl,-reexport_library ${LIB_PATH}")
endif()

if(MSVC)
  # Build the LLVM-C.dll library that exports the C API.

  set(LLVM_LINK_COMPONENTS
    ${LLVM_DYLIB_COMPONENTS}
    )

  llvm_map_components_to_libnames(LIB_NAMES ${LLVM_DYLIB_COMPONENTS})
  list(REMOVE_DUPLICATES LIB_NAMES)

  # The python script needs to know whether symbols are prefixed with underscores or not.
  if(LLVM_HOST_TRIPLE STREQUAL "i686-pc-win32")
    set(GEN_UNDERSCORE "--underscore")
  else()
    set(GEN_UNDERSCORE "")
  endif()

  # Get the full name to the libs so the python script understands them.
  foreach(lib ${LIB_NAMES})
    list(APPEND FULL_LIB_NAMES ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/${lib}.lib)
  endforeach()

  # Generate the exports file dynamically.
  set(GEN_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/gen-msvc-exports.py)

  set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/libllvm-c.exports)

  add_custom_command(OUTPUT ${LLVM_EXPORTED_SYMBOL_FILE}
    COMMAND ${PYTHON_EXECUTABLE} ${GEN_SCRIPT} ${FULL_LIB_NAMES} ${GEN_UNDERSCORE} --nm ${LLVM_TOOLS_BINARY_DIR}/llvm-nm -o ${LLVM_EXPORTED_SYMBOL_FILE}
    DEPENDS ${LIB_NAMES} llvm-nm
    COMMENT "Generating export list for LLVM-C"
    VERBATIM )

  # Finally link the target.
  add_llvm_library(LLVM-C SHARED ${SOURCES} DEPENDS intrinsics_gen)

endif()
