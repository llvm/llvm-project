
if( ${LLVM_TABLEGEN} STREQUAL "tblgen" )
  set(CX_NATIVE_TG_DIR "${CMAKE_BINARY_DIR}/native")
  set(LLVM_TABLEGEN "${CX_NATIVE_TG_DIR}/bin/tblgen")

  add_custom_command(OUTPUT ${CX_NATIVE_TG_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CX_NATIVE_TG_DIR}
    COMMENT "Creating ${CX_NATIVE_TG_DIR}...")

  add_custom_command(OUTPUT ${CX_NATIVE_TG_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -UMAKE_TOOLCHAIN_FILE -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
    WORKING_DIRECTORY ${CX_NATIVE_TG_DIR}
    DEPENDS ${CX_NATIVE_TG_DIR}
    COMMENT "Configuring native TableGen...")

  add_custom_command(OUTPUT ${LLVM_TABLEGEN}
    COMMAND ${CMAKE_BUILD_TOOL}
    DEPENDS ${CX_NATIVE_TG_DIR}/CMakeCache.txt
    WORKING_DIRECTORY ${CX_NATIVE_TG_DIR}/utils/TableGen
    COMMENT "Building native TableGen...")
  add_custom_target(NativeTableGen DEPENDS ${LLVM_TABLEGEN})

  add_dependencies(tblgen NativeTableGen)

  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CX_NATIVE_TG_DIR})
endif()
