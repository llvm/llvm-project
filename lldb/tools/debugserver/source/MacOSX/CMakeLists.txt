#add_subdirectory(arm64)
#add_subdirectory(arm)
add_subdirectory(i386)
#add_subdirectory(ppc)
add_subdirectory(x86_64)
add_subdirectory(DarwinLog)

include_directories(..)

set(generated_mach_interfaces
  ${CMAKE_CURRENT_BINARY_DIR}/mach_exc.h
  ${CMAKE_CURRENT_BINARY_DIR}/mach_excServer.c
  ${CMAKE_CURRENT_BINARY_DIR}/mach_excUser.c
  )
add_custom_command(OUTPUT ${generated_mach_interfaces}
  COMMAND mig ${CMAKE_CURRENT_SOURCE_DIR}/dbgnub-mig.defs
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dbgnub-mig.defs
  )

set(DEBUGSERVER_VERS_GENERATED_FILE ${CMAKE_CURRENT_BINARY_DIR}/debugserver_vers.c)
set_source_files_properties(${DEBUGSERVER_VERS_GENERATED_FILE} PROPERTIES GENERATED 1)

add_custom_command(OUTPUT ${DEBUGSERVER_VERS_GENERATED_FILE}
  COMMAND ${LLDB_SOURCE_DIR}/scripts/generate-vers.pl
          ${LLDB_SOURCE_DIR}/lldb.xcodeproj/project.pbxproj debugserver
          > ${DEBUGSERVER_VERS_GENERATED_FILE}
  DEPENDS ${LLDB_SOURCE_DIR}/scripts/generate-vers.pl
          ${LLDB_SOURCE_DIR}/lldb.xcodeproj/project.pbxproj
  )

add_lldb_tool(debugserver INCLUDE_IN_FRAMEWORK
  HasAVX.s
  CFBundle.cpp
  CFString.cpp
  Genealogy.cpp
  MachException.cpp
  MachProcess.mm
  MachTask.mm
  MachThread.cpp
  MachThreadList.cpp
  MachVMMemory.cpp
  MachVMRegion.cpp
  OsLogger.cpp
  ${generated_mach_interfaces}
  ${DEBUGSERVER_VERS_GENERATED_FILE}

  LINK_LIBS
    lldbDebugserverCommon
    lldbDebugserverMacOSX_I386
    lldbDebugserverMacOSX_X86_64
    lldbDebugserverMacOSX_DarwinLog
  )

set_source_files_properties(
  HasAVX.s
  # Necessary since compilation will fail with stand-alone assembler
  PROPERTIES LANGUAGE C COMPILE_FLAGS "-x assembler-with-cpp"
  )

set(LLDB_CODESIGN_IDENTITY "lldb_codesign"
  CACHE STRING "Identity used for code signing. Set to empty string to skip the signing step.")
if (NOT ("${LLDB_CODESIGN_IDENTITY}" STREQUAL ""))
  execute_process(
    COMMAND xcrun -f codesign_allocate
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE CODESIGN_ALLOCATE
    )
  add_custom_command(TARGET debugserver
    POST_BUILD
    # Note: --entitlements option removed (see comment above).
    COMMAND ${CMAKE_COMMAND} -E env CODESIGN_ALLOCATE=${CODESIGN_ALLOCATE}
            codesign --force --sign ${LLDB_CODESIGN_IDENTITY}
            $<TARGET_FILE:debugserver>
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
endif()
