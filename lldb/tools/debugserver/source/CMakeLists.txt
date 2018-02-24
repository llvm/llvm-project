include(CheckCXXCompilerFlag)
include(CheckLibraryExists)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/..)
include_directories(${LLDB_SOURCE_DIR}/source)
include_directories(MacOSX/DarwinLog)

include_directories(MacOSX)
#include_directories(${CMAKE_CURRENT_BINARY_DIR}/MacOSX)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -Wl,-sectcreate,__TEXT,__info_plist,${CMAKE_CURRENT_SOURCE_DIR}/../resources/lldb-debugserver-Info.plist")

check_cxx_compiler_flag("-Wno-gnu-zero-variadic-macro-arguments"
                        CXX_SUPPORTS_NO_GNU_ZERO_VARIADIC_MACRO_ARGUMENTS)
if (CXX_SUPPORTS_NO_GNU_ZERO_VARIADIC_MACRO_ARGUMENTS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-gnu-zero-variadic-macro-arguments")
endif ()

check_cxx_compiler_flag("-Wno-zero-length-array"
                        CXX_SUPPORTS_NO_ZERO_LENGTH_ARRAY)
if (CXX_SUPPORTS_NO_ZERO_LENGTH_ARRAY)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-zero-length-array")
endif ()

check_cxx_compiler_flag("-Wno-extended-offsetof"
                        CXX_SUPPORTS_NO_EXTENDED_OFFSETOF)
if (CXX_SUPPORTS_NO_EXTENDED_OFFSETOF)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-extended-offsetof")
endif ()

check_library_exists(compression compression_encode_buffer "" HAVE_LIBCOMPRESSION)

add_subdirectory(MacOSX)

set(generated_mach_interfaces
  ${CMAKE_CURRENT_BINARY_DIR}/mach_exc.h
  ${CMAKE_CURRENT_BINARY_DIR}/mach_excServer.c
  ${CMAKE_CURRENT_BINARY_DIR}/mach_excUser.c
  )
add_custom_command(OUTPUT ${generated_mach_interfaces}
  COMMAND mig ${CMAKE_CURRENT_SOURCE_DIR}/MacOSX/dbgnub-mig.defs
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/MacOSX/dbgnub-mig.defs
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

set(lldbDebugserverCommonSources
  DNBArch.cpp
  DNBBreakpoint.cpp
  DNB.cpp
  DNBDataRef.cpp
  DNBError.cpp
  DNBLog.cpp
  DNBRegisterInfo.cpp
  DNBThreadResumeActions.cpp
  JSON.cpp
  StdStringExtractor.cpp
  # JSON reader depends on the following LLDB-common files
  ${LLDB_SOURCE_DIR}/source/Host/common/StringConvert.cpp
  ${LLDB_SOURCE_DIR}/source/Host/common/SocketAddress.cpp
  # end JSON reader dependencies
  libdebugserver.cpp
  PseudoTerminal.cpp
  PThreadEvent.cpp
  PThreadMutex.cpp
  RNBContext.cpp
  RNBRemote.cpp
  RNBServices.cpp
  RNBSocket.cpp
  SysSignal.cpp
  TTYState.cpp
  
  MacOSX/CFBundle.cpp
  MacOSX/CFString.cpp
  MacOSX/Genealogy.cpp
  MacOSX/MachException.cpp
  MacOSX/MachProcess.mm
  MacOSX/MachTask.mm
  MacOSX/MachThread.cpp
  MacOSX/MachThreadList.cpp
  MacOSX/MachVMMemory.cpp
  MacOSX/MachVMRegion.cpp
  MacOSX/OsLogger.cpp
  ${generated_mach_interfaces}
  ${DEBUGSERVER_VERS_GENERATED_FILE})

add_library(lldbDebugserverCommon ${lldbDebugserverCommonSources})


set(LLDB_CODESIGN_IDENTITY "lldb_codesign"
  CACHE STRING "Identity used for code signing. Set to empty string to skip the signing step.")

if(NOT LLDB_CODESIGN_IDENTITY STREQUAL "")
  set(DEBUGSERVER_PATH $<TARGET_FILE:debugserver> CACHE PATH "Path to debugserver.")
  set(SKIP_DEBUGSERVER OFF CACHE BOOL "Skip building the in-tree debug server")
else()
  execute_process(
    COMMAND xcode-select -p
    OUTPUT_VARIABLE XCODE_DEV_DIR)
  string(STRIP ${XCODE_DEV_DIR} XCODE_DEV_DIR)
  set(DEBUGSERVER_PATH
    "${XCODE_DEV_DIR}/../SharedFrameworks/LLDB.framework/Resources/debugserver" CACHE PATH "Path to debugserver.")
  set(SKIP_DEBUGSERVER ON CACHE BOOL "Skip building the in-tree debug server")
endif()
message(STATUS "Path to the lldb debugserver: ${DEBUGSERVER_PATH}")

if (APPLE)
  if(IOS)
    find_library(BACKBOARD_LIBRARY BackBoardServices
      PATHS ${CMAKE_OSX_SYSROOT}/System/Library/PrivateFrameworks)
    find_library(FRONTBOARD_LIBRARY FrontBoardServices
      PATHS ${CMAKE_OSX_SYSROOT}/System/Library/PrivateFrameworks)
    find_library(SPRINGBOARD_LIBRARY SpringBoardServices
      PATHS ${CMAKE_OSX_SYSROOT}/System/Library/PrivateFrameworks)
    find_library(MOBILESERVICES_LIBRARY MobileCoreServices
      PATHS ${CMAKE_OSX_SYSROOT}/System/Library/PrivateFrameworks)
    find_library(LOCKDOWN_LIBRARY lockdown)

    if(NOT BACKBOARD_LIBRARY)
      set(SKIP_DEBUGSERVER ON CACHE BOOL "Skip building the in-tree debug server" FORCE)
    endif()
  else()
    find_library(COCOA_LIBRARY Cocoa)
  endif()
endif()

if(HAVE_LIBCOMPRESSION)
  set(LIBCOMPRESSION compression)
endif()

if(NOT SKIP_DEBUGSERVER)
  target_link_libraries(lldbDebugserverCommon
                        INTERFACE ${COCOA_LIBRARY}
                        ${CORE_FOUNDATION_LIBRARY}
                        ${FOUNDATION_LIBRARY}
                        ${BACKBOARD_LIBRARY}
                        ${FRONTBOARD_LIBRARY}
                        ${SPRINGBOARD_LIBRARY}
                        ${MOBILESERVICES_LIBRARY}
                        ${LOCKDOWN_LIBRARY}
                        lldbDebugserverArchSupport
                        lldbDebugserverDarwin_DarwinLog
                        ${LIBCOMPRESSION})
  if(HAVE_LIBCOMPRESSION)
    set_property(TARGET lldbDebugserverCommon APPEND PROPERTY
                 COMPILE_DEFINITIONS HAVE_LIBCOMPRESSION)
  endif()
  set(LLVM_OPTIONAL_SOURCES ${lldbDebugserverCommonSources})
  add_lldb_tool(debugserver INCLUDE_IN_FRAMEWORK
    debugserver.cpp

    LINK_LIBS
      lldbDebugserverCommon
    )
  if(IOS)
    set_property(TARGET lldbDebugserverCommon APPEND PROPERTY COMPILE_DEFINITIONS
      WITH_LOCKDOWN
      WITH_FBS
      WITH_BKS
      )
    set_property(TARGET debugserver APPEND PROPERTY COMPILE_DEFINITIONS
      WITH_LOCKDOWN
      WITH_FBS
      WITH_BKS
      )
    set_property(TARGET lldbDebugserverCommon APPEND PROPERTY COMPILE_FLAGS
      -F${CMAKE_OSX_SYSROOT}/System/Library/PrivateFrameworks
      )
  endif()
endif()

if(IOS)
  add_library(lldbDebugserverCommon_NonUI ${lldbDebugserverCommonSources})
  target_link_libraries(lldbDebugserverCommon_NonUI
                      INTERFACE ${COCOA_LIBRARY}
                      ${CORE_FOUNDATION_LIBRARY}
                      ${FOUNDATION_LIBRARY}
                      lldbDebugserverArchSupport
                      lldbDebugserverDarwin_DarwinLog
                      ${LIBCOMPRESSION})
  if(HAVE_LIBCOMPRESSION)
    set_property(TARGET lldbDebugserverCommon_NonUI APPEND PROPERTY
                 COMPILE_DEFINITIONS HAVE_LIBCOMPRESSION)
  endif()

  add_lldb_tool(debugserver-nonui
    debugserver.cpp

    LINK_LIBS
      lldbDebugserverCommon_NonUI
    )
endif()

set(entitlements_xml ${CMAKE_CURRENT_SOURCE_DIR}/debugserver-macosx-entitlements.plist)
if(IOS)
  set(entitlements_xml ${CMAKE_CURRENT_SOURCE_DIR}/debugserver-entitlements.plist)
endif()

set(LLDB_USE_ENTITLEMENTS_Default On)
if("${LLDB_CODESIGN_IDENTITY}" STREQUAL "lldb_codesign")
  set(LLDB_USE_ENTITLEMENTS_Default Off)
endif()
option(LLDB_USE_ENTITLEMENTS "Use entitlements when codesigning (Defaults Off when using lldb_codesign identity, otherwise On)" ${LLDB_USE_ENTITLEMENTS_Default})

if (SKIP_DEBUGSERVER)
  if (CMAKE_HOST_APPLE)
    # If we haven't built a signed debugserver, copy the one from the system.
    add_custom_target(debugserver
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DEBUGSERVER_PATH} ${CMAKE_BINARY_DIR}/bin
      VERBATIM
      COMMENT "Copying the system debugserver to LLDB's binaries directory.")
  endif()
else()
  if(LLDB_USE_ENTITLEMENTS)
    set(entitlements_flags --entitlements ${entitlements_xml})
  endif()
  execute_process(
    COMMAND xcrun -f codesign_allocate
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE CODESIGN_ALLOCATE
    )
  add_custom_command(TARGET debugserver
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E env CODESIGN_ALLOCATE=${CODESIGN_ALLOCATE}
            codesign --force --sign ${LLDB_CODESIGN_IDENTITY}
            ${entitlements_flags}
            $<TARGET_FILE:debugserver>
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  if(IOS)
    add_custom_command(TARGET debugserver-nonui
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E env CODESIGN_ALLOCATE=${CODESIGN_ALLOCATE}
              codesign --force --sign ${LLDB_CODESIGN_IDENTITY}
              ${entitlements_flags}
              $<TARGET_FILE:debugserver>
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
  endif()
endif()




