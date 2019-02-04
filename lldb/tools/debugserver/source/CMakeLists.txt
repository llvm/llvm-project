include(CheckCXXCompilerFlag)
include(CheckLibraryExists)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/..)
include_directories(${LLDB_SOURCE_DIR}/source)
include_directories(MacOSX/DarwinLog)

include_directories(MacOSX)

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

# LLDB-specific identity, currently used for code signing debugserver.
set(LLDB_CODESIGN_IDENTITY "" CACHE STRING
    "Override code sign identity for debugserver and for use in tests; falls back to LLVM_CODESIGNING_IDENTITY if set or lldb_codesign otherwise (Darwin only)")

# Determine which identity to use and store it in the separate cache entry.
# We will query it later for LLDB_TEST_COMMON_ARGS.
if(LLDB_CODESIGN_IDENTITY)
  set(LLDB_CODESIGN_IDENTITY_USED ${LLDB_CODESIGN_IDENTITY} CACHE INTERNAL "" FORCE)
elseif(LLVM_CODESIGNING_IDENTITY)
  set(LLDB_CODESIGN_IDENTITY_USED ${LLVM_CODESIGNING_IDENTITY} CACHE INTERNAL "" FORCE)
else()
  set(LLDB_CODESIGN_IDENTITY_USED lldb_codesign CACHE INTERNAL "" FORCE)
endif()

# Override locally, so the identity is used for targets created in this scope.
set(LLVM_CODESIGNING_IDENTITY ${LLDB_CODESIGN_IDENTITY_USED})

option(LLDB_NO_DEBUGSERVER "Disable the debugserver target" OFF)
option(LLDB_USE_SYSTEM_DEBUGSERVER "Use the system's debugserver instead of building it from source (Darwin only)." OFF)

# Incompatible options
if(LLDB_NO_DEBUGSERVER AND LLDB_USE_SYSTEM_DEBUGSERVER)
  message(FATAL_ERROR "Inconsistent options: LLDB_NO_DEBUGSERVER and LLDB_USE_SYSTEM_DEBUGSERVER")
endif()

# Try to locate the system debugserver.
# Subsequent feasibility checks depend on it.
if(APPLE AND CMAKE_HOST_APPLE)
  execute_process(
    COMMAND xcode-select -p
    OUTPUT_VARIABLE xcode_dev_dir)
  string(STRIP ${xcode_dev_dir} xcode_dev_dir)

  set(debugserver_rel_path "LLDB.framework/Resources/debugserver")
  set(debugserver_shared "${xcode_dev_dir}/../SharedFrameworks/${debugserver_rel_path}")
  set(debugserver_private "${xcode_dev_dir}/Library/PrivateFrameworks/${debugserver_rel_path}")

  if(EXISTS ${debugserver_shared})
    set(system_debugserver ${debugserver_shared})
  elseif(EXISTS ${debugserver_private})
    set(system_debugserver ${debugserver_private})
  endif()
endif()

# Handle unavailability
if(LLDB_USE_SYSTEM_DEBUGSERVER)
  if(system_debugserver)
    set(use_system_debugserver ON)
  elseif(APPLE AND CMAKE_HOST_APPLE)
    # Binary not found on system. Keep cached variable, to try again on reconfigure.
    message(SEND_ERROR
      "LLDB_USE_SYSTEM_DEBUGSERVER option set, but no debugserver found in:\
        ${debugserver_shared}\
        ${debugserver_private}")
  else()
    # Non-Apple target platform or non-Darwin host. Reset invalid cached variable.
    message(WARNING "Reverting invalid option LLDB_USE_SYSTEM_DEBUGSERVER (Darwin only)")
    set(LLDB_USE_SYSTEM_DEBUGSERVER OFF CACHE BOOL "" FORCE)
  endif()
elseif(NOT LLDB_NO_DEBUGSERVER)
  # Default case: on Darwin we need the right code signing ID.
  # See lldb/docs/code-signing.txt for details.
  if(CMAKE_HOST_APPLE AND NOT LLVM_CODESIGNING_IDENTITY STREQUAL "lldb_codesign")
    set(problem "Cannot code sign debugserver with LLVM_CODESIGNING_IDENTITY '${LLVM_CODESIGNING_IDENTITY}'.")
    set(advice "Pass -DLLDB_CODESIGN_IDENTITY=lldb_codesign to override the LLVM value for debugserver.")
    if(system_debugserver)
      set(effect "Will fall back to system's debugserver.")
      set(use_system_debugserver ON)
    else()
      set(effect "debugserver will not be available.")
    endif()
    message(WARNING "${problem} ${effect} ${advice}")
  else()
    set(build_and_sign_debugserver ON)
  endif()
endif()

# TODO: We don't use the $<TARGET_FILE:debugserver> generator expression here,
# because the value of DEBUGSERVER_PATH is used to build LLDB_DOTEST_ARGS,
# which is used for configuring lldb-dotest.in, which does not have a generator
# step at the moment.
set(default_debugserver_path "${LLVM_RUNTIME_OUTPUT_INTDIR}/debugserver${CMAKE_EXECUTABLE_SUFFIX}")

# Remember where debugserver binary goes and whether or not we have to test it.
set(DEBUGSERVER_PATH "" CACHE FILEPATH "Path to debugserver")
set(SKIP_TEST_DEBUGSERVER OFF CACHE BOOL "Building the in-tree debugserver was skipped")

# Reset values in all cases in order to correctly support reconfigurations.
if(use_system_debugserver)
  add_custom_target(debugserver
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${system_debugserver} ${LLVM_RUNTIME_OUTPUT_INTDIR}
    COMMENT "Copying the system debugserver to LLDB's binaries directory.")

  # Don't test debugserver itself.
  # Tests that require debugserver will use the copy.
  set(DEBUGSERVER_PATH ${default_debugserver_path} CACHE FILEPATH "" FORCE)
  set(SKIP_TEST_DEBUGSERVER ON CACHE BOOL "" FORCE)

  message(STATUS "Copy system debugserver from: ${system_debugserver}")
elseif(build_and_sign_debugserver)
  # Build, sign and test debugserver (below)
  set(DEBUGSERVER_PATH ${default_debugserver_path} CACHE FILEPATH "" FORCE)
  set(SKIP_TEST_DEBUGSERVER OFF CACHE BOOL "" FORCE)

  message(STATUS "lldb debugserver: ${DEBUGSERVER_PATH}")
else()
  # No tests for debugserver, no tests that require it.
  set(DEBUGSERVER_PATH "" CACHE FILEPATH "" FORCE)
  set(SKIP_TEST_DEBUGSERVER ON CACHE BOOL "" FORCE)

  message(STATUS "lldb debugserver will not be available.")
endif()

if(APPLE)
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
      set(SKIP_TEST_DEBUGSERVER ON CACHE BOOL "" FORCE)
    endif()
  else()
    find_library(COCOA_LIBRARY Cocoa)
  endif()
endif()

if(HAVE_LIBCOMPRESSION)
  set(LIBCOMPRESSION compression)
endif()

if(LLDB_USE_ENTITLEMENTS)
  if(IOS)
    set(entitlements ${CMAKE_CURRENT_SOURCE_DIR}/debugserver-entitlements.plist)
  else()
    # Same entitlements file as used for lldb-server
    set(entitlements ${LLDB_SOURCE_DIR}/resources/debugserver-macosx-entitlements.plist)
  endif()
endif()

if(build_and_sign_debugserver)
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
  add_lldb_tool(debugserver
    debugserver.cpp

    LINK_LIBS
      lldbDebugserverCommon

    ENTITLEMENTS
      ${entitlements}
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

      ENTITLEMENTS
        ${entitlements}
      )
  endif()
endif()
