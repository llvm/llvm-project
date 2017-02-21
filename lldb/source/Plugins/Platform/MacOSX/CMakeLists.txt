list(APPEND PLUGIN_PLATFORM_MACOSX_SOURCES
  PlatformDarwin.cpp
  PlatformDarwinKernel.cpp
  PlatformMacOSX.cpp
  PlatformRemoteiOS.cpp
  PlatformRemoteAppleTV.cpp
  PlatformRemoteAppleWatch.cpp
  )

list(APPEND PLUGIN_PLATFORM_MACOSX_DARWIN_ONLY_SOURCES
  PlatformAppleSimulator.cpp
  PlatformiOSSimulator.cpp
  PlatformiOSSimulatorCoreSimulatorSupport.mm
  PlatformAppleTVSimulator.cpp
  PlatformAppleWatchSimulator.cpp
  )

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  include_directories(${LIBXML2_INCLUDE_DIR})
  list(APPEND PLUGIN_PLATFORM_MACOSX_SOURCES
    ${PLUGIN_PLATFORM_MACOSX_DARWIN_ONLY_SOURCES})
else()
  list(APPEND LLVM_OPTIONAL_SOURCES
    ${PLUGIN_PLATFORM_MACOSX_DARWIN_ONLY_SOURCES})
endif()

add_lldb_library(lldbPluginPlatformMacOSX PLUGIN
  ${PLUGIN_PLATFORM_MACOSX_SOURCES}
  
  LINK_LIBS
    clangBasic
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginPlatformPOSIX
  LINK_COMPONENTS
    Support
    )
