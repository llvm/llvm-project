set(SWIFT_SOURCES HostInfoMacOSXSwift.cpp)
set(LLVM_OPTIONAL_SOURCES ${SWIFT_SOURCES})
if (NOT LLDB_ENABLE_SWIFT_SUPPORT)
  unset(SWIFT_SOURCES)
  set(PLUGIN_DEPENDENCY_ARG NO_PLUGIN_DEPENDENCIES)
  set(LLDB_PLUGIN_DEPENDENCIES)
else()
  set(PLUGIN_DEPENDENCY_ARG)
  set(LLDB_PLUGIN_DEPENDENCIES lldbPluginPlatformMacOSX)
endif()

remove_module_flags()
include_directories(..)

add_lldb_library(lldbHostMacOSXObjCXX ${PLUGIN_DEPENDENCY_ARG}
  Host.mm
  HostInfoMacOSX.mm
  HostThreadMacOSX.mm

  ${SWIFT_SOURCES}

  LINK_LIBS
    lldbUtility
    ${LLDB_PLUGIN_DEPENDENCIES}
    ${EXTRA_LIBS}

  LINK_COMPONENTS
    Support
    TargetParser
  )

target_compile_options(lldbHostMacOSXObjCXX PRIVATE
  -fno-objc-exceptions
  -Wno-deprecated-declarations)
