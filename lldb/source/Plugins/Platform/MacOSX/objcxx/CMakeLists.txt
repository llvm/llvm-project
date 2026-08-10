remove_module_flags()
include_directories(.)

add_lldb_library(lldbPluginPlatformMacOSXObjCXX
  PlatformiOSSimulatorCoreSimulatorSupport.mm

  LINK_COMPONENTS
    Object
    Support
  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbUtility
    ${EXTRA_LIBS}
  )


target_compile_options(lldbPluginPlatformMacOSXObjCXX PRIVATE -fno-objc-exceptions)
