add_lldb_unittest(LanguageRuntimeObjCGNUstepTests
  GNUstepObjCClassDescriptorTest.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginGNUstepObjCRuntime
    lldbPluginPlatformLinux
    lldbPluginPlatformWindows
  )
