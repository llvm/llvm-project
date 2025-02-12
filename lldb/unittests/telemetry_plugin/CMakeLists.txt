if (LLVM_BUILD_TELEMETRY)
  add_lldb_library(lldbPluginTelemetryFakePlugin PLUGIN
                   FakePlugin.cpp

  LINK_LIBS
    lldbCore
    lldbUtility
    lldbPluginProcessUtility
  LINK_COMPONENTS
    Support
    Telemetry                   
  )

  add_lldb_unittest(TestFakePlugin
  TestFakePlugin.cpp

  LINK_LIBS
    lldbPluginTelemetryFakePlugin
    lldbCore
    lldbUtilityHelpers
    LLVMTestingSupport

  LINK_COMPONENTS
    Support
    Telemetry                   
  )
endif()
