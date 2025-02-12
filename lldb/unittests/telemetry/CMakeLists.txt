if (LLVM_BUILD_TELEMETRY)
   add_subdirectory(plugin)

  add_lldb_unittest(FakePluginTests
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
