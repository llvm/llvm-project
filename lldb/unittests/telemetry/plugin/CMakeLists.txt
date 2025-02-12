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
endif()
