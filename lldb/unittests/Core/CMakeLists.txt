if (LLVM_BUILD_TELEMETRY)
  set(TELEMETRY_DEPS Telemetry)
endif()

add_lldb_unittest(LLDBCoreTests
  CommunicationTest.cpp
  DiagnosticEventTest.cpp
  DumpDataExtractorTest.cpp
  DumpRegisterInfoTest.cpp
  FormatEntityTest.cpp
  MangledTest.cpp
  ModuleSpecTest.cpp
  ProgressReportTest.cpp
  RichManglingContextTest.cpp
  SourceLocationSpecTest.cpp
  SourceManagerTest.cpp
  TelemetryTest.cpp
  UniqueCStringMapTest.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbPluginObjectFileELF
    lldbPluginObjectFileMachO
    lldbPluginObjectFilePECOFF
    lldbPluginPlatformMacOSX
    lldbPluginSymbolFileSymtab
    lldbSymbol
    lldbUtilityHelpers
    lldbHostHelpers
    LLVMTestingSupport
  LINK_COMPONENTS
    Support
    ${TELEMETRY_DEPS}
  )
