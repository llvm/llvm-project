set(SWIFT_SOURCES
  SwiftDemanglingPartsTest.cpp
)

set(LLVM_OPTIONAL_SOURCES ${SWIFT_SOURCES})
if (NOT LLDB_ENABLE_SWIFT_SUPPORT)
  unset(SWIFT_SOURCES)
endif()

add_lldb_unittest(LLDBCoreTests
  DebuggerTest.cpp
  CommunicationTest.cpp
  DiagnosticEventTest.cpp
  DumpDataExtractorTest.cpp
  DumpRegisterInfoTest.cpp
  FormatEntityTest.cpp
  MangledTest.cpp
  ModuleSpecTest.cpp
  PluginManagerTest.cpp
  ProgressReportTest.cpp
  RichManglingContextTest.cpp
  SourceLocationSpecTest.cpp
  SourceManagerTest.cpp
  SwiftDemanglingPartsTest.cpp
  TelemetryTest.cpp
  UniqueCStringMapTest.cpp

  ${SWIFT_SOURCES}

  LINK_COMPONENTS
    Support
    Telemetry
  LINK_LIBS
    lldbCore
    lldbHost
    lldbPluginCPlusPlusLanguage
    lldbPluginObjectFileELF
    lldbPluginObjectFileMachO
    lldbPluginObjectFilePECOFF
    lldbPluginPlatformMacOSX
    lldbPluginSymbolFileSymtab
    lldbSymbol
    lldbUtilityHelpers
    lldbHostHelpers
    LLVMTestingSupport
  )
