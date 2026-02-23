lldb_tablegen(ProcessKDPProperties.inc -gen-lldb-property-defs
  SOURCE ProcessKDPProperties.td
  TARGET LLDBPluginProcessMacOSXKernelPropertiesGen)

lldb_tablegen(ProcessKDPPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE ProcessKDPProperties.td
  TARGET LLDBPluginProcessMacOSXKernelPropertiesEnumGen)

lldb_tablegen(ProcessKDPProperties.json -dump-json
  SOURCE ProcessKDPProperties.td
  TARGET LLDBPluginProcessMacOSXKernelPropertiesJsonGen)

if (NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
  return()
endif()

add_lldb_library(lldbPluginProcessMacOSXKernel PLUGIN
  CommunicationKDP.cpp
  ProcessKDP.cpp
  ProcessKDPLog.cpp
  RegisterContextKDP_arm.cpp
  RegisterContextKDP_arm64.cpp
  RegisterContextKDP_x86_64.cpp
  ThreadKDP.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginDynamicLoaderDarwinKernel
    lldbPluginDynamicLoaderStatic
    lldbPluginProcessUtility
  )

add_dependencies(lldbPluginProcessMacOSXKernel
  LLDBPluginProcessMacOSXKernelPropertiesGen
  LLDBPluginProcessMacOSXKernelPropertiesEnumGen)
