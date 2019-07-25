lldb_tablegen(Properties.inc -gen-lldb-property-defs
  SOURCE Properties.td
  TARGET LLDBPluginDynamicLoaderDarwinKernelPropertiesGen)

lldb_tablegen(PropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE Properties.td
  TARGET LLDBPluginDynamicLoaderDarwinKernelPropertiesEnumGen)

add_lldb_library(lldbPluginDynamicLoaderDarwinKernel PLUGIN
  DynamicLoaderDarwinKernel.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginPlatformMacOSX
  )

add_dependencies(lldbPluginDynamicLoaderDarwinKernel
  LLDBPluginDynamicLoaderDarwinKernelPropertiesGen
  LLDBPluginDynamicLoaderDarwinKernelPropertiesEnumGen)
