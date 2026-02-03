lldb_tablegen(DynamicLoaderDarwinKernelProperties.inc -gen-lldb-property-defs
  SOURCE DynamicLoaderDarwinKernelProperties.td
  TARGET LLDBPluginDynamicLoaderDarwinKernelPropertiesGen)

lldb_tablegen(DynamicLoaderDarwinKernelPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE DynamicLoaderDarwinKernelProperties.td
  TARGET LLDBPluginDynamicLoaderDarwinKernelPropertiesEnumGen)

lldb_tablegen(DynamicLoaderDarwinKernelProperties.json -gen-lldb-property-docs-json
  SOURCE DynamicLoaderDarwinKernelProperties.td
  TARGET LLDBPluginDynamicLoaderDarwinKernelPropertiesJsonGen)

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
    lldbPluginObjectFileMachO
  )

add_dependencies(lldbPluginDynamicLoaderDarwinKernel
  LLDBPluginDynamicLoaderDarwinKernelPropertiesGen
  LLDBPluginDynamicLoaderDarwinKernelPropertiesEnumGen)
