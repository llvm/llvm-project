lldb_tablegen(JITLoaderROARProperties.inc -gen-lldb-property-defs
  SOURCE JITLoaderROARProperties.td
  TARGET LLDBPluginJITLoaderROARPropertiesGen)

lldb_tablegen(JITLoaderROARPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE JITLoaderROARProperties.td
  TARGET LLDBPluginJITLoaderROARPropertiesEnumGen)

add_lldb_library(lldbPluginJITLoaderROAR PLUGIN
  JITLoaderROAR.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginObjectFileMachO
  LINK_COMPONENTS
    Support
  )

add_dependencies(lldbPluginJITLoaderROAR
  LLDBPluginJITLoaderROARPropertiesGen
  LLDBPluginJITLoaderROARPropertiesEnumGen)

target_include_directories(lldbPluginJITLoaderROAR PRIVATE "${CMAKE_SOURCE_DIR}/../roar/include")
