add_lldb_library(lldbPluginDynamicLoaderWasmDYLD PLUGIN
  DynamicLoaderWasmDYLD.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbTarget
  )
