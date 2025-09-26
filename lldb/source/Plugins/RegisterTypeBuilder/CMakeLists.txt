set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND RegisterTypeBuilder)

add_lldb_library(lldbPluginRegisterTypeBuilderClang PLUGIN
  RegisterTypeBuilderClang.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbTarget
  )
