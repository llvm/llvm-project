add_lldb_library(lldbPluginDynamicLoaderWindowsDYLD PLUGIN
  DynamicLoaderWindowsDYLD.cpp
  MSVCRTCFrameRecognizer.cpp

  LINK_COMPONENTS
    Support
    TargetParser
  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbValueObject
  )
