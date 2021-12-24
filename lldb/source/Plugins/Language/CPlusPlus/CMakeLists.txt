add_lldb_library(lldbPluginCPlusPlusLanguage PLUGIN
  BlockPointer.cpp
  CPlusPlusLanguage.cpp
  CPlusPlusNameParser.cpp
  CxxStringTypes.cpp
  GenericBitset.cpp
  GenericOptional.cpp
  LibCxx.cpp
  LibCxxAtomic.cpp
  LibCxxInitializerList.cpp
  LibCxxList.cpp
  LibCxxMap.cpp
  LibCxxQueue.cpp
  LibCxxTuple.cpp
  LibCxxUnorderedMap.cpp
  LibCxxVariant.cpp
  LibCxxVector.cpp
  LibStdcpp.cpp
  LibStdcppTuple.cpp
  LibStdcppUniquePointer.cpp
  MSVCUndecoratedNameParser.cpp

  LINK_LIBS
    lldbCore
    lldbDataFormatters
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginClangCommon
    lldbPluginCPPRuntime
    lldbPluginTypeSystemClang

  LINK_COMPONENTS
    Support
)
