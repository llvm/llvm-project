add_lldb_library(lldbPluginCPlusPlusLanguage PLUGIN
  BlockPointer.cpp
  CPlusPlusLanguage.cpp
  CPlusPlusNameParser.cpp
  CxxStringTypes.cpp
  LibCxx.cpp
  LibCxxAtomic.cpp
  LibCxxInitializerList.cpp
  LibCxxList.cpp
  LibCxxMap.cpp
  LibCxxUnorderedMap.cpp
  LibCxxVector.cpp
  LibStdcpp.cpp
  LibStdcppTuple.cpp
  LibStdcppUniquePointer.cpp

  LINK_LIBS
    lldbCore
    lldbDataFormatters
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
  LINK_COMPONENTS
    Support
)
