lldb_tablegen(LanguageCPlusPlusProperties.inc -gen-lldb-property-defs
  SOURCE LanguageCPlusPlusProperties.td
  TARGET LLDBPluginLanguageCPlusPlusPropertiesGen)

lldb_tablegen(LanguageCPlusPlusPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE LanguageCPlusPlusProperties.td
  TARGET LLDBPluginLanguageCPlusPlusPropertiesEnumGen)

add_lldb_library(lldbPluginCPlusPlusLanguage PLUGIN
  BlockPointer.cpp
  Coroutines.cpp
  CPlusPlusLanguage.cpp
  CPlusPlusNameParser.cpp
  CxxStringTypes.cpp
  Generic.cpp
  GenericBitset.cpp
  GenericList.cpp
  GenericOptional.cpp
  LibCxx.cpp
  LibCxxAtomic.cpp
  LibCxxInitializerList.cpp
  LibCxxMap.cpp
  LibCxxQueue.cpp
  LibCxxRangesRefView.cpp
  LibCxxSliceArray.cpp
  LibCxxProxyArray.cpp
  LibCxxSpan.cpp
  LibCxxTuple.cpp
  LibCxxUnorderedMap.cpp
  LibCxxVariant.cpp
  LibCxxValarray.cpp
  LibCxxVector.cpp
  LibStdcpp.cpp
  LibStdcppTuple.cpp
  LibStdcppUniquePointer.cpp
  MsvcStl.cpp
  MsvcStlSmartPointer.cpp
  MsvcStlTuple.cpp
  MsvcStlVariant.cpp
  MsvcStlVector.cpp
  MSVCUndecoratedNameParser.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbDataFormatters
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbValueObject
    lldbPluginClangCommon
    lldbPluginCPPRuntime
    lldbPluginTypeSystemClang
)

add_dependencies(lldbPluginCPlusPlusLanguage
  LLDBPluginLanguageCPlusPlusPropertiesGen
  LLDBPluginLanguageCPlusPlusPropertiesEnumGen)
