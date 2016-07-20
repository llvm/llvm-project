set(LLVM_NO_RTTI 1)

add_lldb_library(lldbPluginSwiftLanguage
  FoundationValueTypes.cpp
  ObjCRuntimeSyntheticProvider.cpp
  SwiftArray.cpp
  SwiftBasicTypes.cpp
  SwiftDictionary.cpp
  SwiftFormatters.cpp
  SwiftHashedContainer.cpp
  SwiftLanguage.cpp
  SwiftMetatype.cpp
  SwiftOptionSet.cpp
  SwiftOptional.cpp
  SwiftSet.cpp
)

