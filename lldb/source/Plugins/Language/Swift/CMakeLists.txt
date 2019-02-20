set(LLVM_NO_RTTI 1)

add_lldb_library(lldbPluginSwiftLanguage PLUGIN
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

  LINK_LIBS
    lldbCore
    lldbDataFormatters
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginObjCLanguage
    swiftAST
    swiftClangImporter
    clangAST

  LINK_COMPONENTS
    Support
)
if(CMAKE_CXX_COMPILER_ID STREQUAL Clang AND NOT SWIFT_COMPILER_MSVC_LIKE)
  target_compile_options(lldbPluginSwiftLanguage PRIVATE
    -Wno-dollar-in-identifier-extension)
endif()

