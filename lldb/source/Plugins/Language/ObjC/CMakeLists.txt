add_lldb_library(lldbPluginObjCLanguage PLUGIN
  ObjCLanguage.cpp
  CF.cpp
  CFBasicHash.cpp
  Cocoa.cpp
  CoreMedia.cpp
  NSArray.cpp
  NSDictionary.cpp
  NSError.cpp
  NSException.cpp
  NSIndexPath.cpp
  NSSet.cpp
  NSString.cpp

  LINK_LIBS
    lldbCore
    lldbDataFormatters
    lldbExpression
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbValueObject
    lldbPluginAppleObjCRuntime
    lldbPluginClangCommon
    lldbPluginTypeSystemClang
  CLANG_LIBS
    clangAST
)

if (CXX_SUPPORTS_NO_GNU_ANONYMOUS_STRUCT)
  target_compile_options(lldbPluginObjCLanguage
    PRIVATE -Wno-gnu-anonymous-struct)
endif ()

if (CXX_SUPPORTS_NO_NESTED_ANON_TYPES)
  target_compile_options(lldbPluginObjCLanguage
    PRIVATE -Wno-nested-anon-types)
endif ()
