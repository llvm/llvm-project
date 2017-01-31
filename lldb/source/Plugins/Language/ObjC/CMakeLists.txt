add_lldb_library(lldbPluginObjCLanguage PLUGIN
  ObjCLanguage.cpp
  CF.cpp
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
    clangAST
    lldbCore
    lldbDataFormatters
    lldbExpression
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginAppleObjCRuntime
)
