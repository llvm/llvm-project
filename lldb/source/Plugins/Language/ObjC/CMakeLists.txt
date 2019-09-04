set(EXTRA_CXXFLAGS "")

if (CXX_SUPPORTS_NO_GNU_ANONYMOUS_STRUCT)
  set(EXTRA_CXXFLAGS ${EXTRA_CXXFLAGS} -Wno-gnu-anonymous-struct)
endif ()

if (CXX_SUPPORTS_NO_NESTED_ANON_TYPES)
  set(EXTRA_CXXFLAGS ${EXTRA_CXXFLAGS} -Wno-nested-anon-types)
endif ()

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
    lldbPluginClangCommon
    lldbPluginSwiftLanguage
    lldbPluginClangCommon

  EXTRA_CXXFLAGS ${EXTRA_CXXFLAGS}
)
