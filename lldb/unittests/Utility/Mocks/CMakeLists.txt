add_lldb_library(lldbUtilityMocks
  MockTildeExpressionResolver.cpp

  LINK_LIBS
    lldbUtility

  LINK_COMPONENTS
    Support
  )
