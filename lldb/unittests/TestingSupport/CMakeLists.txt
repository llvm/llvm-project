set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL ON)
add_lldb_library(lldbUtilityHelpers
  MockTildeExpressionResolver.cpp
  TestUtilities.cpp

  LINK_COMPONENTS
    Support
    ObjectYAML
    TestingSupport
  LINK_LIBS
    lldbUtility
    liblldb
    llvm_gtest
  )

add_subdirectory(Host)
add_subdirectory(Symbol)
