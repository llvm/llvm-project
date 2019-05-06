set(EXCLUDE_FROM_ALL ON)
add_lldb_library(lldbUtilityHelpers
  MockTildeExpressionResolver.cpp
  TestUtilities.cpp

  LINK_LIBS
    lldbUtility

  LINK_COMPONENTS
    Support
  )

add_dependencies(lldbUtilityHelpers yaml2obj)
add_definitions(-DYAML2OBJ="$<TARGET_FILE:yaml2obj>")