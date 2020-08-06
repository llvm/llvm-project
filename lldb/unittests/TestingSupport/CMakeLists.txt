# Gtest 1.8.0 uses tr1/tuple which is deprecated on MSVC, so we force it off.
add_definitions(-DGTEST_HAS_TR1_TUPLE=0)

set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL ON)
add_lldb_library(lldbUtilityHelpers
  MockTildeExpressionResolver.cpp
  TestUtilities.cpp

  LINK_LIBS
    lldbUtility
    lldbSymbolHelpers

  LINK_COMPONENTS
    Support
    ObjectYAML
  )
include_directories(${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include)

add_subdirectory(Symbol)
