set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL ON)
add_lldb_library(lldbSBUtilityHelpers
  SBTestUtilities.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbHost
    lldbUtility
    lldbUtilityHelpers
    llvm_gtest
    liblldb
  )
