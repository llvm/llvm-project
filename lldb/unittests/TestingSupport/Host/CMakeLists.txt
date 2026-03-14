set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL ON)
add_lldb_library(lldbHostHelpers
  SocketTestUtilities.cpp

  LINK_LIBS
    lldbHost
    LLVMTestingSupport
    llvm_gtest
  )
