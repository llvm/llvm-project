add_llvm_library(gtest_main
  TestMain.cpp

  LINK_LIBS
  gtest

  LINK_COMPONENTS
  Support # Depends on llvm::cl
  )
