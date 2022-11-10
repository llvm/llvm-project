add_llvm_library(llvm_gtest_main
  TestMain.cpp

  LINK_LIBS
  llvm_gtest

  LINK_COMPONENTS
  Support # Depends on llvm::cl

  # This library is not meant to be in the install tree, only the build tree.
  BUILDTREE_ONLY
  )
