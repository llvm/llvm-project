set(BUILDTREE_ONLY BUILDTREE_ONLY)
if (LLVM_INSTALL_GTEST)
  set(BUILDTREE_ONLY "")
endif ()

add_llvm_library(llvm_gtest_main
  TestMain.cpp

  LINK_LIBS
  llvm_gtest

  LINK_COMPONENTS
  Support # Depends on llvm::cl

  ${BUILDTREE_ONLY}
  )
