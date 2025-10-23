set(BUILDTREE_ONLY BUILDTREE_ONLY)
if (LLVM_INSTALL_GTEST)
  set(BUILDTREE_ONLY "")
endif ()

add_llvm_library("${gtest_name}_main"
  TestMain.cpp

  LINK_LIBS
  "${gtest_name}"

  LINK_COMPONENTS
  Support # Depends on llvm::cl

  ${BUILDTREE_ONLY}
  )
