add_mlir_library(MLIRLoopLikeInterfaceTestPasses
  TestBlockInLoop.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRLoopLikeInterface
  )
