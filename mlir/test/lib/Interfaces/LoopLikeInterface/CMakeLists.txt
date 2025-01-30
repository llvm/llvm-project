add_mlir_library(MLIRLoopLikeInterfaceTestPasses
  TestBlockInLoop.cpp

  EXCLUDE_FROM_LIBMLIR
  )
mlir_target_link_libraries(MLIRLoopLikeInterfaceTestPasses PUBLIC
  MLIRPass
  MLIRLoopLikeInterface
  MLIRFuncDialect
  )
