set(LLVM_TARGET_DEFINITIONS TestTilingInterfaceTransformOps.td)
mlir_tablegen(TestTilingInterfaceTransformOps.h.inc -gen-op-decls)
mlir_tablegen(TestTilingInterfaceTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTestTilingInterfaceTransformOpsIncGen)

add_mlir_library(MLIRTilingInterfaceTestPasses
  TestTilingInterfaceTransformOps.cpp

  DEPENDS
  MLIRTestTilingInterfaceTransformOpsIncGen

  EXCLUDE_FROM_LIBMLIR
  )
mlir_target_link_libraries(MLIRTilingInterfaceTestPasses PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRIndexDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTensorTilingInterfaceImpl
  MLIRTransformDialect
  )
