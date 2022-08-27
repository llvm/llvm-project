add_mlir_library(MLIRTilingInterfaceTestPasses
  TestTilingInterface.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithmeticDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTensorTilingInterfaceImpl
  )
