add_mlir_library(MLIRTilingInterfaceTestPasses
  TestTilingInterface.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRArithmetic
  MLIRLinalg
  MLIRLinalgTransforms
  MLIRMemRef
  MLIRSCF
  MLIRSCFTransforms
  MLIRTensor
  )
