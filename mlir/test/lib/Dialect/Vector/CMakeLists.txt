# Exclude tests from libMLIR.so
add_mlir_library(MLIRVectorTestPasses
  TestVectorTransforms.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAnalysis
  MLIRLinalg
  MLIRLinalgTransforms
  MLIRLLVMIR
  MLIRMemRef
  MLIRPass
  MLIRSCF
  MLIRStandard
  MLIRTransformUtils
  MLIRVector
  MLIRVectorToSCF
  MLIRX86Vector
  )
