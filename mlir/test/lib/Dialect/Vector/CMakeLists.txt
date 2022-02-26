# Exclude tests from libMLIR.so
add_mlir_library(MLIRVectorTestPasses
  TestVectorTransforms.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAnalysis
  MLIRFunc
  MLIRLinalg
  MLIRLinalgTransforms
  MLIRLLVMIR
  MLIRMemRef
  MLIRPass
  MLIRSCF
  MLIRTransformUtils
  MLIRVector
  MLIRVectorToSCF
  MLIRX86Vector
  )
