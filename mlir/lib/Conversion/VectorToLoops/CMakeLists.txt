add_mlir_conversion_library(MLIRVectorToLoops
  ConvertVectorToLoops.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToLoops

  LINK_LIBS
  MLIRAffineOps
  MLIREDSC
  MLIRIR
  MLIRLLVMIR
  MLIRLoopOps
  MLIRStandardOps
  MLIRTransforms
  MLIRTransformUtils
  MLIRVectorOps
  LLVMCore
  LLVMSupport
  )
