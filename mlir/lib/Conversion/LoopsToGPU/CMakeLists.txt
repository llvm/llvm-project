set(LIBS
  MLIRAffineOps
  MLIRAffineToStandard
  MLIRGPU
  MLIRIR
  MLIRLinalgOps
  MLIRLoopOps
  MLIRPass
  MLIRStandardOps
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
  LLVMSupport
)

add_mlir_conversion_library(MLIRLoopsToGPU
  LoopsToGPU.cpp
  LoopsToGPUPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LoopsToGPU

  LINK_LIBS
  ${LIBS}
  )
add_dependencies(MLIRLoopsToGPU ${LIBS})
