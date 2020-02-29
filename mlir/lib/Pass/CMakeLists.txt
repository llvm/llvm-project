file(GLOB globbed *.c *.cpp)
add_mlir_library(MLIRPass
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Pass

  LINK_LIBS
  MLIRAnalysis
  MLIRIR
  MLIRSupport
  LLVMIRReader
  LLVMSupport
  )
