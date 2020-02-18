file(GLOB globbed *.c *.cpp)
add_llvm_library(MLIRPass
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
add_dependencies(MLIRPass MLIRAnalysis MLIRIR LLVMSupport)
