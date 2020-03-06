file(GLOB globbed *.c *.cpp)
add_llvm_library(MLIRPass
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Pass
  )
target_link_libraries(MLIRPass
  PUBLIC
  MLIRAnalysis
  MLIRIR
  LLVMSupport)
