file(GLOB globbed *.c *.cpp)
add_llvm_library(MLIRLoopOps
  ${globbed}
  EDSC/Builders.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/LoopOps
  )
add_dependencies(MLIRLoopOps

  MLIREDSC
  MLIRLoopLikeInterfaceIncGen
  MLIRLoopOpsIncGen
  MLIRStandardOps
  LLVMSupport
  )
target_link_libraries(MLIRLoopOps

  MLIREDSC
  MLIRIR
  MLIRStandardOps
  LLVMSupport
  )

add_subdirectory(Transforms)
