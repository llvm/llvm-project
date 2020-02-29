file(GLOB globbed *.c *.cpp)
add_mlir_dialect_library(MLIRLoopOps
  ${globbed}
  EDSC/Builders.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/LoopOps

  LINK_LIBS
  MLIREDSC
  MLIRIR
  MLIRStandardOps
  LLVMSupport
  )
add_dependencies(MLIRLoopOps
  MLIRLoopLikeInterfaceIncGen
  MLIRLoopOpsIncGen
  )

add_subdirectory(Transforms)
