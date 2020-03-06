file(GLOB globbed *.c *.cpp)
add_mlir_dialect_library(MLIRLoopOps
  ${globbed}
  EDSC/Builders.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/LoopOps
  )
add_dependencies(MLIRLoopOps
  MLIRLoopLikeInterfaceIncGen
  MLIRLoopOpsIncGen
  )
target_link_libraries(MLIRLoopOps
  PUBLIC
  MLIREDSC
  MLIRIR
  MLIRStandardOps
  LLVMSupport
  )

add_subdirectory(Transforms)
