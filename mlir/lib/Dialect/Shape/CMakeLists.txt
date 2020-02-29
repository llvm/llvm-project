file(GLOB globbed *.c *.cpp)
add_mlir_dialect_library(MLIRShape
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Shape
  )
add_dependencies(MLIRShape MLIRShapeOpsIncGen LLVMSupport)
target_link_libraries(MLIRShape LLVMSupport)
