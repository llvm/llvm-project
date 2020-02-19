add_llvm_library(MLIRAffineOps
  AffineOps.cpp
  AffineValueMap.cpp
  EDSC/Builders.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AffineOps
  )
add_dependencies(MLIRAffineOps

  MLIRAffineOpsIncGen
  MLIREDSC
  MLIRIR
  MLIRLoopLikeInterfaceIncGen
  MLIRStandardOps
  )
target_link_libraries(MLIRAffineOps

  MLIREDSC
  MLIRIR
  MLIRStandardOps
  )
