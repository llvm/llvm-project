set(LIBS

  MLIREDSC
  MLIRIR
  MLIRLinalgOps
  MLIRLoopOps
  MLIRPass
  MLIRStandardOps
  MLIRTransformUtils
  )
  
add_llvm_library(MLIRLinalgUtils

  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg
  DEPENDS
  intrinsics_gen
  )

add_dependencies(MLIRLinalgUtils ${LIBS})
target_link_libraries(MLIRLinalgUtils ${LIBS})
