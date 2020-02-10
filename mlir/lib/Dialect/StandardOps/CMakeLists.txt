file(GLOB globbed *.c *.cpp)
add_llvm_library(MLIRStandardOps
  ${globbed}
  EDSC/Builders.cpp
  EDSC/Intrinsics.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/StandardOps
  )
add_dependencies(MLIRStandardOps

  MLIRCallOpInterfacesIncGen
  MLIREDSC
  MLIRIR
  MLIRStandardOpsIncGen
  LLVMSupport
  )
target_link_libraries(MLIRStandardOps

  MLIREDSC
  MLIRIR
  LLVMSupport
  )
