add_mlir_dialect_library(MLIRAVX512
  IR/AVX512Dialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AVX512

  DEPENDS
  MLIRAVX512IncGen
  )
target_link_libraries(MLIRAVX512
  PUBLIC
  MLIRIR
  MLIRSideEffects
  LLVMSupport
  )
