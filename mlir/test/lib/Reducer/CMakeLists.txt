# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestReducer
  MLIRTestReducer.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  LINK_COMPONENTS
  Core
  )
mlir_target_link_libraries(MLIRTestReducer PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  )
