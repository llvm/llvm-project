# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestPass
  TestDynamicPipeline.cpp
  TestPassManager.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Pass

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  )
