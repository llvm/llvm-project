# Exclude tests from libMLIR.so
add_mlir_library(MLIRXeGPUTestPasses
  TestXeGPUTransforms.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/XeGPU

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRXeGPUTransforms
  MLIRXeGPUDialect
  MLIRSupport
  )

