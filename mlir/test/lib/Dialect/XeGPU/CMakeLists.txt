add_mlir_dialect_library(MLIRXeGPUTestPasses
  TestXeGPUTransforms.cpp

  EXCLUDE_FROM_LIBMLIR
)

mlir_target_link_libraries(MLIRXeGPUTestPasses PUBLIC
  MLIRAffineUtils
  MLIRIR
  MLIRMemRefDialect
  MLIRXeGPUDialect
  MLIRPass
  MLIRTransforms
  MLIRGPUDialect
  MLIRXeGPUTransforms
)
