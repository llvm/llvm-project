add_mlir_library(MLIRXeGPUTestPasses
  TestXeGPUTransforms.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  mlir-headers
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
