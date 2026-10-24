add_mlir_library(MLIRXeGPUTestPasses
  TestXeGPUTransforms.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  mlir-headers
)

mlir_target_link_libraries(MLIRXeGPUTestPasses PUBLIC
  MLIRAffineUtils
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRUBDialect
  MLIRVectorDialect
  MLIRVectorToXeGPU
  MLIRXeGPUDialect
  MLIRPass
  MLIRTransforms
  MLIRGPUDialect
  MLIRXeGPUTransforms
)
