add_mlir_dialect_library(MLIRXeGPUDialect
  XeGPUDialect.cpp
  XeGPUOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mlir/Dialect/XeGPU

  DEPENDS
  MLIRXeGPUIncGen
  MLIRXeGPUAttrsIncGen
  MLIRXeGPUEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRViewLikeInterface
)
